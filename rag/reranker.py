import gc

import torch
from sentence_transformers import CrossEncoder

from config.settings import settings

_reranker = None


def get_reranker():
    """懒加载 CrossEncoder 单例（模型较大，不要每请求加载一次）。"""
    global _reranker
    if _reranker is None:
        if not settings.reranker_path:
            raise ValueError("未配置 reranker 模型：请在 .env 里填 RERANKER_PATH")
        _reranker = CrossEncoder(settings.reranker_path)
    return _reranker


def warmup():
    """加载模型并跑一次空推理，供启动时预热。"""
    return get_reranker().rank("warmup", ["warmup"])


def release():
    """释放重排模型。

    进程退出时操作系统本来就会回收内存，这里是为了让释放时机确定；而如果模型跑在
    显卡上，torch 的缓存分配器不会主动把显存还回去，必须显式 empty_cache。
    """
    global _reranker
    _reranker = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def rerank(query, rows, top_n=None):
    """对融合后的文档做交叉编码重排。

    :param query: 查询的问题
    :param rows: [{id, doc, meta, ...}]，通常来自 fusion.reciprocal_rank_fusion()。
    :param top_n: 只取前 N 条；None 表示全部。
    :return: [{id, doc, meta, ..., rerank_score}]，按 rerank_score 降序。
             原来的字段（如 rrf_score）会保留，便于回溯来源。

    rerank_score 用 Sigmoid 压到 0~1，便于做阈值过滤。注意模型自身的默认
    activation_fn 是 Identity（原始 logits，正数即相关），若更想要那种可解释
    的符号语义，把下面的 activation_fn 去掉即可。
    """
    if not rows:
        return []
    ranks = get_reranker().rank(
        query,
        [row["doc"] for row in rows],
        top_k=top_n,
        return_documents=False,
        activation_fn=torch.nn.Sigmoid(),
    )
    return [{**rows[item["corpus_id"]], "rerank_score": item["score"]} for item in ranks]


if __name__ == '__main__':
    print(warmup())
