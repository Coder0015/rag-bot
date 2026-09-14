from typing import Iterator

from config.settings import settings
from rag.bm25_retriever import get_retriever
from rag.fusion import reciprocal_rank_fusion
from rag.generator import get_chat_model
from rag.reranker import rerank
from rag.vector_store import get_collection

RETRIEVAL_POOL = 20   # 向量和 BM25 各取多少条候选去融合。太小的话 RRF 没有融合空间
RERANK_POOL = 10      # 融合后送进精排多少条。精排耗时正比于条数（约 180ms/条），
                      # 不设上限的话 40 条候选要跑 7 秒
DEFAULT_TOP_N = 5     # 最终返回几条给模型


def _source_of(hit) -> str:
    """取来源标识：txt/pdf 用 source，json 用 category，都没有就退回 id。"""
    meta = hit.get("meta") or {}
    return meta.get("source") or meta.get("category") or hit["id"]


def build_context(hits) -> str:
    """把检索结果编号拼成给模型看的上下文。"""
    blocks = [
        f"[{index}]（来源：{_source_of(hit)}）\n{hit['doc']}"
        for index, hit in enumerate(hits, start=1)
    ]
    return "\n\n".join(blocks)


def _score_of(hit):
    """取该行的相关度分数。

    不同阶段产出的字段不一样（精排 rerank_score、融合 rrf_score、向量 distance），
    量纲也不统一，只适合当调试参考，别拿来跨阶段比较。
    """
    for key in ("rerank_score", "rrf_score", "distance"):
        if key in hit:
            return round(float(hit[key]), 4)
    return None


def sources_of(hits) -> list:
    """给前端用的检索结果，带片段全文，供调试面板展示。"""
    return [
        {
            "id": hit["id"],
            "source": _source_of(hit),
            "score": _score_of(hit),
            "text": hit["doc"],
        }
        for hit in hits
    ]


class Pipeline:
    """编排：检索 → 拼上下文 → 生成。RAG 的唯一对外入口。
    不持有模型实例——model_id 每次调用时传，方便切模型。
    """

    def retrieve(self, query, collection_name=None, top_n=DEFAULT_TOP_N):
        """混合检索：向量 + BM25 → RRF 融合 → CrossEncoder 重排。

        两路各取 RETRIEVAL_POOL 条去融合（池子小了 RRF 没空间），
        融合结果截到 RERANK_POOL 条送精排，最后返回 top_n 条。

        重排依赖 RERANKER_PATH，没配就只返回融合结果——和 main.py 的预热逻辑一致，
        这样没配重排的环境也能正常跑。返回 [{id, doc, meta, ...}]。
        """
        vec_hits = get_collection(collection_name).search(query, n_results=RETRIEVAL_POOL)
        bm25_hits = get_retriever(collection_name).retrieve(query, top_n=RETRIEVAL_POOL)
        fused = reciprocal_rank_fusion([vec_hits, bm25_hits])

        if not settings.reranker_path:
            return fused[:top_n]
        return rerank(query, fused[:RERANK_POOL], top_n=top_n)

    def answer(
        self,
        query,
        model_id=None,
        reasoning=False,
        collection_name=None,
        top_n=DEFAULT_TOP_N,
    ) -> dict:
        """一次性回答，返回 {answer, sources}。"""
        hits = self.retrieve(query, collection_name, top_n)
        text = get_chat_model(model_id, reasoning).simple_chat(
            query, build_context(hits)
        )
        return {"answer": text, "sources": sources_of(hits)}

    def stream_answer(
        self,
        query,
        model_id=None,
        reasoning=False,
        hits=None,
        collection_name=None,
        top_n=DEFAULT_TOP_N,
    ) -> Iterator[str]:
        """流式回答。已检索过就把 hits 传进来，避免重复检索一次。"""
        if hits is None:
            hits = self.retrieve(query, collection_name, top_n)
        yield from get_chat_model(model_id, reasoning).stream_chat(
            query, build_context(hits)
        )
