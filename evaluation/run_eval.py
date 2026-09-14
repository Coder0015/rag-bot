"""跑评测：对比三种检索配置在固定测试集上的命中率。

用法：
    python evaluation/run_eval.py              # 用已有评测库（没有就自动建）
    python evaluation/run_eval.py --rebuild    # 先删库重建（语料变了之后用）

指标：
    hit@k    前 k 条里至少命中一条标准答案的比例
    MRR      标准答案排在第几位的倒数的均值——关心"排得多靠前"
    ms       平均单次检索耗时

注意：测试集里每个问题只有一条标准答案，所以 hit@k 就等于 recall@k。
真实场景下同一个问题可能有多个合理答案，这里的指标应当看作**下界**。
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.bootstrap import bootstrap
from rag.bm25_retriever import get_retriever
from rag.fusion import reciprocal_rank_fusion
from rag.pipeline import RERANK_POOL, RETRIEVAL_POOL
from rag.reranker import rerank
from rag.vector_store import get_collection, get_manager

EVAL_KB = "eval_kb"
DATASET = Path(__file__).resolve().parent / "dataset.jsonl"
KS = (3, 5)


def load_dataset():
    with open(DATASET, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# 三档都用 rag/pipeline.py 里的同一套参数，测的就是线上实际行为
def retrieve_vector(query, k):
    rows = get_collection(EVAL_KB).search(query, n_results=RETRIEVAL_POOL)
    return [r["id"] for r in rows[:k]]


def retrieve_fused(query, k):
    vec = get_collection(EVAL_KB).search(query, n_results=RETRIEVAL_POOL)
    bm25 = get_retriever(EVAL_KB).retrieve(query, top_n=RETRIEVAL_POOL)
    return [r["id"] for r in reciprocal_rank_fusion([vec, bm25])[:k]]


def retrieve_full(query, k):
    vec = get_collection(EVAL_KB).search(query, n_results=RETRIEVAL_POOL)
    bm25 = get_retriever(EVAL_KB).retrieve(query, top_n=RETRIEVAL_POOL)
    fused = reciprocal_rank_fusion([vec, bm25])
    return [r["id"] for r in rerank(query, fused[:RERANK_POOL], top_n=k)]


CONFIGS = [
    ("① 纯向量", retrieve_vector),
    ("② 向量+BM25+RRF", retrieve_fused),
    ("③ 全量（+ranker）", retrieve_full),
]


def evaluate(dataset, retrieve):
    """逐条算命中情况，返回总指标和按时类型拆分的指标。"""
    stats = defaultdict(lambda: {"n": 0, "hits": {k: 0 for k in KS}, "rr": 0.0})

    for case in dataset:
        expected = set(case["expected_ids"])
        max_k = max(KS)
        t = time.time()
        ids = retrieve(case["query"], max_k)
        elapsed = time.time() - t

        rank = next((i + 1 for i, x in enumerate(ids) if x in expected), None)

        for bucket in ("all", case["type"]):
            s = stats[bucket]
            s["n"] += 1
            s["ms"] = s.get("ms", 0.0) + elapsed * 1000
            if rank:
                s["rr"] += 1 / rank
                for k in KS:
                    if rank <= k:
                        s["hits"][k] += 1

    out = {}
    for bucket, s in stats.items():
        n = s["n"]
        out[bucket] = {
            "n": n,
            "ms": s["ms"] / n,
            "mrr": s["rr"] / n,
            **{f"hit@{k}": s["hits"][k] / n for k in KS},
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="先删库重建")
    args = parser.parse_args()

    if args.rebuild:
        manager = get_manager()
        if manager.exists(EVAL_KB):
            manager.drop(EVAL_KB)
            print(f"已删除旧评测库 {EVAL_KB}")

    result = bootstrap(collection_name=EVAL_KB)
    if result["created"]:
        print(f"已建评测库：入库 {result['files']} 个文件 / {result['chunks']} 个片段")
    else:
        print(f"复用已有评测库 {EVAL_KB}（语料变了请加 --rebuild）")

    dataset = load_dataset()
    print(f"测试集：{len(dataset)} 条\n")

    # 每个配置只跑一次，两张表共用结果（重排那档跑一遍就要几分钟）
    results = {}
    for name, fn in CONFIGS:
        print(f"  跑 {name} ...", flush=True)
        results[name] = evaluate(dataset, fn)

    print(f"\n{'配置':<20} {'hit@3':>8} {'hit@5':>8} {'MRR':>8} {'ms':>8}")
    print("-" * 56)
    for name, _ in CONFIGS:
        r = results[name]["all"]
        print(f"{name:<20} {r['hit@3']:>8.1%} {r['hit@5']:>8.1%} {r['mrr']:>8.4f} {r['ms']:>8.0f}")

    print("\n按时类型拆分（hit@3 / MRR）：")
    print(f"{'配置':<20} {'语义类':>18} {'故障码类':>18}")
    print("-" * 58)
    for name, _ in CONFIGS:
        sym, code = results[name]["symptom"], results[name]["code"]
        print(f"{name:<20} {sym['hit@3']:>9.1%}/{sym['mrr']:.3f} {code['hit@3']:>9.1%}/{code['mrr']:.3f}")


if __name__ == "__main__":
    main()
