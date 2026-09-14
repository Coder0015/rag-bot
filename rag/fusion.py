def reciprocal_rank_fusion(result_lists, k=60):
    """RRF（倒数排名融合）：把多路检索结果按名次融合成一路。

    公式为每个文档累加 1 / (k + rank)，rank 从 1 开始。
    只看名次、不看各路的绝对分数，所以向量距离和 BM25 分数量纲不同也没关系
    —— 这正是 RRF 适合混合检索的原因。
    k 是平滑常数，默认 60（原论文取值）：k 越大，靠前名次之间的差距越被抹平。

    :param result_lists: 多路结果，每路是 [{id, doc, meta, ...}]，已按相关性降序。
    :param k: 平滑常数。
    :return: [{id, doc, meta, rrf_score}]，按 rrf_score 降序。文档同时出现在多路时分数累加。
    """
    scores = {}
    rows = {}
    for results in result_lists:
        for rank, row in enumerate(results, start=1):
            doc_id = row["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            rows.setdefault(doc_id, row)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "id": doc_id,
            "doc": rows[doc_id]["doc"],
            "meta": rows[doc_id]["meta"],
            "rrf_score": score,
        }
        for doc_id, score in ranked
    ]


def fused_documents(result_lists, k=60):
    """RRF 融合后只取文档文本，可直接传给 rerank 模型。

    需要回填 id / meta（比如做引用展示）时，请自行保留 reciprocal_rank_fusion()
    的完整结果：CrossEncoder.rank() 返回的 corpus_id 就是这里列表的下标。
    另外 rank() 自带 top_k 参数，截断在那一步做即可，不必先切这个列表。
    """
    return [row["doc"] for row in reciprocal_rank_fusion(result_lists, k=k)]
