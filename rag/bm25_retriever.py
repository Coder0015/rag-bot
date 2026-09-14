import jieba
from rank_bm25 import BM25Okapi

from config.settings import settings
from rag.vector_store import get_collection


def _tokenize(text):
    """jieba 分词并统一小写。"""
    return [token.lower() for token in jieba.cut(text) if token.strip()]


class BM25Retriever:
    """基于某个知识库全量文档构建的 BM25 索引。"""
    """
    k1 词频饱和：越大，重复出现的词越持续加分；越小越快饱和、趋于"只算有没有出现"，k1=0 即二值化。常见 1.2~2.0。
    b 长度归一化：越大，长文档被惩罚越狠；b=0 完全不看长度（长文档占优），b=1 长短一视同仁。区间需要在0-1之间，常见 0.75。
    两者只在语料足够大时才有效：语料太小（如 N=2）时 rank_bm25 的 idf 会归零，分数恒为 0，调参无意义。
    """
    def __init__(self, ids, documents, metadatas=None, k1=1.2, b=0.75):

        self.ids = list(ids)
        self.documents = list(documents)
        self.metadatas = list(metadatas) if metadatas else [{}] * len(self.documents)
        # BM25Okapi 不接受空语料（会 ZeroDivisionError），空库时置 None
        self.bm25 = (
            BM25Okapi([_tokenize(doc) for doc in self.documents], k1=k1, b=b)
            if self.documents
            else None
        )

    def retrieve(self, query, top_n=5):
        """返回按 BM25 分数降序的 [{id, doc, meta, score}]；空库返回 []。"""
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda index: -scores[index])[:top_n]
        return [
            {
                "id": self.ids[index],
                "doc": self.documents[index],
                "meta": self.metadatas[index],
                "score": float(scores[index]),
            }
            for index in ranked
        ]


_cache = {}


def get_retriever(name=None):
    """按库名懒加载 BM25 索引，首次访问时从向量库全量取文档。"""
    name = name or settings.collection_name
    if name not in _cache:
        rows = get_collection(name).get_all()
        _cache[name] = BM25Retriever(
            [row["id"] for row in rows],
            [row["doc"] for row in rows],
            [row["meta"] for row in rows],
        )
    return _cache[name]


def refresh_bm25(name=None):
    """入库 / 删库后失效缓存。不传库名则清空全部。"""
    if name is None:
        _cache.clear()
    else:
        _cache.pop(name, None)


def release():
    """释放全部 BM25 索引缓存（关闭时调用）。"""
    refresh_bm25()
