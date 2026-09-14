import gc
import ipaddress
import re

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction, SentenceTransformerEmbeddingFunction
from config.settings import settings


def _rows(ids, documents, metadatas, distances=None):
    """把 chromadb 的平行数组拼成行字典，顺带把嵌套的 [0] 层拆掉。"""
    rows = []
    for index, doc_id in enumerate(ids):
        row = {"id": doc_id, "doc": documents[index], "meta": metadatas[index]}
        if distances is not None:
            row["distance"] = distances[index]
        rows.append(row)
    return rows


COLLECTION_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$")


def _is_ip_address(name):
    """chromadb 不接受 IP 地址形式的库名。

    用标准库判定而不是手写正则：999.1.1.1 这类不是合法 IP，chromadb 是放行的，
    粗糙的 \\d+.\\d+.\\d+.\\d+ 正则会误拦。
    """
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


def validate_collection_name(name):
    """
    校验库名。库名来自外部输入（对话中选库、用户建库），属于系统边界，
    提前拦下并给出可读报错，而不是让 chromadb 抛一串英文 ValidationError。
    规则与 chromadb 1.5 完全一致：3~512 个字符、仅允许字母/数字/下划线/连字符/点、
    首尾必须是字母或数字、不能含连续的点、不能是 IP 地址形式。
    """
    if not isinstance(name, str):
        raise ValueError(f"库名必须是字符串，收到 {type(name).__name__}: {name!r}")
    if not 3 <= len(name) <= 512:
        raise ValueError(f"库名长度必须是 3~512 个字符，当前 {len(name)} 个：{name!r}")
    if not COLLECTION_NAME_PATTERN.match(name):
        raise ValueError(
            f"库名非法：{name!r}。只允许字母、数字、下划线、连字符、点，"
            "且必须以字母或数字开头和结尾（不支持中文，中文名请另建 ASCII 名做映射）"
        )
    if ".." in name:
        raise ValueError(f"库名不能含连续的点：{name!r}")
    if _is_ip_address(name):
        raise ValueError(f"库名不能是 IP 地址形式：{name!r}")
    return name


class ChromaDB:
    """单个 collection 的句柄，只管 CRUD，生命周期由 ChromaDBManager 负责。"""

    def __init__(self, collection_name, client, embedding_function, create_if_missing=False):
        self.client = client
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.collection = self._open(create_if_missing)

    def _open(self, create_if_missing):
        if create_if_missing:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
        return self.client.get_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def upsert(self, ids, documents, metadatas=None):
        """存在则覆盖、不存在则插入，重复执行不会报错。"""
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def delete(self, ids):
        """按 id 删除。"""
        self.collection.delete(ids=ids)

    def delete_by_filter(self, where):
        """按元数据条件删除，例如 {"source": "a.pdf"}。多条件必须写以 $and 包裹
            {"$and": [{"source": "a.pdf"}, {"type": "XX"}]}命中多少删多少"""
        self.collection.delete(where=where)

    def update(self, ids, documents=None, metadatas=None):
        """只更新传入的字段；documents 变了会自动重算向量。"""
        self.collection.update(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query, n_results=5, where=None):
        """语义检索，返回按距离升序排列的 [{id, doc, meta, distance}]。"""
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return _rows(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )

    def get_all(self):
        """全量取出，供 BM25 建索引使用。"""
        data = self.collection.get(include=["documents", "metadatas"])
        return _rows(data["ids"], data["documents"], data["metadatas"])

    def get_by_ids(self, ids):
        """按 id 取回。"""
        data = self.collection.get(ids=ids, include=["documents", "metadatas"])
        return _rows(data["ids"], data["documents"], data["metadatas"])

    def count(self):
        """当前 collection 里的文档数。"""
        return self.collection.count()


class ChromaDBManager:
    """持有唯一 client，按名字分发各 collection 的句柄。多库场景的唯一入口。"""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        if settings.ollama_embedding_model:
            self.embedding_function = OllamaEmbeddingFunction(
                model_name=settings.ollama_embedding_model,
                url=settings.ollama_base_url,
            )
        elif settings.embedding_model_path:
            self.embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model_path,
            )
        else:
            raise ValueError(
                "未配置 embedding 模型：请在 .env 里填 OLLAMA_EMBEDDING_MODEL 或 EMBEDDING_MODEL_PATH"
            )

        self._handles = {}

    def get(self, name, create_if_missing=False):
        """取某个库的句柄。库名非法、或库不存在都会报错，避免静默建出意外的库。"""
        validate_collection_name(name)
        if name not in self._handles:
            self._handles[name] = ChromaDB(
                name, self.client, self.embedding_function, create_if_missing
            )
        return self._handles[name]

    def create(self, name):
        """显式建库（建库入库场景），已存在则直接返回。库名非法则报错。"""
        validate_collection_name(name)
        self._handles[name] = ChromaDB(
            name, self.client, self.embedding_function, create_if_missing=True
        )
        return self._handles[name]

    def list_names(self):
        """所有可选库名，供对话时选择或做白名单校验。"""
        return [collection.name for collection in self.client.list_collections()]

    def exists(self, name):
        """
        库是否存在。库名非法时返回 False——非法的名字不可能是已存在的库，
        这样拿它做白名单校验时不用套 try/except。"""
        try:
            validate_collection_name(name)
        except ValueError:
            return False
        return name in self.list_names()

    def drop(self, name):
        """删库，并清掉缓存句柄。库名非法则报错；库名合法但不存在会抛 NotFoundError。"""
        validate_collection_name(name)
        self.client.delete_collection(name)
        self._handles.pop(name, None)


_manager = None


def get_manager():
    """模块级单例，保证全进程只有一个 client 和一套 embedding function。"""
    global _manager
    if _manager is None:
        _manager = ChromaDBManager()
    return _manager


def get_collection(name=None):
    """取库句柄，默认用 settings.collection_name；库不存在会报错。"""
    return get_manager().get(name or settings.collection_name)


def create_collection(name=None):
    """建库，已存在则直接返回；默认用 settings.collection_name。"""
    return get_manager().create(name or settings.collection_name)


def collection_exists(name=None):
    """库是否存在，默认查 settings.collection_name。"""
    return get_manager().exists(name or settings.collection_name)


def list_collections():
    """列出所有库名。"""
    return get_manager().list_names()


def release():
    """释放 ChromaDB 客户端和 embedding function。

    本地 SentenceTransformer 模式下，embedding function 自己持有模型权重，
    释放它能真正把内存还回去；Ollama 模式只是个 HTTP 客户端，占不了多少。
    """
    global _manager
    _manager = None
    gc.collect()
