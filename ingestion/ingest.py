import json
import uuid
import tempfile
from rag.bm25_retriever import refresh_bm25
from rag.vector_store import get_collection

CHUNK_SIZE = 300      # 每块的目标大小
CHUNK_OVERLAP = 50      # 每块之间的重叠部分


def ingest_structured(content: bytes, collection_name=None) -> int:
    """链1：结构化JSON（岗位数组），一条不切直接入库"""
    data = json.loads(content)
    # 兼容单条和数组
    items = data if isinstance(data, list) else [data]
    if not items:
        return 0
    ids = [it.get("id") or f"doc-{uuid.uuid4().hex[:8]}" for it in items]
    docs = [it["text"] for it in items]
    metas = [{k: v for k, v in it.items() if k not in ("id", "text")} for it in items]
    get_collection(collection_name).upsert(ids=ids, documents=docs, metadatas=metas)
    refresh_bm25(collection_name)
    return len(items)


def ingest_text(content: bytes, source: str, collection_name=None) -> int:
    """链2：纯文本，递归字符切分后入库"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text = content.decode("utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 中文文档的关键：定义分隔符优先级，尽量在句子/段落边界切
        separators=["\n\n", "\n。", "。", "\n", "，", " ", ""],
    )
    chunks = splitter.split_text(text)

    ids = [f"{source}-{i}" for i in range(len(chunks))]
    metas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    get_collection(collection_name).upsert(ids=ids, documents=chunks, metadatas=metas)
    refresh_bm25(collection_name)
    return len(chunks)


def ingest_pdf(content: bytes, source: str, collection_name=None) -> int:
    """链3：PDF，先解析再切分（依赖pypdf）"""
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(content)
        path = f.name
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    if not text.strip():
        raise ValueError("PDF文本提取为空，可能是扫描件")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_text(text)
    ids = [f"{source}-{i}" for i in range(len(chunks))]
    metas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    get_collection(collection_name).upsert(ids=ids, documents=chunks, metadatas=metas)
    refresh_bm25(collection_name)
    return len(chunks)
