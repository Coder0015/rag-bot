from pathlib import Path

from config.settings import settings
from ingestion.ingest import ingest_pdf, ingest_structured, ingest_text
from ingestion.loader import list_source_files
from rag.vector_store import collection_exists, create_collection


def bootstrap(data_dir=None, collection_name=None):
    """首次启动引导：默认库不存在时建库，并把数据目录下的 txt / pdf / json 全部入库。
    库已存在则整体跳过，不会重复入库。返回本次执行摘要。
    """
    data_dir = Path(data_dir or settings.data_dir)
    collection_name = collection_name or settings.collection_name

    if collection_exists(collection_name):
        return {"created": False, "collection": collection_name, "files": 0, "chunks": 0}

    data_dir.mkdir(parents=True, exist_ok=True)
    create_collection(collection_name)

    files = list_source_files(data_dir)
    chunks = 0
    for path in files:
        content = path.read_bytes()
        source = str(path.relative_to(data_dir))
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            chunks += ingest_pdf(content, source, collection_name)
        elif suffix == ".json":
            # ingest_structured 用 JSON 自身的 id 做标识，不接收 source
            chunks += ingest_structured(content, collection_name)
        else:
            chunks += ingest_text(content, source, collection_name)

    return {"created": True, "collection": collection_name, "files": len(files), "chunks": chunks}
