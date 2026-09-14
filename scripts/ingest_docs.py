"""把数据目录（默认 data/）下的 txt / pdf 入库。
用法：
    python scripts/ingest_docs.py                # 默认库不存在则创建并入库
    python scripts/ingest_docs.py my_collection  # 指定库名
    python scripts/ingest_docs.py --force        # 先删库再重建
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from ingestion.bootstrap import bootstrap
from rag.vector_store import get_manager


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    collection_name = args[0] if args else None
    force = "--force" in sys.argv

    if force:
        manager = get_manager()
        name = collection_name or settings.collection_name
        if manager.exists(name):
            manager.drop(name)
            print(f"已删除旧库：{name}")

    result = bootstrap(collection_name=collection_name)
    if result["created"]:
        print(
            f"已创建知识库{result['collection']}："
            f"入库 {result['files']} 个文件，共 {result['chunks']} 个片段。"
        )
    else:
        print(f"知识库{result['collection']}已存在，未做改动。加 --force 可删库重建。")


if __name__ == "__main__":
    main()
