from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".pdf", ".json"}


def list_source_files(data_dir):
    """递归扫描数据目录，返回其中所有 txt / pdf / json 文件的路径（已排序）。"""
    root = Path(data_dir)
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
