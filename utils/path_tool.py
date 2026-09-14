from pathlib import Path


def get_project_path() -> str:
    """返回项目根目录的绝对路径，与当前工作目录无关。"""
    return str(Path(__file__).resolve().parent.parent)


if __name__ == '__main__':
    print(get_project_path())
