from typing import Iterator, Optional

from chromadb.errors import NotFoundError

from rag.generator import list_available_models
from rag.pipeline import Pipeline, sources_of
from rag.vector_store import list_collections

DEFAULT_TOP_N = 5


class NotFound(ValueError):
    """请求的资源不存在，比如知识库名写错了。"""


def list_models() -> list:
    """可选的对话模型。"""
    return list_available_models()


def list_knowledge_bases() -> list:
    """可选的知识库。"""
    return list_collections()


def answer(
    query: str,
    model_id: Optional[str] = None,
    reasoning: bool = False,
    collection: Optional[str] = None,
    top_n: int = DEFAULT_TOP_N,
) -> dict:
    """一次性问答，返回 {answer, sources}。"""
    try:
        return Pipeline().answer(
            query,
            model_id=model_id,
            reasoning=reasoning,
            collection_name=collection,
            top_n=top_n,
        )
    except NotFoundError as exc:
        raise NotFound(f"知识库不存在：{collection}") from exc


def stream_events(
    query: str,
    model_id: Optional[str] = None,
    reasoning: bool = False,
    collection: Optional[str] = None,
    top_n: int = DEFAULT_TOP_N,
) -> Iterator[tuple]:
    """流式问答。事件顺序：

        ("sources", 来源列表)      先发来源是刻意的，前端可以先渲染引用
        ("reasoning", 思考片段)    只在开了思考模式时出现，可能有很多条
        ("token", 正文片段)        正文，逐段流

    注意 reasoning 和 token 会**交错**：模型可能先想一段、写一段、再想一段。
    """
    pipeline = Pipeline()
    try:
        hits = pipeline.retrieve(query, collection_name=collection, top_n=top_n)
    except NotFoundError as exc:
        raise NotFound(f"知识库不存在：{collection}") from exc

    yield "sources", sources_of(hits)
    for kind, text in pipeline.stream_answer(
        query, model_id=model_id, reasoning=reasoning, hits=hits
    ):
        yield ("reasoning" if kind == "reasoning" else "token"), text
