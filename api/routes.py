import json
from typing import Iterator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api import service

router = APIRouter(tags=["rag"])


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    model_id: Optional[str] = Field(
        None, description="如 deepseek:deepseek-flash；不传用默认模型"
    )
    reasoning: bool = Field(False, description="是否开启思考模式")
    collection: Optional[str] = Field(None, description="知识库名；不传用配置里的默认库")
    top_n: int = Field(service.DEFAULT_TOP_N, ge=1, le=20, description="检索条数")


def _frame(event: str, data) -> str:
    """
    拼一个 SSE 帧。
    data 走 JSON 而不是裸文本：token 里可能含换行，裸文本会破坏 SSE 的字段分隔。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/models", summary="可选的对话模型")
def get_models():
    return service.list_models()


@router.get("/collections", summary="可选的知识库")
def get_collections():
    return service.list_knowledge_bases()


@router.post("/chat", summary="一次性问答")
def chat(req: ChatRequest):
    try:
        return service.answer(
            req.query, req.model_id, req.reasoning, req.collection, req.top_n
        )
    except service.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/chat/stream", summary="流式问答（SSE）")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    事件流：先 sources，再若干 token，最后 done。
    注意错误也走事件（event: error），不是 HTTP 状态码——流一旦开始推送，
    状态码就已经发出去了，改不了。
    """
    def events() -> Iterator[str]:
        try:
            for event, data in service.stream_events(
                    req.query, req.model_id, req.reasoning, req.collection, req.top_n
            ):
                yield _frame(event, data)
        except (service.NotFound, ValueError) as exc:
            yield _frame("error", str(exc))
        finally:
            yield _frame("done", None)

    return StreamingResponse(events(), media_type="text/event-stream")
