import gc
from typing import Iterator, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from prompts.main_prompts import MainPrompts

DEEPSEEK = "deepseek"
OLLAMA = "ollama"


def list_available_models() -> list:
    """获取模型列表.env 里配了哪个模型就出现哪个，没配的不出现。"""
    pairs = []
    if settings.deepseek_model:
        pairs.append((DEEPSEEK, settings.deepseek_model))
    if settings.ollama_chat_model:
        pairs.append((OLLAMA, settings.ollama_chat_model))
    return [
        {"id": f"{provider}:{model}", "provider": provider, "model": model}
        for provider, model in pairs
    ]


def _build(provider: str, model: str, reasoning: bool):
    """构造 chat model。
    两个 provider 控制思考的机制不一样，所以分开写：
      - DeepSeek: extra_body 里的 thinking.type（官方文档）
      - Ollama:   reasoning=True/False
    """
    if provider == DEEPSEEK:
        return init_chat_model(
            model=model,
            model_provider=DEEPSEEK,
            api_key=settings.deepseek_api_key,
            extra_body={"thinking": {"type": "enabled" if reasoning else "disabled"}},
        )

    if provider == OLLAMA:
        return init_chat_model(
            model=model,
            model_provider=OLLAMA,
            base_url=settings.ollama_base_url,
            reasoning=reasoning,
        )

    raise ValueError(f"未知的模型提供商：{provider!r}（只支持 {DEEPSEEK} / {OLLAMA}）")


class ChatModel:
    def __init__(self, provider: str, model: str, reasoning: bool = False):
        self.provider = provider
        self.model_name = model
        self.reasoning = reasoning
        self.model = _build(provider, model, reasoning)
        # 以后要接 agent，在这里 create_agent(model=self.model) 即可

    @staticmethod
    def _messages(question: str, context: str) -> list:
        return [
            SystemMessage(content=MainPrompts.system_prompt),
            HumanMessage(
                content=MainPrompts.default_chat_prompt.format(
                    question=question, context=context
                )
            ),
        ]

    def simple_chat(self, question: str, context: str) -> str:
        """一次性对话"""
        return self.model.invoke(self._messages(question, context)).content

    def stream_chat(self, question: str, context: str) -> Iterator[tuple]:
        """
        流式对话，逐段产出 (类型, 文本)。
        两个 provider 都把思考放在 chunk.additional_kwargs["reasoning_content"]。
        """
        for chunk in self.model.stream(self._messages(question, context)):
            thought = (chunk.additional_kwargs or {}).get("reasoning_content")
            if thought:
                yield "reasoning", thought
            if chunk.content:
                yield "content", chunk.content


_chat_models = {}


def get_chat_model(model_id: Optional[str] = None, reasoning: bool = False) -> ChatModel:
    """
    按 (model_id, 是否思考) 取模型，带缓存。
    两个 provider 的思考开关不是同一个参数，所以实例必须按开关分别缓存，否则「开思考」和「关思考」会拿到同一个模型实例。
    """
    available = list_available_models()
    if not available:
        raise ValueError(
            "没有可用模型：请在 .env 里配置 DEEPSEEK_MODEL 或 OLLAMA_CHAT_MODEL"
        )

    if model_id is None:
        model_id = available[0]["id"]
    elif model_id not in {item["id"] for item in available}:
        raise ValueError(
            f"不支持的模型：{model_id!r}，可选 {[item['id'] for item in available]}"
        )

    cache_key = (model_id, reasoning)
    if cache_key not in _chat_models:
        # 只切第一个冒号：ollama 的模型名本身带冒号
        provider, model = model_id.split(":", 1)
        _chat_models[cache_key] = ChatModel(provider, model, reasoning)
    return _chat_models[cache_key]


def release():
    """释放缓存的聊天模型实例。

    每个实例内部持有 httpx 连接池，清掉引用后由 GC 回收。DeepSeek 是远程调用，
    Ollama 也是独立服务，所以这里没有本地模型权重，占的资源不大——但保持和其他
    资源一致的释放语义。
    """
    _chat_models.clear()
    gc.collect()


if __name__ == "__main__":
    print("可用模型：", list_available_models())
    print(get_chat_model().simple_chat(question="你好", context="这是一段测试上下文"))
