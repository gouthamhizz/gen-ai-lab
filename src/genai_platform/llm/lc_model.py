"""LangChain adapter: makes our gateway a first-class LCEL chat model.

Why: LCEL chains (prompt | llm | parser), `.astream()`, and Langfuse's callback handler all expect
a `BaseChatModel`. Wrapping the gateway keeps retries/fallback/rate-limit logic in ONE place while
every chain gets streaming, callbacks and token-usage metadata for free.
"""
from __future__ import annotations

from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict

from genai_platform.llm.gateway import LLMGateway, Usage, get_gateway

_ROLE = {"human": "user", "ai": "assistant", "system": "system"}


def to_dicts(messages: List[BaseMessage]) -> list[dict]:
    return [{"role": _ROLE.get(m.type, "user"), "content": m.content if isinstance(m.content, str) else str(m.content)} for m in messages]


def _usage_md(u: Usage) -> dict:
    return {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens, "total_tokens": u.total}


class GatewayChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    gateway: Any = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None

    @property
    def _llm_type(self) -> str:
        return "genai-platform-gateway"

    def _gw(self) -> LLMGateway:
        return self.gateway or get_gateway()

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        r = self._gw().chat(to_dicts(messages), self.temperature, self.max_tokens, stop)
        msg = AIMessage(content=r.text, usage_metadata=_usage_md(r.usage),
                        response_metadata={"model_name": f"{r.provider}/{r.model}", "latency_s": r.latency_s})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        for ch in self._gw().stream(to_dicts(messages), self.temperature, self.max_tokens, stop):
            if ch.text:
                cg = ChatGenerationChunk(message=AIMessageChunk(content=ch.text))
                if run_manager:
                    run_manager.on_llm_new_token(ch.text, chunk=cg)
                yield cg
            if ch.usage:
                yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=_usage_md(ch.usage)))
