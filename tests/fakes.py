"""Fakes for the true externals: the embedder and the chat model."""

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import Field


class FlakyEmbeddings(FakeEmbeddings):
    """A fake embedder that can be told to fail, as a real one does when it runs out of memory."""

    fail: bool = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return super().embed_documents(texts)


class BrokenChatModel(BaseChatModel):
    """A chat model that always fails, as it would when the machine runs out of memory."""

    @property
    def _llm_type(self) -> str:
        return "broken"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401  # Matches the BaseChatModel signature.
    ) -> ChatResult:
        raise RuntimeError("out of memory")


class RecordingChatModel(FakeListChatModel):
    """Replies like FakeListChatModel and keeps every message list it received."""

    received: list[list[BaseMessage]] = Field(default_factory=list[list[BaseMessage]])

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401  # Matches the FakeListChatModel signature.
    ) -> str:
        self.received.append(list(messages))
        return super()._call(messages, stop, run_manager, **kwargs)
