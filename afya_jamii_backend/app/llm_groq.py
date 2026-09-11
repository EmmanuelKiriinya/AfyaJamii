"""Groq-backed language model client.

The model name, temperature, token budget, and timeouts all come from the
environment (see ``app.config``), and the prompt itself is read from a JSON
file under ``app/prompts`` (see ``app.prompt_loader``). Nothing in this module
hard-codes either.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from app.config import settings
from app.prompt_loader import Prompt, PromptError, PromptLibrary

logger = logging.getLogger(__name__)

# Shown to users when the upstream model is unreachable. Deliberately plain:
# it must never read as clinical advice.
UNAVAILABLE_MESSAGE = (
    "I can't reach the advice service right now. Please try again in a few minutes. "
    "If this is urgent, contact your nearest health facility, or call 1199 "
    "(Kenya Red Cross) or 999."
)


class LLMUnavailableError(RuntimeError):
    """Raised when the language model cannot serve a request."""


@dataclass(frozen=True)
class LLMStatus:
    """A snapshot of the client's health, for the /health endpoint."""

    ready: bool
    model: str
    prompt_id: Optional[str] = None
    prompt_version: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ready": self.ready, "model": self.model}
        if self.prompt_id:
            payload["prompt"] = f"{self.prompt_id}@{self.prompt_version}"
        if self.error:
            payload["error"] = self.error
        return payload


class AfyaJamiiLLM:
    """Wraps the Groq chat model together with the clinical prompt."""

    def __init__(self, library: PromptLibrary | None = None) -> None:
        self._library = library or PromptLibrary(
            settings.prompt_directory,
            reload_on_change=settings.PROMPT_RELOAD_ON_CHANGE,
        )
        self._llm: Optional[ChatGroq] = None
        self._prompt: Optional[Prompt] = None
        self._chain = None
        self._error: Optional[str] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """Build the model client and chain. Returns True when usable.

        Failures are logged and recorded rather than raised: the rest of the
        API — authentication, vitals capture, risk scoring, history — stays
        available even when the advice service is down.
        """
        try:
            self._prompt = self._library.get(settings.PROMPT_NAME)

            self._llm = ChatGroq(
                model=settings.LLM_MODEL_NAME,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                timeout=settings.LLM_TIMEOUT_SECONDS,
                max_retries=settings.LLM_MAX_RETRIES,
                api_key=settings.GROQ_API_KEY,
            )

            template = PromptTemplate(
                template=self._prompt.template,
                input_variables=list(self._prompt.input_variables),
            )
            self._chain = template | self._llm | StrOutputParser()

            self._error = None
            logger.info(
                "Language model ready: %s (prompt %s v%s, temperature %.2f)",
                settings.LLM_MODEL_NAME,
                self._prompt.id,
                self._prompt.version,
                settings.LLM_TEMPERATURE,
            )
            return True

        except PromptError as exc:
            self._error = f"prompt error: {exc}"
            logger.error("Cannot load prompt for the language model: %s", exc)
        except Exception as exc:  # noqa: BLE001 - start-up must not crash the API
            self._error = str(exc)
            logger.exception("Failed to initialise the Groq client")

        self._llm = None
        self._chain = None
        return False

    # ── Inspection ─────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._chain is not None

    def status(self) -> LLMStatus:
        return LLMStatus(
            ready=self.is_ready,
            model=settings.LLM_MODEL_NAME,
            prompt_id=self._prompt.id if self._prompt else None,
            prompt_version=self._prompt.version if self._prompt else None,
            error=self._error,
        )

    # ── Inference ──────────────────────────────────────────────────────────

    async def generate_advice(self, values: Mapping[str, Any]) -> str:
        """Run the clinical prompt and return the model's reply.

        Raises:
            LLMUnavailableError: if the client is not ready or the call fails.
        """
        if self._chain is None:
            raise LLMUnavailableError("The language model is not initialised")

        assert self._prompt is not None  # guaranteed once the chain exists
        payload = {name: values.get(name, "") for name in self._prompt.input_variables}

        try:
            response = await self._chain.ainvoke(payload)
        except Exception as exc:  # noqa: BLE001 - upstream errors vary widely
            logger.exception("Language model request failed")
            raise LLMUnavailableError(str(exc)) from exc

        text = (response or "").strip()
        if not text:
            raise LLMUnavailableError("The language model returned an empty response")
        return text


afya_llm = AfyaJamiiLLM()


def initialize_llm_service() -> bool:
    """Initialise the shared client. Called once during application start-up."""
    return afya_llm.initialize()
