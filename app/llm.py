"""Thin wrapper over any OpenAI-compatible API: timeouts, retries, fallback provider, JSON repair."""
from __future__ import annotations

import json
import logging
import re
import time
import random

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from .config import Settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    """A model call failed in a way the user should be told about."""


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and extra prose."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as original_error:
        decoder = json.JSONDecoder()

        for index, char in enumerate(text):
            if char != "{":
                continue

            try:
                obj, _ = decoder.raw_decode(text[index:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise ValueError("no valid JSON object found") from original_error

    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object")

    return obj


def _msg(e: Exception) -> str:
    return str(getattr(e, "message", e))[:200]


def _retry_after(e: Exception, attempt: int) -> float:
    try:
        wait = float(e.response.headers.get("retry-after", ""))  # type: ignore[attr-defined]
    except (TypeError, ValueError, AttributeError):
        wait = 2 ** (attempt + 1)
    return min(max(wait, 1.0), 20.0)


class LLMClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self._primary = self._make(
            settings.api_key,
            settings.base_url,
            settings.llm_timeout,
        )

        has_fallback = settings.fallback_base_url and settings.fallback_model

        self._fallback = (
            self._make(
                settings.fallback_api_key,
                settings.fallback_base_url,
                settings.fallback_timeout,
            )
            if has_fallback
            else None
        )

    def _make(self, key: str, url: str, timeout: float):
        if not key:
            return None

        return OpenAI(
            api_key=key,
            base_url=url,
            timeout=timeout,
            max_retries=0,
        )

    @property
    def configured(self) -> bool:
        return self._primary is not None

    def chat_text(self, role: str, system: str, user: str, max_tokens: int = 1500,
                  json_mode: bool = False, temperature: float = 0.2) -> str:
        if self._primary is None:
            raise LLMError("LLM_API_KEY is not set on the server.")
        targets = [(self._primary, self.s.model_for(role))]
        if self._fallback is not None:
            targets.append((self._fallback, self.s.fallback_model))
        errors = []
        for client, model in targets:
            try:
                return self._call(client, model, system, user, max_tokens, json_mode, temperature)
            except LLMError as e:
                log.warning("model %s failed for role %s: %s", model, role, e)
                errors.append(f"{model}: {e}")
        raise LLMError(" | ".join(errors))

    def _call(
        self,
        client,
        model,
        system,
        user,
        max_tokens,
        json_mode,
        temperature,
    ) -> str:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        max_attempts = 3
        last_error = "unknown error"

        for attempt in range(max_attempts):
            try:
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                content = content.strip()

                if content:
                    return content

                last_error = "empty response"

            except RateLimitError as e:
                # No insistir inmediatamente contra el mismo límite.
                raise LLMError(
                    f"rate limit reached: {_msg(e)}"
                ) from e

            except BadRequestError as e:
                detail = _msg(e)

                # Intentar sin JSON mode únicamente si el error
                # parece estar relacionado con esa opción.
                if (
                    "response_format" in kwargs
                    and any(
                        term in detail.lower()
                        for term in (
                            "response_format",
                            "json_object",
                            "json mode",
                            "structured output",
                        )
                    )
                ):
                    kwargs.pop("response_format")
                    last_error = f"JSON mode unsupported: {detail}"
                    continue

                raise LLMError(
                    f"bad request: {detail}"
                ) from e

            except (APITimeoutError, APIConnectionError) as e:
                last_error = (
                    f"connection/timeout error: {type(e).__name__}"
                )

            except APIStatusError as e:
                if e.status_code < 500:
                    raise LLMError(
                        f"HTTP {e.status_code}: {_msg(e)}"
                    ) from e

                last_error = f"server error {e.status_code}"

            # No esperar después del último intento.
            if attempt < max_attempts - 1:
                delay = min(2 ** attempt, 8)
                time.sleep(delay + random.uniform(0, 0.5))

        raise LLMError(
            f"{last_error} after {max_attempts} attempts"
        )

    def chat_json(self, role: str, system: str, user: str, required=(), max_tokens: int = 1200) -> dict:
        raw = self.chat_text(role, system, user, max_tokens=max_tokens, json_mode=True, temperature=0)
        for attempt in range(2):
            try:
                obj = parse_json(raw)
                missing = [k for k in required if k not in obj]
                if missing:
                    raise ValueError(f"missing keys {missing}")
                return obj
            except ValueError as e:
                if attempt == 1:
                    raise LLMError(f"malformed JSON from {role}: {e}") from e
                repair = f"{user}\n\nYour previous reply was invalid ({e}). Reply again with ONLY the JSON object."
                raw = self.chat_text(role, system, repair, max_tokens=max_tokens, json_mode=True, temperature=0)
        raise LLMError("unreachable")
