"""Thin wrapper over any OpenAI-compatible API: timeouts, retries, fallback provider, JSON repair."""
from __future__ import annotations

import json
import logging
import re
import time

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
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("no JSON object found")
        obj = json.loads(match.group(0))
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
        self._primary = self._make(settings.api_key, settings.base_url)
        has_fallback = settings.fallback_base_url and settings.fallback_model
        self._fallback = self._make(settings.fallback_api_key, settings.fallback_base_url) if has_fallback else None

    def _make(self, key: str, url: str):
        if not key:
            return None
        return OpenAI(api_key=key, base_url=url, timeout=self.s.llm_timeout, max_retries=0)

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

    def _call(self, client, model, system, user, max_tokens, json_mode, temperature) -> str:
        kwargs = dict(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last = "unknown error"
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(**kwargs)
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return text
                last = "empty response"
            except RateLimitError as e:
                last = "rate limit reached"
                time.sleep(_retry_after(e, attempt))
            except (APITimeoutError, APIConnectionError):
                last = f"timeout or connection error (limit {self.s.llm_timeout:.0f}s)"
                time.sleep(1 + attempt)
            except BadRequestError as e:
                if "response_format" in kwargs:  # model without JSON mode: retry without it
                    kwargs.pop("response_format")
                    last = "json mode not supported"
                    continue
                raise LLMError(f"bad request: {_msg(e)}") from e
            except APIStatusError as e:
                if e.status_code >= 500:
                    last = f"server error {e.status_code}"
                    time.sleep(1 + attempt)
                else:  # 401 bad key, 404 unknown model, ...: retrying will not help
                    raise LLMError(f"HTTP {e.status_code}: {_msg(e)}") from e
        raise LLMError(f"{last} after 3 attempts")

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
