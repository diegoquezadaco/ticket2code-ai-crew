"""OpenAI-compatible LLM client with shared Groq TPM protection and Ollama fallback."""
from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from collections import deque
from typing import Any
from urllib.parse import urlparse

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
    """A provider or response error that the pipeline can report cleanly."""


# ---------------------------------------------------------------------------
# Shared Groq rate-limit state
# Shared by all LLMClient instances within this Python process.
# ---------------------------------------------------------------------------

_GROQ_LOCK = threading.RLock()
_GROQ_CONDITION = threading.Condition(_GROQ_LOCK)

# Each entry is [monotonic timestamp, reserved token count].
_GROQ_USAGE: dict[str, deque[list[float]]] = {}
_GROQ_COOLDOWN_UNTIL: dict[str, float] = {}
_GROQ_SEMAPHORES: dict[str, threading.BoundedSemaphore] = {}

_WINDOW_SECONDS = 60.0



# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def parse_json(text: str) -> dict:
    """Parse a JSON object, tolerating Markdown fences or surrounding prose."""
    text = (text or "").strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        obj = json.loads(text)

    except json.JSONDecodeError as original_error:
        decoder = json.JSONDecoder()
        obj = None

        # Recover a JSON object embedded in surrounding text.
        for index, char in enumerate(text):
            if char != "{":
                continue

            try:
                candidate, _ = decoder.raw_decode(text[index:])

                if isinstance(candidate, dict):
                    obj = candidate
                    break

            except json.JSONDecodeError:
                continue

        if obj is None:
            raise ValueError("no valid JSON object found") from original_error

    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object")

    return obj


def _provider_error(e: Exception, limit: int = 800) -> str:
    """Return a concise provider error without logging prompts or credentials."""
    body: Any = getattr(e, "body", None)

    if body:
        try:
            detail = json.dumps(
                body,
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            detail = str(body)
    else:
        detail = str(getattr(e, "message", e))

    return detail[:limit]


def _msg(e: Exception) -> str:
    """Compatibility helper for concise provider error messages."""
    return _provider_error(e, limit=800)


def _retry_after_seconds(e: Exception, default: float) -> float:
    """Read Retry-After or the retry delay reported in the error message."""
    try:
        response = getattr(e, "response", None)

        if response is not None:
            raw = response.headers.get("retry-after", "")

            if raw:
                return min(max(float(raw), 1.0), 300.0)

    except (AttributeError, TypeError, ValueError):
        pass

    # Some Groq errors include: "Please try again in 20.32s."
    detail = _provider_error(e)

    match = re.search(
        r"try again in\s+(\d+(?:\.\d+)?)\s*s",
        detail,
        flags=re.IGNORECASE,
    )

    if match:
        try:
            seconds = float(match.group(1))
            return min(max(seconds, 1.0), 300.0)
        except ValueError:
            pass

    return default


def _is_groq_url(url: str | None) -> bool:
    """Return True when the configured endpoint belongs to Groq."""
    if not url:
        return False

    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return "groq.com" in url.lower()

    return host == "groq.com" or host.endswith(".groq.com")


def _estimate_tokens(system: str, user: str, max_tokens: int) -> int:
    """
    Estimate prompt plus completion tokens.

    This is a preventive approximation, not an exact tokenizer.
    The prompt estimate uses roughly four characters per token.
    """
    prompt_chars = len(system or "") + len(user or "")
    estimated_input = max(1, (prompt_chars + 3) // 4)

    # Reserve part of the requested completion allowance.
    # Increase this fraction if your models typically generate long responses.
    estimated_output = max(128, int(max_tokens * 0.50))

    return estimated_input + estimated_output


def _trim_groq_usage_locked(
    now: float,
    usage: deque[list[float]],
) -> None:
    """Remove reservations older than the rolling 60-second window."""
    cutoff = now - _WINDOW_SECONDS

    while usage and usage[0][0] <= cutoff:
        usage.popleft()


def _groq_set_cooldown(model_key: str, seconds: float) -> None:
    """Apply a cooldown to one Groq model without blocking other models."""
    seconds = max(60.0, min(float(seconds), 300.0))

    with _GROQ_CONDITION:
        _GROQ_COOLDOWN_UNTIL[model_key] = max(
            _GROQ_COOLDOWN_UNTIL.get(model_key, 0.0),
            time.monotonic() + seconds,
        )
        _GROQ_CONDITION.notify_all()

    log.warning(
        "Groq cooldown activated: model=%s duration_seconds=%.1f",
        model_key,
        seconds,
    )


def _groq_reserve(
    model_key: str,
    estimated_tokens: int,
    budget: int,
    max_wait_seconds: float,
) -> float:
    """
    Reserve estimated tokens in one model's rolling 60-second window.

    Wait for capacity to become available before reporting temporary
    budget exhaustion. Returns the reservation timestamp.
    """
    if budget <= 0:
        raise LLMError("Groq TPM budget must be greater than zero.")

    if estimated_tokens > budget:
        raise LLMError(
            f"Estimated request size exceeds the configured Groq TPM "
            f"budget for {model_key} "
            f"({estimated_tokens} > {budget}). "
            "Reduce the prompt or max_tokens."
        )

    deadline = time.monotonic() + max(0.0, max_wait_seconds)

    while True:
        now = time.monotonic()

        with _GROQ_CONDITION:
            usage = _GROQ_USAGE.setdefault(model_key, deque())
            _trim_groq_usage_locked(now, usage)

            cooldown_remaining = (
                _GROQ_COOLDOWN_UNTIL.get(model_key, 0.0) - now
            )

            wait_remaining = deadline - now

            if cooldown_remaining > 0:
                if wait_remaining <= 0:
                    raise LLMError(
                        f"Groq cooldown remains active for {model_key}; "
                        f"approximately {cooldown_remaining:.1f} seconds remain."
                    )

                _GROQ_CONDITION.wait(
                    timeout=min(
                        cooldown_remaining,
                        wait_remaining,
                        1.0,
                    )
                )
                continue

            used_tokens = sum(int(entry[1]) for entry in usage)

            if used_tokens + estimated_tokens <= budget:
                usage.append([now, float(estimated_tokens)])

                log.debug(
                    "Groq TPM reservation: model=%s estimated=%d "
                    "reserved=%d budget=%d",
                    model_key,
                    estimated_tokens,
                    used_tokens + estimated_tokens,
                    budget,
                )

                return now

            if wait_remaining <= 0:
                raise LLMError(
                    "Groq TPM budget temporarily exhausted for "
                    f"{model_key}: estimated used={used_tokens}, "
                    f"requested={estimated_tokens}, budget={budget}, "
                    f"waited up to {max_wait_seconds:.1f} seconds."
                )

            oldest_timestamp = usage[0][0] if usage else now

            until_window_frees = max(
                0.05,
                oldest_timestamp + _WINDOW_SECONDS - now,
            )

            _GROQ_CONDITION.wait(
                timeout=min(
                    until_window_frees,
                    wait_remaining,
                    1.0,
                )
            )


def _groq_adjust_reservation(
    model_key: str,
    reservation_timestamp: float,
    actual_tokens: int,
) -> None:
    """Replace a model's estimated reservation with actual usage."""
    with _GROQ_CONDITION:
        usage = _GROQ_USAGE.get(model_key, deque())

        for entry in usage:
            if entry[0] == reservation_timestamp:
                entry[1] = float(max(0, actual_tokens))
                break

        _GROQ_CONDITION.notify_all()

def _groq_semaphore_for(model_key: str) -> threading.BoundedSemaphore:
    """Return a request semaphore dedicated to one Groq model."""
    with _GROQ_LOCK:
        semaphore = _GROQ_SEMAPHORES.get(model_key)

        if semaphore is None:
            semaphore = threading.BoundedSemaphore(1)
            _GROQ_SEMAPHORES[model_key] = semaphore

        return semaphore


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, settings: Settings):
        self.s = settings

        self._primary = self._make(
            getattr(settings, "api_key", ""),
            getattr(settings, "base_url", ""),
            getattr(settings, "llm_timeout", 60),
        )

        fallback_url = getattr(settings, "fallback_base_url", "")
        fallback_model = getattr(settings, "fallback_model", "")

        has_fallback = bool(fallback_url and fallback_model)

        # Ollama's OpenAI-compatible endpoint commonly accepts any nonempty
        # API key. "ollama" is a placeholder, not a real credential.
        fallback_key = (
            getattr(settings, "fallback_api_key", None) or "ollama"
        )

        self._fallback = (
            self._make(
                fallback_key,
                fallback_url,
                getattr(settings, "fallback_timeout", 380),
            )
            if has_fallback
            else None
        )

        self._primary_is_groq = _is_groq_url(
            getattr(settings, "base_url", "")
        )

    def _make(self, key: str | None, url: str | None, timeout: float):
        """Create an OpenAI-compatible client with SDK retries disabled."""
        if not url:
            return None

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
        """Whether the primary provider is configured."""
        return self._primary is not None

    # -----------------------------------------------------------------------
    # Public text interface
    # -----------------------------------------------------------------------

    def chat_text(
        self,
        role: str,
        system: str,
        user: str,
        max_tokens: int = 1500,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        """
        Generate text using the primary provider and then the local fallback.

        Provider failures are isolated here so a Groq failure does not
        prevent the fallback from being attempted.
        """
        if self._primary is None:
            raise LLMError(
                "Primary LLM client is not configured. "
                "Check LLM_API_KEY and the primary base URL."
            )

        targets = [
            (
                self._primary,
                self.s.model_for(role),
                self._primary_is_groq,
                "primary",
            )
        ]

        if self._fallback is not None:
            targets.append(
                (
                    self._fallback,
                    self.s.fallback_model,
                    False,
                    "fallback",
                )
            )

        errors: list[str] = []

        for client, model, is_groq, provider_name in targets:
            try:
                result = self._call(
                    client=client,
                    model=model,
                    system=system,
                    user=user,
                    provider_name=provider_name,
                    role=role,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    temperature=temperature,
                    is_groq=is_groq,
                )

                log.debug(
                    "LLM request succeeded: provider=%s, role=%s, model=%s",
                    provider_name,
                    role,
                    model,
                )

                return result

            except LLMError as e:
                log.warning(
                    "LLM provider failed: provider=%s, role=%s, model=%s, "
                    "error=%s",
                    provider_name,
                    role,
                    model,
                    e,
                )

                errors.append(f"{provider_name}/{model}: {e}")

        raise LLMError(
            "All configured LLM providers failed. " + " | ".join(errors)
        )

    # -----------------------------------------------------------------------
    # Internal request implementation
    # -----------------------------------------------------------------------

    def _call(
        self,
        client,
        model: str,
        system: str,
        user: str,
        provider_name: str,
        role: str,
        max_tokens: int,
        json_mode: bool,
        temperature: float,
        is_groq: bool = False,
    ) -> str:
        """Send a request with retries, rate-limit protection and error handling."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }

        # Keep the output-token limit for the primary provider.
        # Let the local fallback generate without an application-level output cap.
        if provider_name != "fallback":
            kwargs["max_tokens"] = max_tokens
        # Reduce reasoning-token consumption for supported Groq models.
        if is_groq:
            if model.startswith("openai/gpt-oss-"):
                kwargs["extra_body"] = {
                    "reasoning_effort": "low",
                }

            elif model == "qwen/qwen3.8-27b":
                effort = (
                    "none"
                    if role in {"reviewer", "documentarian", "router"}
                    else "low"
                )

                kwargs["extra_body"] = {
                    "reasoning_effort": effort,
                }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Local requests should not spend several minutes retrying.
        # Groq also gets a bounded retry policy.
        max_attempts = 2 if is_groq else 1

        last_error = "unknown error"

        for attempt in range(max_attempts):
            reservation_timestamp: float | None = None
            groq_semaphore = _groq_semaphore_for(model) if is_groq else None
            semaphore_acquired = False

            try:
                if is_groq:
                    # Only one Groq request at a time in this process.
                    wait_seconds = float(
                        getattr(self.s, "groq_wait_seconds", 5)
                    )

                    semaphore_acquired = groq_semaphore.acquire(
                        timeout=max(0.0, wait_seconds)
                    )

                    if not semaphore_acquired:
                        raise LLMError(
                            "Timed out waiting for the shared Groq request slot."
                        )

                    budget_getter = getattr(self.s, "groq_tpm_budget_for", None)

                    if callable(budget_getter):
                        budget = int(budget_getter(model))
                    else:
                        budget = int(getattr(self.s, "groq_tpm_budget", 4500))

                    estimated_tokens = _estimate_tokens(
                        system=system,
                        user=user,
                        max_tokens=max_tokens,
                    )

                    reservation_timestamp = _groq_reserve(
                        model_key=model,
                        estimated_tokens=estimated_tokens,
                        budget=budget,
                        max_wait_seconds=wait_seconds,
                    )

                if not is_groq:
                    messages = kwargs.get("messages", [])
                    prompt_chars = sum(
                        len(str(message.get("content", "")))
                        for message in messages
                    )

                    log.info(
                        "Ollama request: model=%s, prompt_chars=%d, max_tokens=%s",
                        model,
                        prompt_chars,
                        kwargs.get("max_tokens"),
                    )
                    # log.info(
                    #     "Ollama response: model=%s finish_reason=%s "
                    #     "prompt_tokens=%s completion_tokens=%s",
                    #     model,
                    #     response.choices[0].finish_reason,
                    #     getattr(response.usage, "prompt_tokens", None),
                    #     getattr(response.usage, "completion_tokens", None),
                    # )

                request_started = time.perf_counter()

                messages = kwargs.get("messages", [])
                prompt_chars = sum(
                    len(str(message.get("content", "")))
                    for message in messages
                )

                try:
                    response = client.chat.completions.create(**kwargs)

                except Exception:
                    elapsed = time.perf_counter() - request_started

                    log.exception(
                        "LLM request failed: provider_name=%s role=%s model=%s "
                        "duration_seconds=%.2f prompt_chars=%d max_tokens=%s",
                        provider_name,
                        role,
                        model,
                        elapsed,
                        prompt_chars,
                        kwargs.get("max_tokens"),
                    )
                    raise

                elapsed = time.perf_counter() - request_started

                usage = getattr(response, "usage", None)

                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

                log.info(
                    "LLM request completed: provider=%s role=%s model=%s "
                    "duration_seconds=%.2f prompt_chars=%d max_tokens=%s "
                    "prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                    provider_name,
                    role,
                    model,
                    elapsed,
                    prompt_chars,
                    kwargs.get("max_tokens"),
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                )
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)

                choices = getattr(response, "choices", None) or []
                finish_reason = getattr(choices[0], "finish_reason", None) if choices else None

                if not is_groq:
                    log.info(
                        "Ollama response: model=%s finish_reason=%s "
                        "prompt_tokens=%s completion_tokens=%s",
                        model,
                        finish_reason,
                        prompt_tokens,
                        completion_tokens,
                    )

                content = response.choices[0].message.content or ""
                content = content.strip()

                if not content:
                    last_error = "empty response"

                    if attempt < max_attempts - 1:
                        time.sleep(0.5)
                        continue

                    break

                # If the provider reports actual token usage, update the
                # successful Groq reservation with that value.
                if is_groq and reservation_timestamp is not None:
                    usage = getattr(response, "usage", None)

                    if usage is not None:
                        actual_tokens = getattr(
                            usage,
                            "total_tokens",
                            None,
                        )

                        if actual_tokens is not None:
                            _groq_adjust_reservation(
                                model,
                                reservation_timestamp,
                                int(actual_tokens),
                            )
                log.info(
                    "LLM finish reason: role=%s model=%s finish_reason=%s",
                    role,
                    model,
                    response.choices[0].finish_reason,
                )
                return content

            except RateLimitError as e:
                detail = _msg(e)

                if is_groq:
                    default_cooldown = float(
                        getattr(self.s, "groq_cooldown_seconds", 60)
                    )

                    cooldown = _retry_after_seconds(
                        e,
                        default=default_cooldown,
                    )

                    _groq_set_cooldown(model,cooldown)

                raise LLMError(
                    f"HTTP 429 rate limit reached: {detail}"
                ) from e

            except BadRequestError as e:
                detail = _msg(e)
                lower_detail = detail.lower()

                # Some OpenAI-compatible servers do not support JSON mode.
                # Retry once without response_format if that is the problem.
                json_mode_terms = (
                    "response_format",
                    "json_object",
                    "json mode",
                    "structured output",
                    "json_validate_failed",
                    "failed to generate json",
                )

                if (
                    "response_format" in kwargs
                    and any(term in lower_detail for term in json_mode_terms)
                ):
                    kwargs.pop("response_format")

                    log.warning(
                        "Model %s rejected JSON mode; retrying without "
                        "response_format.",
                        model,
                    )

                    last_error = f"JSON mode unsupported: {detail}"
                    continue

                raise LLMError(
                    f"HTTP 400 bad request: {detail}"
                ) from e

            except (APITimeoutError, APIConnectionError) as e:
                last_error = (
                    f"{type(e).__name__}: {_msg(e)}"
                )

                log.warning(
                    "LLM connection/timeout: model=%s, attempt=%d/%d, "
                    "error=%s",
                    model,
                    attempt + 1,
                    max_attempts,
                    last_error,
                )

            except APIStatusError as e:
                detail = _msg(e)

                if e.status_code < 500:
                    raise LLMError(
                        f"HTTP {e.status_code}: {detail}"
                    ) from e

                last_error = (
                    f"HTTP {e.status_code}: {detail}"
                )

                log.warning(
                    "LLM server error: model=%s, attempt=%d/%d, error=%s",
                    model,
                    attempt + 1,
                    max_attempts,
                    last_error,
                )

            except LLMError:
                # Do not retry rate-limit guard/cooldown errors here.
                # chat_text() will move on to the fallback.
                raise

            except Exception as e:
                # Preserve unexpected failures as controlled provider errors.
                log.exception(
                    "Unexpected LLM error: model=%s",
                    model,
                )

                raise LLMError(
                    f"Unexpected {type(e).__name__}: {_msg(e)}"
                ) from e

            finally:
                if semaphore_acquired and groq_semaphore is not None:
                    groq_semaphore.release()

            # Backoff only for retryable errors; never wait after the last try.
            if attempt < max_attempts - 1:
                delay = min(2 ** attempt, 4)
                time.sleep(delay + random.uniform(0, 0.3))

        raise LLMError(
            f"{last_error} after {max_attempts} attempts"
        )

    # -----------------------------------------------------------------------
    # Public JSON interface
    # -----------------------------------------------------------------------

    def chat_json(
        self,
        role: str,
        system: str,
        user: str,
        required=(),
        max_tokens: int = 1200,
    ) -> dict:
        """
        Generate a JSON object and validate required keys.

        If parsing or validation fails, ask the configured providers to repair
        the response once. The repair request still uses chat_text(), so it
        receives the same Groq guard and fallback behavior.
        """
        raw = self.chat_text(
            role=role,
            system=system,
            user=user,
            max_tokens=max_tokens,
            json_mode=True,
            temperature=0,
        )

        for attempt in range(2):
            try:
                obj = parse_json(raw)

                missing = [
                    key for key in required
                    if key not in obj
                ]

                if missing:
                    raise ValueError(
                        f"missing required keys: {missing}"
                    )

                return obj

            except ValueError as e:
                if attempt == 1:
                    raise LLMError(
                        f"Malformed JSON from role '{role}': {e}"
                    ) from e

                log.warning(
                    "Invalid JSON from role=%s; requesting one repair: %s",
                    role,
                    e,
                )

                repair_prompt = (
                    f"{user}\n\n"
                    "Your previous response was invalid. "
                    f"Validation error: {e}\n"
                    "Return ONLY one valid JSON object. "
                    "Do not include Markdown fences, explanations, "
                    "or text outside the JSON object."
                )

                raw = self.chat_text(
                    role=role,
                    system=system,
                    user=repair_prompt,
                    max_tokens=max_tokens,
                    json_mode=True,
                    temperature=0,
                )

        raise LLMError("Unreachable JSON validation state.")