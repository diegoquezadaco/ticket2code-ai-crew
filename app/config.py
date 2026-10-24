"""Load application and provider settings from environment variables."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings shared by the API, pipeline, and LLM client."""

    api_key: str
    base_url: str
    model: str
    planner_model: str
    coder_model: str
    reviewer_model: str
    documentarian_model: str
    fallback_api_key: str
    fallback_base_url: str
    fallback_model: str
    max_retries: int
    llm_timeout: float
    max_sessions: int
    groq_tpm_budget: int = 5900
    groq_wait_seconds: float = 45.0
    groq_cooldown_seconds: float = 60.0
    fallback_timeout: float = 380.0

    def model_for(self, role: str) -> str:
        """Return a role-specific model or the configured default model."""
        per_role = {
            "planner": self.planner_model,
            "coder": self.coder_model,
            "reviewer": self.reviewer_model,
            "documentarian": self.documentarian_model,
        }
        return per_role.get(role) or self.model

    def groq_tpm_budget_for(self, model: str) -> int:
        """
        Return the configured rolling token budget for a specific Groq model.

        Model-specific environment variables override GROQ_TPM_BUDGET.
        """
        suffix = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").upper()
        env_name = f"GROQ_TPM_BUDGET_{suffix}"

        return _int(env_name, self.groq_tpm_budget)


def get_settings() -> Settings:
    """Build settings from the current process environment."""
    return Settings(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        planner_model=os.getenv("PLANNER_MODEL", ""),
        coder_model=os.getenv("CODER_MODEL", ""),
        reviewer_model=os.getenv("REVIEWER_MODEL", ""),
        documentarian_model=os.getenv("DOCUMENTARIAN_MODEL", ""),
        fallback_api_key=os.getenv("FALLBACK_API_KEY", ""),
        fallback_base_url=os.getenv("FALLBACK_BASE_URL", ""),
        fallback_model=os.getenv("FALLBACK_MODEL", ""),
        max_retries=max(0, _int("MAX_RETRIES", 5)),
        llm_timeout=_float("LLM_TIMEOUT_SECONDS", 60),
        fallback_timeout=_float("FALLBACK_TIMEOUT_SECONDS", 380),
        max_sessions=_int("MAX_SESSIONS", 400),
        groq_tpm_budget=_int("GROQ_TPM_BUDGET", 5900),
        groq_wait_seconds=_float("GROQ_WAIT_SECONDS", 45.0),
        groq_cooldown_seconds=_float("GROQ_COOLDOWN_SECONDS", 60.0),

    )
