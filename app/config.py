"""Settings read from environment variables. No secrets live in the repo."""
from __future__ import annotations

import os
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
    max_retries: int  # Coder retries after the first attempt
    llm_timeout: float
    max_sessions: int
    fallback_timeout: float

    def model_for(self, role: str) -> str:
        per_role = {
            "planner": self.planner_model,
            "coder": self.coder_model,
            "reviewer": self.reviewer_model,
            "documentarian": self.documentarian_model,
        }
        return per_role.get(role) or self.model


def get_settings() -> Settings:
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
        fallback_timeout=_float("FALLBACK_TIMEOUT_SECONDS", 90),
        max_sessions=_int("MAX_SESSIONS", 400),
    )
