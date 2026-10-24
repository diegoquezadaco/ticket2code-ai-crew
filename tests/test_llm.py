"""Tests for LLM requests, error handling, and provider logging."""

from types import SimpleNamespace

import pytest

from app.llm import LLMClient, LLMError


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def create(self, **kwargs):
        if self.error:
            raise self.error
        return self.response


def make_client(response=None, error=None):
    llm = LLMClient.__new__(LLMClient)
    llm.s = SimpleNamespace()
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions(response=response, error=error)
        )
    )
    return llm, client


def call(llm, client):
    return llm._call(
        client=client,
        model="test-model",
        system="system",
        user="user",
        provider_name="primary",
        role="planner",
        max_tokens=100,
        json_mode=False,
        temperature=0,
    )


def test_request_failure_log_includes_provider_and_role(caplog):
    llm, client = make_client(error=RuntimeError("request failed"))

    with pytest.raises(LLMError, match="Unexpected RuntimeError: request failed"):
        call(llm, client)

    assert any(
        "LLM request failed: provider_name=primary role=planner" in record.message
        for record in caplog.records
    )


def test_request_completion_log_includes_provider_and_role(caplog):
    caplog.set_level("INFO")
    response = SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="done"),
        )],
    )
    llm, client = make_client(response=response)

    assert call(llm, client) == "done"
    assert any(
        "LLM request completed: provider=primary role=planner" in record.message
        for record in caplog.records
    )
