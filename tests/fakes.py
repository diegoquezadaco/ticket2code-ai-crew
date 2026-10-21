"""Scripted stand-in for LLMClient so the pipeline can be tested offline and deterministically."""
from app.llm import LLMError


class FakeLLM:
    configured = True

    def __init__(self, script):
        # role -> list of responses; the last one repeats. An Exception instance is raised.
        self.script = {role: list(items) for role, items in script.items()}
        self.calls = []

    def _next(self, role, user):
        self.calls.append((role, user))
        queue = self.script[role]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    def chat_json(self, role, system, user, required=(), max_tokens=0):
        return self._next(role, user)

    def chat_text(self, role, system, user, **kwargs):
        return self._next(role, user)

    def roles_called(self):
        return [r for r, _ in self.calls]


def route(intent="new_ticket", lang="en"):
    return {"intent": intent, "lang": lang, "reason": "test"}


PLAN = {
    "title": "Coupon validation",
    "language": "Python",
    "assumptions": ["Coupons live in a dict"],
    "tasks": [{"id": 1, "title": "validate", "details": "check coupon"}],
    "acceptance_criteria": ["rejects expired coupons"],
}
GOOD_CODE = "### FILE: coupons.py\n```python\ndef validate(c):\n    return True\n```\nNOTES: done"
BROKEN_CODE = "### FILE: coupons.py\n```python\ndef validate(c:\n    return True\n```"
APPROVE = {"approved": True, "issues": [], "feedback": "ok"}
REJECT = {"approved": False, "issues": ["expiry is not checked"], "feedback": "missing expiry check"}
