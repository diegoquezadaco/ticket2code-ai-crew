"""Orchestrator: a small explicit state machine. Yields events so the UI can show every decision live."""
from __future__ import annotations

import logging
import time
from typing import Iterator

from . import agents
from .config import Settings
from .llm import LLMError
from .session import Session

log = logging.getLogger(__name__)

LABELS = {
    "es": {
        "approved": "Aprobado por el Reviewer en el intento {n}.",
        "not_approved": "NO APROBADO: se agotaron los {n} intentos sin aprobación del Reviewer. "
                        "Entrego el último intento con los problemas pendientes; no lo considero terminado.",
        "assumptions": "Supuestos",
        "pending": "Problemas pendientes",
        "files": "Archivos",
    },
    "en": {
        "approved": "Approved by the Reviewer on attempt {n}.",
        "not_approved": "NOT APPROVED: all {n} attempts were used without Reviewer approval. "
                        "Delivering the last attempt with its open issues; do not treat it as finished.",
        "assumptions": "Assumptions",
        "pending": "Open issues",
        "files": "Files",
    },
}
FENCE = {".py": "python", ".js": "javascript", ".ts": "typescript", ".java": "java", ".json": "json",
         ".md": "markdown", ".html": "html", ".css": "css", ".sql": "sql", ".go": "go"}


def format_reply(lang: str, approved: bool, attempts: int, plan: dict, files: dict, issues: list) -> str:
    L = LABELS[lang]
    parts = [L["approved" if approved else "not_approved"].format(n=attempts)]
    if plan.get("assumptions"):
        parts.append(L["assumptions"] + ":\n" + "\n".join(f"- {a}" for a in plan["assumptions"]))
    if not approved and issues:
        parts.append(L["pending"] + ":\n" + "\n".join(f"- {i}" for i in issues))
    parts.append(L["files"] + ":")
    for path, content in files.items():
        ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
        parts.append(f"{path}\n```{FENCE.get(ext, '')}\n{content}```")
    return "\n\n".join(parts)


def run_turn(session: Session, message: str, llm, settings: Settings) -> Iterator[dict]:
    t0 = time.time()
    events: list = []
    stage = "router"

    def log_event(agent: str, text: str, data=None) -> dict:
        e = {"type": "log", "t": round(time.time() - t0, 1), "agent": agent, "message": text}
        if data is not None:
            e["data"] = data
        events.append(e)
        return e

    prior_history = list(session.history)
    intent = "unknown"
    try:
        route = agents.route(llm, message, prior_history, session.summary)
        intent, lang = route["intent"], route["lang"]
        if not session.files:  # nothing delivered yet: follow-ups cannot refer to anything
            intent = {"change_request": "new_ticket", "question": "chat"}.get(intent, intent)
        session.lang = lang
        session.history.append({"role": "user", "content": message})
        yield log_event("router", f"intent: {intent}", route)

        if intent in ("question", "chat"):
            stage = "assistant"
            text = agents.answer(llm, message, prior_history, session.plan, session.files, session.open_issues)
            yield log_event("assistant", "Answered from the conversation context")
            session.history.append({"role": "assistant", "content": text[:600]})
            yield {"type": "final", "status": "answer", "reply": text, "t": round(time.time() - t0, 1)}
            return

        is_change = intent == "change_request"
        stage = "planner"
        plan = agents.make_plan(
            llm, message, lang,
            session.plan if is_change else None,
            session.files if is_change else {},
            session.open_issues if is_change else [],
        )
        yield log_event(
            "planner",
            f"Plan '{plan['title']}': {len(plan['tasks'])} tasks, {len(plan['assumptions'])} assumptions",
            plan,
        )

        files = dict(session.files) if is_change else {}
        max_attempts = settings.max_retries + 1
        feedback, approved, issues, attempts = None, False, [], 0
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            stage = "coder"
            note = "addressing reviewer feedback" if feedback else "first version"
            yield log_event("coder", f"Attempt {attempt}/{max_attempts}: {note}",
                            {"feedback": feedback} if feedback else None)
            new_files, notes = agents.write_code(llm, plan, files, feedback, attempt)
            files.update(new_files)
            yield log_event("coder", f"Wrote {len(new_files)} file(s): {', '.join(new_files)}",
                            {"notes": notes} if notes else None)

            stage = "reviewer"
            problems = agents.static_checks(files)
            if problems:
                yield log_event("checks", "Automated checks failed", {"problems": problems})
                review = {"approved": False, "issues": problems,
                          "feedback": "Fix these automated check failures:\n- " + "\n- ".join(problems)}
            else:
                yield log_event("checks", "Syntax checks passed")
                review = agents.review(llm, plan, files, lang)
            issues = review["issues"]
            if review["approved"]:
                yield log_event("reviewer", "Approved", {"feedback": review["feedback"]} if review["feedback"] else None)
                approved = True
                break
            yield log_event("reviewer", f"Rejected with {len(issues)} issue(s)", review)
            feedback = review["feedback"]

        if not approved:
            yield log_event("system", f"Retry limit reached ({max_attempts} attempts) without approval", {"open_issues": issues})

        session.plan, session.files = plan, files
        session.open_issues = [] if approved else issues
        status = "approved" if approved else "not_approved"
        session.summary = f"{plan['title']} | files: {', '.join(files)} | status: {status.upper()}"
        session.history.append({"role": "assistant", "content": session.summary + ("" if approved else f" | open issues: {issues}")})
        yield {
            "type": "final", "status": status, "attempts": attempts, "max_attempts": max_attempts,
            "reply": format_reply(lang, approved, attempts, plan, files, issues),
            "t": round(time.time() - t0, 1),
        }
    except LLMError as e:
        session.history.append({"role": "assistant", "content": f"[error] pipeline failed at {stage}: {e}"})
        yield log_event("system", f"Failed at {stage}: {e}")
        yield {"type": "error", "stage": stage, "message": str(e)}
    except Exception as e:  # never fail silently
        log.exception("unexpected failure at stage %s", stage)
        yield {"type": "error", "stage": stage,
               "message": f"Unexpected internal error ({type(e).__name__}). Details are in the server logs."}
    finally:
        session.runs.append({"request": message, "intent": intent, "events": events})
        del session.runs[:-20]
        del session.history[:-20]
