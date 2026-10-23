"""Orchestrator: a small explicit state machine. Yields events so the UI can show every decision live."""
from __future__ import annotations

import logging
import time
import re
from typing import Iterator

from . import agents
from .agents import generate_file_tree, get_deterministic_commands
from .config import Settings
from .llm import LLMError
from .session import Session

log = logging.getLogger(__name__)

# Etiquetas ampliadas para incluir secciones de entrega
LABELS = {
    "es": {
        "approved": "Aprobado por el Reviewer en el intento {n}.",
        "not_approved": "NO Aprobado: se agotaron los {n} intentos sin aprobación del Reviewer.",
        "assumptions": "Supuestos",
        "pending": "Problemas pendientes",
        "files": "Archivos",
        "file_tree": "Árbol de archivos",
        "steps": "Paso a paso para continuar",
    },
    "en": {
        "approved": "Approved by the Reviewer on attempt {n}.",
        "not_approved": "NOT APPROVED: all {n} attempts were used without Reviewer approval.",
        "assumptions": "Assumptions",
        "pending": "Open issues",
        "files": "Files",
        "file_tree": "File Tree",
        "steps": "Step-by-step Guide",
    },
}

FENCE = {
    ".py": "python",
    ".js": "javascript",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".md": "markdown",
    ".sh": "bash",
    ".txt": "text",
}

def format_reply(
    lang: str,
    approved: bool,
    attempts: int,
    plan: dict,
    files: dict,
    issues: list,
    failure: str | None = None,
) -> str:
    L = LABELS[lang]
    if failure:
        partial_title = (
            "Entrega parcial: el proceso se detuvo antes de la revisión."
            if lang == "es"
            else "Partial delivery: processing stopped before review."
        )
        parts = [failure, partial_title]
    else:
        parts = [L["approved" if approved else "not_approved"].format(n=attempts)]

    if plan.get("assumptions"):
        parts.append(L["assumptions"] + ":\n" + "\n".join(f"- {a}" for a in plan["assumptions"]))
        
    if not approved and issues:
        parts.append(L["pending"] + ":\n" + "\n".join(f"- {i}" for i in issues))
        
    # 1. Agregar el Árbol de Archivos de manera determinista
    tree_str = generate_file_tree(files)
    parts.append(f"### {L['file_tree']}:\n```text\n{tree_str}\n```")
    
    # 2. Agregar el Paso a Paso determinista
    commands = get_deterministic_commands(files)
    step_lines = []
    if any(path.endswith(".py") for path in files):
        step_lines.append(
            "1. Guarda los archivos respetando el árbol anterior y abre una terminal "
            "en la carpeta raíz del proyecto."
            if lang == "es" else
            "1. Save the files following the tree above, then open a terminal in the project root."
        )
        step_lines.append("2. Create and activate a virtual environment:\n   ```bash\n   python3 -m venv .venv && source .venv/bin/activate\n   ```")
        next_number = 3
    else:
        next_number = 1

    for command in commands["install"]:
        step_lines.append(f"{next_number}. Install dependencies:\n   ```bash\n   {command}\n   ```")
        next_number += 1

    if commands["test"]:
        for command in commands["test"]:
            step_lines.append(f"{next_number}. Run the tests:\n   ```bash\n   {command}\n   ```")
            next_number += 1
    else:
        step_lines.append(
            f"{next_number}. No test command was detected; add or run tests appropriate to this project."
        )
        next_number += 1

    if commands["run"]:
        for command in commands["run"]:
            step_lines.append(f"{next_number}. Run the application:\n   ```bash\n   {command}\n   ```")
            next_number += 1
    else:
        step_lines.append(
            f"{next_number}. No executable entry point was detected, so there is no safe run command to suggest."
        )

    parts.append(f"### {L['steps']}:\n" + "\n".join(step_lines))

    # 3. Listado completo de archivos generados
    parts.append(L["files"] + ":")
    for path, content in files.items():
        ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
        language = FENCE.get(ext, "")
        longest_backtick_run = max(
            (len(match.group(0)) for match in re.finditer(r"`+", content)),
            default=0,
        )
        fence = "`" * max(3, longest_backtick_run + 1)
        parts.append(
            f"{path}\n{fence}{language}\n{content.rstrip()}\n{fence}"
        )
        
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
    lang = session.lang if session.lang in LABELS else "es"
    plan = session.plan or {}
    delivery_files = {}
    attempts = 0

    try:
        pending = session.pending_clarification
        if pending:
            # This is the one and only clarification response. Continue without asking again.
            session.pending_clarification = None
            lang = pending["lang"]
            intent = pending["intent"]
            message = (
                f"Original request:\n{pending['request']}\n\n"
                f"User's answers to the clarification questions:\n{message}\n\n"
                "Proceed using these answers. If any detail remains unspecified, make a reasonable "
                "assumption and record it; do not ask more questions."
            )
            route = {"intent": intent, "lang": lang, "reason": "Received clarification answers"}
        else:
            route = agents.route(llm, message, prior_history, session.summary)
            intent, lang = route["intent"], route["lang"]

            if not session.files:
                intent = {"change_request": "new_ticket", "question": "chat"}.get(intent, intent)
            session.lang = lang

            if intent == "clarification":
                questions = route["questions"][:5]
                if not questions:
                    questions = [{
                        "question": "Which option best matches what you want?",
                        "options": ["A simple version", "A more complete version", "Other / specify"],
                    }]

                lines = []
                for index, item in enumerate(questions, start=1):
                    lines.append(f"{index}. {item['question']}")
                    if item["options"]:
                        lines.append("   Options: " + " / ".join(item["options"]))

                reply = "\n".join(lines)
                session.pending_clarification = {
                    "request": message,
                    "intent": route["resume_intent"],
                    "lang": lang,
                }
                session.history.append({"role": "user", "content": message})
                session.history.append({"role": "assistant", "content": reply})
                yield log_event("router", "Asked one round of clarification questions", {
                    "questions": questions,
                })
                yield {
                    "type": "final", "status": "clarification", "reply": reply,
                    "t": round(time.time() - t0, 1),
                }
                return

        session.lang = lang
        session.history.append({"role": "user", "content": message})
        yield log_event("router", f"intent: {intent}", route)

        # Keep the existing question/chat handler here.
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
        
        if plan.get("questions"):
            questions_text = "\n".join(f"- {q}" for q in plan["questions"])
            reply_msg = f"Necesito aclarar algunos puntos antes de continuar:\n{questions_text}"
            session.history.append({"role": "assistant", "content": reply_msg})
            yield log_event("planner", "Plan returned clarifying questions due to missing details", plan)
            yield {"type": "final", "status": "clarification", "reply": reply_msg, "t": round(time.time() - t0, 1)}
            return

        yield log_event(
            "planner",
            f"Plan '{plan['title']}': {len(plan['tasks'])} tasks, {len(plan['assumptions'])} assumptions",
            plan,
        )

        files = dict(session.files) if is_change else {}
        delivery_files = dict(files)
        max_attempts = settings.max_retries + 1
        feedback_code: list[str] = []
        feedback_docs: list[str] = []
        approved, issues, attempts = False, [], 0

        for attempt in range(1, max_attempts + 1):
            attempts = attempt

            # Ejecutar el Coder en la primera ronda o cuando haya observaciones de código.
            if attempt == 1 or feedback_code:
                stage = "coder"
                coder_files = {
                    path: content
                    for path, content in files.items()
                    if path.rsplit("/", 1)[-1].lower() != "readme.md"
                }
                coder_feedback = "\n".join(feedback_code) or None
                note = "addressing code feedback" if coder_feedback else "first version"
                yield log_event(
                    "coder",
                    f"Attempt {attempt}/{max_attempts}: {note}",
                    {"feedback": coder_feedback} if coder_feedback else None,
                )

                new_files, notes = agents.write_code(
                    llm, plan, coder_files, coder_feedback, attempt
                )

                # README.md pertenece exclusivamente al Documenter.
                files.update({
                    path: content
                    for path, content in new_files.items()
                    if path.rsplit("/", 1)[-1].lower() != "readme.md"
                })
                delivery_files = dict(files)
                yield log_event(
                    "coder",
                    f"Wrote {len(new_files)} file(s): "
                    f"{', '.join(new_files) if new_files else '(none)'}",
                    {"notes": notes} if notes else None,
                )
            else:
                yield log_event(
                    "coder",
                    "Skipped: only documentation changes are requested",
                )

            # Regenerar el README completo con los archivos y comandos actuales.
            stage = "documenter"
            yield log_event(
                "documenter",
                "Generating README.md",
                {"feedback": feedback_docs} if feedback_docs else None,
            )
            files["README.md"] = agents.document_project(
                llm, plan, files, lang, feedback_docs or None
            )
            delivery_files = dict(files)

            stage = "reviewer"
            code_problems = agents.static_checks(files)
            docs_problems = (
                agents.documentation_checks(plan, files)
                if hasattr(agents, "documentation_checks")
                else []
            )

            if code_problems:
                review = {
                    "approved": False,
                    "code_issues": code_problems,
                    "documentation_issues": docs_problems,
                    "feedback": "Fix automated code checks:\n"
                    + "\n".join(code_problems),
                }
                yield log_event(
                    "checks",
                    "Automated checks failed",
                    {"problems": code_problems},
                )
            else:
                yield log_event("checks", "Syntax checks passed")
                review = agents.review(llm, plan, files, lang)

                # Asegura la compatibilidad si review() devuelve el formato anterior.
                review.setdefault("code_issues", [])
                review.setdefault("documentation_issues", [])
                if review.get("issues") and not (
                    review["code_issues"] or review["documentation_issues"]
                ):
                    review["code_issues"] = review["issues"]

                if docs_problems:
                    review["documentation_issues"] = (
                        list(review["documentation_issues"]) + docs_problems
                    )

                review["approved"] = (
                    review.get("approved") is True
                    and not review["code_issues"]
                    and not review["documentation_issues"]
                )

            code_issues = review.get("code_issues", [])
            documentation_issues = review.get("documentation_issues", [])
            issues = list(code_issues) + list(documentation_issues)

            if review["approved"]:
                yield log_event(
                    "reviewer",
                    "Approved",
                    {"feedback": review.get("feedback", "")},
                )
                approved = True
                break

            yield log_event("reviewer", f"Rejected with {len(issues)} issue(s)", review)

            # Enrutar observaciones al agente responsable.
            feedback_code = list(code_issues)
            feedback_docs = list(documentation_issues)

            # Evita rondas sin acción si el Reviewer rechaza sin clasificar el problema.
            if not feedback_code and not feedback_docs:
                feedback_code = [
                    review.get("feedback") or "Reviewer rejected the delivery; inspect the code."
                ]

        if not approved:
            yield log_event(
                "system",
                f"Retry limit reached ({max_attempts} attempts) without approval",
                {"open_issues": issues},
            )

        session.plan, session.files = plan, files
        session.open_issues = [] if approved else issues
        status = "approved" if approved else "not_approved"
        session.summary = f"{plan['title']} | files: {', '.join(files)} | status: {status.upper()}"
        session.history.append({"role": "assistant", "content": session.summary + ("" if approved else f" | open issues: {issues}")})
        
        # Asignamos la etapa a formatter antes de construir la respuesta final
        stage = "formatter"
        yield {
            "type": "final", "status": status, "attempts": attempts, "max_attempts": max_attempts,
            "reply": format_reply(lang, approved, attempts, plan, files, issues),
            "t": round(time.time() - t0, 1),
        }
    except LLMError as e:
        session.history.append({"role": "assistant", "content": f"[error] pipeline failed at {stage}: {e}"})
        yield log_event("system", f"Failed at {stage}: {e}")
        yield {"type": "error", "stage": stage, "message": str(e)}
    except Exception as e:
        error_message = f'Falló en la etapa "{stage}": {e}'
        log.exception(error_message)

        # delivery_files se actualiza tras cada respuesta válida del Coder
        # y del Documenter, por lo que conserva lo ya generado.
        partial_files = dict(delivery_files)
        if partial_files:
            session.files = partial_files
        if plan:
            session.plan = plan

        reply = (
            format_reply(
                lang=lang,
                approved=False,
                attempts=attempts,
                plan=plan,
                files=partial_files,
                issues=issues,
                failure=error_message,
            )
            if partial_files
            else error_message
        )

        session.history.append({"role": "assistant", "content": reply[:600]})
        yield log_event("system", error_message)
        yield {
            "type": "final",
            "status": "partial_error" if partial_files else "error",
            "reply": reply,
            "t": round(time.time() - t0, 1),
        }
        return