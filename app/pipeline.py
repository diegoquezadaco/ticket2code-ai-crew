"""Orchestrator: a small explicit state machine. Yields events so the UI can show every decision live."""
from __future__ import annotations

import logging
import time
import ast
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
    t0 = time.perf_counter()

    events: list = []
    stage = "router"

    # Initialize all state before entering the pipeline.
    intent = "unknown"
    lang = session.lang if session.lang in LABELS else "es"
    plan = session.plan or {}
    files: dict = {}
    delivery_files: dict = {}
    attempts = 0
    max_attempts = settings.max_retries + 1
    approved = False
    docs_approved = False
    issues: list = []
    status = "running"

    def log_event(agent: str, text: str, data=None) -> dict:
        event = {
            "type": "log",
            "t": round(time.perf_counter() - t0, 1),
            "agent": agent,
            "message": text,
        }
        if data is not None:
            event["data"] = data
        events.append(event)
        return event

    def normalize_issues(value) -> list[str]:
        """Convert structured findings into actionable, readable feedback."""
        if value is None:
            return []

        if isinstance(value, str):
            text = value.strip()

            if not text:
                return []

            # Some model responses serialize a dictionary as a string
            # inside code_issues. Parse only Python literals; never use eval().
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = None

                if isinstance(parsed, dict):
                    return normalize_issues(parsed)

            return [text]

        if isinstance(value, dict):
            # Support common reviewer/checker schemas.
            preferred_fields = (
                "description",
                "issue",
                "problem",
                "change_needed",
                "fix",
                "suggestion",
                "file",
                "line",
            )

            parts = [
                f"{key}: {value[key]}"
                for key in preferred_fields
                if value.get(key) not in (None, "")
            ]

            if parts:
                return ["; ".join(parts)]

            # Do not discard a finding when its schema is unexpected.
            return [str(value)]

        if isinstance(value, (list, tuple)):
            result = []

            for item in value:
                result.extend(normalize_issues(item))

            return result

        return [str(value)]

    def is_readme(path: str) -> bool:
        return path.rsplit("/", 1)[-1].lower() == "readme.md"

    try:
        prior_history = list(session.history)

        # ---------------------------------------------------------
        # 1. Router / clarification
        # ---------------------------------------------------------
        pending = session.pending_clarification

        if pending:
            # Continue from the one permitted clarification round.
            session.pending_clarification = None
            lang = pending["lang"]
            intent = pending["intent"]

            message = (
                f"Original request:\n{pending['request']}\n\n"
                f"User's answers to the clarification questions:\n{message}\n\n"
                "Proceed using these answers. If any detail remains unspecified, "
                "make a reasonable assumption and record it; do not ask more questions."
            )

            route = {
                "intent": intent,
                "lang": lang,
                "reason": "Received clarification answers",
            }
        else:
            stage = "router"
            stage_started = time.perf_counter()

            try:
                route = agents.route(
                    llm,
                    message,
                    prior_history,
                    session.summary,
                )
            finally:
                log.info(
                    "Agent timing: stage=router duration_seconds=%.2f",
                    time.perf_counter() - stage_started,
                )

            intent, lang = route["intent"], route["lang"]

            if not session.files:
                intent = {
                    "change_request": "new_ticket",
                    "question": "chat",
                }.get(intent, intent)

            session.lang = lang

            if intent == "clarification":
                questions = route["questions"][:5]

                if not questions:
                    questions = [{
                        "question": "Which option best matches what you want?",
                        "options": [
                            "A simple version",
                            "A more complete version",
                            "Other / specify",
                        ],
                    }]

                lines = []

                for index, item in enumerate(questions, start=1):
                    lines.append(f"{index}. {item['question']}")

                    if item.get("options"):
                        lines.append(
                            "   Options: " + " / ".join(item["options"])
                        )

                reply = "\n".join(lines)

                session.pending_clarification = {
                    "request": message,
                    "intent": route["resume_intent"],
                    "lang": lang,
                }

                session.history.append({
                    "role": "user",
                    "content": message,
                })
                session.history.append({
                    "role": "assistant",
                    "content": reply,
                })

                yield log_event(
                    "router",
                    "Asked one round of clarification questions",
                    {"questions": questions},
                )

                yield {
                    "type": "final",
                    "status": "clarification",
                    "reply": reply,
                    "t": round(time.perf_counter() - t0, 1),
                }
                return

        session.lang = lang
        session.history.append({"role": "user", "content": message})

        yield log_event("router", f"intent: {intent}", route)

        # ---------------------------------------------------------
        # 2. Handle normal questions and chat
        # ---------------------------------------------------------
        if intent in ("question", "chat"):
            stage = "assistant"
            stage_started = time.perf_counter()

            try:
                text = agents.answer(
                    llm,
                    message,
                    prior_history,
                    session.plan,
                    session.files,
                    session.open_issues,
                )
            finally:
                log.info(
                    "Agent timing: stage=assistant duration_seconds=%.2f",
                    time.perf_counter() - stage_started,
                )

            yield log_event(
                "assistant",
                "Answered from the conversation context",
            )

            session.history.append({
                "role": "assistant",
                "content": text[:600],
            })

            yield {
                "type": "final",
                "status": "answer",
                "reply": text,
                "t": round(time.perf_counter() - t0, 1),
            }
            return

        # ---------------------------------------------------------
        # 3. Planner
        # ---------------------------------------------------------
        is_change = intent == "change_request"
        stage = "planner"
        stage_started = time.perf_counter()

        try:
            plan = agents.make_plan(
                llm,
                message,
                lang,
                session.plan if is_change else None,
                session.files if is_change else {},
                session.open_issues if is_change else [],
            )
        finally:
            log.info(
                "Agent timing: stage=planner duration_seconds=%.2f",
                time.perf_counter() - stage_started,
            )

        if plan.get("questions"):
            questions_text = "\n".join(
                f"- {question}" for question in plan["questions"]
            )

            reply_msg = (
                f"Necesito aclarar algunos puntos antes de continuar:\n"
                f"{questions_text}"
            )

            session.history.append({
                "role": "assistant",
                "content": reply_msg,
            })

            yield log_event(
                "planner",
                "Plan returned clarifying questions due to missing details",
                plan,
            )

            yield {
                "type": "final",
                "status": "clarification",
                "reply": reply_msg,
                "t": round(time.perf_counter() - t0, 1),
            }
            return

        yield log_event(
            "planner",
            f"Plan '{plan['title']}': "
            f"{len(plan['tasks'])} tasks, "
            f"{len(plan['assumptions'])} assumptions",
            plan,
        )

        # README is deliberately excluded from implementation work.
        files = {
            path: content
            for path, content in (
                dict(session.files) if is_change else {}
            ).items()
            if not is_readme(path)
        }

        delivery_files = dict(files)

        # =========================================================
        # 4. IMPLEMENTATION RETRY LOOP
        #
        # Coder -> static checks -> reviewer
        #
        # The documentarian and documentation checks must not run
        # anywhere inside this loop.
        # =========================================================
        feedback_code: list[str] = []
        approved = False
        issues = []
        attempts = 0

        # Checkpoint the latest successfully generated implementation.
        last_successful_files = dict(files)
        generation_error = None

        for attempt in range(1, max_attempts + 1):
            attempts = attempt

            # -----------------------------------------------------
            # 4a. Coder
            # -----------------------------------------------------
            stage = "coder"

            coder_files = {
                path: content
                for path, content in files.items()
                if not is_readme(path)
            }

            coder_feedback = "\n".join(feedback_code) or None
            note = (
                "addressing code feedback"
                if coder_feedback
                else "first version"
            )

            yield log_event(
                "coder",
                f"Attempt {attempt}/{max_attempts}: {note}",
                {"feedback": coder_feedback} if coder_feedback else None,
            )

            stage_started = time.perf_counter()

            try:
                new_files, notes = agents.write_code(
                    llm,
                    plan,
                    coder_files,
                    coder_feedback,
                    attempt,
                )
            except Exception as exc:
                generation_error = str(exc)

                # Restore the last successfully generated files.
                files = dict(last_successful_files)
                delivery_files = dict(last_successful_files)

                issues = [
                    f"Code generation failed on attempt {attempt}: {generation_error}"
                ]

                yield log_event(
                    "coder",
                    "Code generation failed; preserving the last successful files",
                    {
                        "attempt": attempt,
                        "error": generation_error,
                        "preserved_files": list(files.keys()),
                    },
                )

                # Exit the implementation loop and continue to finalization.
                break
            finally:
                log.info(
                    "Agent timing: stage=coder attempt=%d duration_seconds=%.2f",
                    attempt,
                    time.perf_counter() - stage_started,
                )


            # The coder never owns README.md.
            files.update({
                path: content
                for path, content in new_files.items()
                if not is_readme(path)
            })

            # Remove any README that might have been inherited or
            # returned accidentally.
            files = {
                path: content
                for path, content in files.items()
                if not is_readme(path)
            }

            delivery_files = dict(files)
            # Save the latest successfully generated source and test files.
            last_successful_files = dict(files)

            yield log_event(
                "coder",
                f"Wrote {len(new_files)} file(s): "
                f"{', '.join(new_files) if new_files else '(none)'}",
                {"notes": notes} if notes else None,
            )

            # -----------------------------------------------------
            # 4b. Static checks on implementation only
            # -----------------------------------------------------
            stage = "static_checks"
            stage_started = time.perf_counter()

            try:
                code_problems = normalize_issues(
                    agents.static_checks(files)
                )
            finally:
                log.info(
                    "Agent timing: stage=static_checks attempt=%d duration_seconds=%.2f",
                    attempt,
                    time.perf_counter() - stage_started,
                )

            if code_problems:
                feedback_code = code_problems
                issues = code_problems

                yield log_event(
                    "checks",
                    "Static checks failed; implementation needs correction",
                    {"problems": code_problems},
                )

                if attempt < max_attempts:
                    continue

                # Retry limit reached. Do not invoke the reviewer
                # for an implementation that failed static checks.
                break

            yield log_event(
                "checks",
                "Static checks passed",
            )

            # -----------------------------------------------------
            # 4c. Reviewer evaluates accepted-plan criteria and code
            # -----------------------------------------------------
            stage = "reviewer"
            stage_started = time.perf_counter()

            try:
                # `files` contains no README.md, so the reviewer
                # evaluates the implementation independently.
                review = agents.review(
                    llm,
                    plan,
                    files,
                    lang,
                )
            finally:
                log.info(
                    "Agent timing: stage=reviewer attempt=%d duration_seconds=%.2f",
                    attempt,
                    time.perf_counter() - stage_started,
                )

            code_issues = normalize_issues(
                review.get("code_issues", [])
            )

            # Compatibility with reviewers that return a generic
            # issues field. Do not use documentation_issues to decide
            # whether the implementation itself is approved.
            if not code_issues and review.get("issues"):
                code_issues = normalize_issues(review["issues"])

            reviewer_approved = review.get("approved") is True

            # Approval requires an explicit approval and no
            # outstanding code findings.
            if reviewer_approved and not code_issues:
                approved = True
                issues = []

                yield log_event(
                    "reviewer",
                    "Implementation approved",
                    {
                        "attempt": attempt,
                        "feedback": review.get("feedback", ""),
                    },
                )
                break

            # Reviewer rejected the implementation.
            feedback_code = code_issues

            if not feedback_code:
                feedback = review.get("feedback", "")

                feedback_code = [
                    str(feedback).strip()
                    if feedback and str(feedback).strip()
                    else (
                        "The reviewer did not approve the implementation. "
                        "Recheck the plan and acceptance criteria, and "
                        "address any implementation gaps."
                    )
                ]

            issues = feedback_code

            yield log_event(
                "reviewer",
                f"Implementation rejected with {len(issues)} issue(s)",
                {
                    "code_issues": issues,
                    "feedback": review.get("feedback", ""),
                },
            )

            if attempt == max_attempts:
                break

        if not approved:
            status = "not_approved"

            yield log_event(
                "system",
                f"Implementation retry limit reached "
                f"({max_attempts} attempts) without approval",
                {"open_issues": issues},
            )

        # =========================================================
        # 5. DOCUMENTATION PHASE
        #
        # Runs only after implementation approval.
        # README revisions never rerun the coder or reviewer.
        # =========================================================
        if approved:
            accepted_files = {
                path: content
                for path, content in files.items()
                if not is_readme(path)
            }

            # Preserve the accepted source/test snapshot. All
            # documentation attempts read from this same snapshot.
            delivery_files = dict(accepted_files)

            docs_max_attempts = settings.max_retries + 1
            feedback_docs: list[str] = []
            docs_approved = False
            docs_issues: list[str] = []

            # Fail explicitly if documentation checks are not
            # configured. Do not silently mark unchecked docs valid.
            documentation_checks = getattr(
                agents,
                "documentation_checks",
                None,
            )

            if not callable(documentation_checks):
                raise RuntimeError(
                    "Documentation validation is required, but "
                    "agents.documentation_checks is unavailable."
                )

            # Generate deterministic commands from the accepted code.
            # Pass them through the plan context so the documentarian
            # can use them rather than inventing commands.
            documentation_plan = dict(plan)
            documentation_plan["deterministic_commands"] = (
                get_deterministic_commands(accepted_files)
            )

            for docs_attempt in range(1, docs_max_attempts + 1):
                stage = "documentarian"

                yield log_event(
                    "documenter",
                    f"Generating README.md "
                    f"(documentation attempt {docs_attempt}/"
                    f"{docs_max_attempts})",
                    {"feedback": feedback_docs} if feedback_docs else None,
                )

                # -------------------------------------------------
                # 5a. Documentarian
                # -------------------------------------------------
                stage_started = time.perf_counter()

                try:
                    readme = agents.document_project(
                        llm,
                        documentation_plan,
                        accepted_files,
                        lang,
                        "\n".join(feedback_docs) if feedback_docs else None,
                    )
                finally:
                    log.info(
                        "Agent timing: stage=documentarian attempt=%d duration_seconds=%.2f",
                        docs_attempt,
                        time.perf_counter() - stage_started,
                    )

                if not isinstance(readme, str) or not readme.strip():
                    feedback_docs = [
                        "README.md must contain non-empty documentation "
                        "for the accepted implementation."
                    ]
                    docs_issues = feedback_docs

                    yield log_event(
                        "documenter",
                        "Generated README.md was empty or invalid",
                        {"problems": docs_issues},
                    )

                    if docs_attempt < docs_max_attempts:
                        continue

                    break

                candidate_files = {
                    **accepted_files,
                    "README.md": readme,
                }

                # -------------------------------------------------
                # 5b. Documentation checks
                # -------------------------------------------------
                stage = "documentation_checks"
                stage_started = time.perf_counter()

                try:
                    docs_issues = normalize_issues(
                        documentation_checks(
                            plan,
                            candidate_files,
                        )
                    )
                finally:
                    log.info(
                        "Agent timing: stage=documentation_checks "
                        "attempt=%d duration_seconds=%.2f",
                        docs_attempt,
                        time.perf_counter() - stage_started,
                    )

                if not docs_issues:
                    docs_approved = True
                    files = candidate_files
                    delivery_files = dict(files)

                    yield log_event(
                        "documentation_checks",
                        "README.md passed documentation checks",
                    )
                    break

                # Only documentation feedback is updated here.
                # Accepted implementation files are never changed.
                feedback_docs = docs_issues

                yield log_event(
                    "documentation_checks",
                    f"README.md failed validation with "
                    f"{len(docs_issues)} issue(s)",
                    {"problems": docs_issues},
                )

                if docs_attempt == docs_max_attempts:
                    break

            if docs_approved:
                status = "approved"
                issues = []
            else:
                # Keep the last generated README for inspection, but
                # explicitly identify it as unvalidated.
                status = "documentation_failed"
                issues = docs_issues

                if "README.md" in candidate_files:
                    files = candidate_files
                else:
                    files = dict(accepted_files)

                delivery_files = dict(files)

                yield log_event(
                    "system",
                    "Implementation was approved, but documentation "
                    "validation exhausted its retry limit",
                    {"open_issues": docs_issues},
                )

        # ---------------------------------------------------------
        # 6. Final session state and response
        # ---------------------------------------------------------
        session.plan = plan
        session.files = dict(files)

        session.open_issues = (
            []
            if status == "approved"
            else list(issues)
        )

        session.summary = (
            f"{plan['title']} | "
            f"files: {', '.join(files)} | "
            f"status: {status.upper()}"
        )

        session.history.append({
            "role": "assistant",
            "content": session.summary
            + ("" if status == "approved" else f" | open issues: {issues}"),
        })

        stage = "formatter"

        reply = format_reply(
            lang=lang,
            approved=approved,
            attempts=attempts,
            plan=plan,
            files=files,
            issues=issues,
        )

        # Do not conflate documentation failure with implementation
        # rejection. The reviewer approved the implementation; the
        # README simply did not pass its own validation.
        if status == "documentation_failed":
            warning = (
                "DOCUMENTATION VALIDATION FAILED: The implementation "
                "was approved, but README.md did not pass documentation "
                "checks within the retry limit.\n\n"
                "Documentation issues:\n"
                + "\n".join(f"- {issue}" for issue in issues)
            )

            reply = warning + "\n\n" + reply

        log.info(
            "Pipeline completed: status=%s attempts=%d duration_seconds=%.2f",
            status,
            attempts,
            time.perf_counter() - t0,
        )

        yield {
            "type": "final",
            "status": status,
            "attempts": attempts,
            "max_attempts": max_attempts,
            "reply": reply,
            "t": round(time.perf_counter() - t0, 1),
        }


    except LLMError as e:
        log.exception("LLM failure at stage=%s", stage)

        partial_files = dict(delivery_files or last_successful_files)

        if partial_files:
            session.files = dict(partial_files)

        if plan:
            session.plan = plan

        error_message = (
            f"LLM request failed during {stage}: {e}"
        )

        reply = (
            format_reply(
                lang=lang,
                approved=False,
                attempts=attempts,
                plan=plan,
                files=partial_files,
                issues=[error_message],
                failure=error_message,
            )
            if partial_files
            else error_message
        )

        session.history.append({
            "role": "assistant",
            "content": reply[:600],
        })

        yield log_event("system", error_message)

        yield {
            "type": "final",
            "status": "partial_error" if partial_files else "error",
            "attempts": attempts,
            "reply": reply,
            "t": round(time.perf_counter() - t0, 1),
        }

    except Exception as e:
        error_message = f'Falló en la etapa "{stage}": {e}'
        log.exception(error_message)

        # Preserve any accepted or partially generated files.
        partial_files = dict(delivery_files)

        if partial_files:
            session.files = partial_files

        if plan:
            session.plan = plan

        reply = (
            format_reply(
                lang=lang,
                approved=approved,
                attempts=attempts,
                plan=plan,
                files=partial_files,
                issues=issues,
                failure=error_message,
            )
            if partial_files
            else error_message
        )

        session.history.append({
            "role": "assistant",
            "content": reply[:600],
        })

        yield log_event(
            "system",
            error_message,
        )

        yield {
            "type": "final",
            "status": "partial_error" if partial_files else "error",
            "reply": reply,
            "t": round(time.perf_counter() - t0, 1),
        }

        return