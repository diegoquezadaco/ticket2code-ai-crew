"""The agent roles. Each one is a prompt plus a small function with a strict input/output contract."""
from __future__ import annotations

import ast
import json
import re

from .llm import LLMClient, LLMError

LANG_NAMES = {"es": "Spanish", "en": "English"}
INTENTS = ("new_ticket", "change_request", "question", "chat", "clarification")

ROUTER_SYS = (
    "Classify the latest message using conversation history and delivered work. "
    "Use clarification only for essential missing details; ask at most 5 questions in one turn, "
    "each with 2-4 concise options, including 'Other / specify'. Otherwise, proceed. "
    "Use new_ticket/change_request when clear, question for questions about existing work, "
    "and chat for ordinary conversation. For clarification, set resume_intent to "
    "new_ticket or change_request. Do not ask questions for chat or existing-work questions. "
    'Return ONLY JSON: {"intent":"new_ticket|change_request|question|chat|clarification",'
    '"resume_intent":"new_ticket|change_request","lang":"es|en","reason":"short",'
    '"questions":[{"question":"...","options":["...","..."]}]}'
)

PLANNER_SYS = (
    "Convert the request into a technical plan. Ask questions only for essential missing details; "
    "otherwise make reasonable assumptions. For changes, plan only the delta and preserve working "
    "behavior. Use the ticket language, defaulting to Python if unspecified. Write text in the "
    "requested language and code in English. "
    'Return ONLY JSON: {"title":str,"language":str,"assumptions":[str],"questions":[str],'
    '"tasks":[{"id":int,"title":str,"details":str}],"acceptance_criteria":[str]}. '
    "When proceeding, provide 3-6 tasks and 3-6 acceptance criteria."
)

CODER_SYS = (
    "Implement the plan completely. Output each file as `### FILE: path` followed by one fenced "
    "code block containing the full file. Output nothing else except optional `NOTES: ...`. "
    "Implement only what the plan requires.Preserve existing behavior outside the requested change."
    "Add tests that meaningfully cover the acceptance criteria without duplicating tests."
    "Do not add extra features, abstractions, or files unless necessary."
    "Keep notes concise."
    "Never modify files that do not need changes."
)

REVIEWER_SYS = (
    "You are a senior software engineer reviewing an implementation."
    "Review the supplied implementation against the plan and acceptance criteria."

    "Return valid JSON only, with these fields:"
    " - approved: boolean"
    " - code_issues: array of strings"
    " - documentation_issues: array of strings"
    " - feedback: short string"

    "Rules:"
    " - Each issue must identify the relevant file and the concrete correction needed."
    " - Do not invent requirements."
    " - Do not reject code because a file is absent from the prompt if its contents are provided elsewhere in the review context."
    " - Do not claim that tests pass unless their results were actually supplied."
    " - Do not reproduce source code in your response."
    " - If there are no implementation issues, return empty issue arrays and approved=true."
    " - If the supplied context is insufficient to verify an acceptance criterion, state the limitation concisely."
    " - Keep the entire response compact."
)

DOCUMENTARIAN_SYS = (
    "Own README.md only. Use the deterministic_commands supplied in the plan as the source"
    "of truth for setup, installation, testing, and run instructions."
    "Do not invent or modify commands. Document only files and functionality present in the accepted implementation."
    "Do not modify source files or tests. Output the complete README between these markers:\n"
    "### README_CONTENT_START\n"
    "README content\n"
    "### README_CONTENT_END"
)

ASSISTANT_SYS = (
    "Answer concisely in the user's language using conversation history, delivered work, the plan, "
    "and deterministic commands. If nothing was delivered, say so and invite a ticket. "
    "After successful delivery, approval, or a request for next steps, include a clean file tree "
    "and sequential, accurate setup, run, and test instructions based on supplied commands."
)


def _history_text(history: list) -> str:
    recent = history[-4:]
    older = history[:-4][-2:]

    messages = [
        f"{m['role']}: {m['content'][:150]}"
        for m in older
    ]
    messages += [
        f"{m['role']}: {m['content'][:400]}"
        for m in recent
    ]

    return "\n".join(messages) or "(empty)"



def render_files(files: dict, limit: int = 14000) -> str:
    if not files:
        return "(no files)"

    items = list(files.items())
    blocks = []

    # Reserve space for file headers, code fences, and truncation markers.
    overhead = sum(len(f"### FILE: {path}\n```\n\n```") for path, _ in items)
    available = max(0, limit - overhead)

    # Initially distribute the remaining budget evenly across files.
    remaining = {path: str(content) for path, content in items}
    budgets = {
        path: available // len(items)
        for path, _ in items
    }

    # Redistribute unused budget from short files to longer files.
    unused = sum(
        max(0, budgets[path] - len(content))
        for path, content in remaining.items()
    )

    while unused > 0:
        candidates = [
            path for path, content in remaining.items()
            if len(content) > budgets[path]
        ]

        if not candidates:
            break

        extra_per_file = max(1, unused // len(candidates))
        distributed = 0

        for path in candidates:
            extra = min(
                extra_per_file,
                len(remaining[path]) - budgets[path],
                unused - distributed,
            )
            budgets[path] += extra
            distributed += extra

            if distributed >= unused:
                break

        if distributed == 0:
            break

        unused -= distributed

    for path, content in items:
        content = str(content)
        budget = budgets[path]

        if len(content) > budget:
            marker = "\n... [content truncated; not the complete file] ..."
            if budget > len(marker):
                content = content[:budget - len(marker)] + marker
            else:
                content = content[:budget]

        blocks.append(f"### FILE: {path}\n```\n{content}\n```")

    return "\n\n".join(blocks)


_FILE_RE = re.compile(r"###\s*FILE:\s*`?([^\n`]+?)`?\s*\n```[^\n]*\n(.*?)\n?```", re.S)


def parse_files(text: str) -> dict:
    files = {}
    for path, content in _FILE_RE.findall(text):
        path = path.strip().replace("\\", "/")
        if not path or path.startswith("/") or ".." in path.split("/"):
            continue
        files[path] = content.rstrip() + "\n"
    return files


def generate_file_tree(files: dict) -> str:
    """Genera una representación visual tipo árbol de archivos de manera determinista."""
    if not files:
        return "(no files)"
    tree = {}
    for path in files.keys():
        parts = path.split("/")
        current = tree
        for part in parts:
            current = current.setdefault(part, {})
            
    def _build_lines(node, prefix=""):
        lines = []
        items = sorted(node.items())
        for i, (name, subtree) in enumerate(items):
            connector = "└── " if i == len(items) - 1 else "├── "
            lines.append(f"{prefix}{connector}{name}")
            if subtree:
                extension = "    " if i == len(items) - 1 else "│   "
                lines.extend(_build_lines(subtree, prefix + extension))
        return lines

    return "\n".join(["project_root/"] + _build_lines(tree))


def get_deterministic_commands(files: dict) -> dict:
    """Calcula de forma determinista los comandos de instalación, ejecución y pruebas."""
    commands = {"install": [], "run": [], "test": []}
    names = list(files)
    contents = "\n".join(files.values())

    has_python = any(path.endswith(".py") for path in names)
    requirements = next((path for path in names if path.endswith("requirements.txt")), None)
    pytest_tests = (
        any("test" in path.lower() for path in names)
        and ("import pytest" in contents or "from pytest" in contents)
    )

    if requirements:
        commands["install"].append(f"python -m pip install -r {requirements}")
    if pytest_tests:
        commands["install"].append("python -m pip install pytest")
        commands["test"].append("python -m pytest -q")
    elif any("test" in path.lower() for path in names) and has_python:
        commands["test"].append("python -m unittest discover")

    # Do not guess that an arbitrary .py file is an executable application.
    entry_names = {"main.py", "run.py", "cli.py"}
    entry = next(
        (path for path in names if path.rsplit("/", 1)[-1] in entry_names),
        None,
    )
    if entry:
        commands["run"].append(f"python {entry}")

    return commands


def static_checks(files: dict) -> list:
    """Deterministic checks that run before the Reviewer."""
    if not files:
        return ["No files were produced."]
    problems = []
    for path, content in files.items():
        if path.endswith(".py"):
            try:
                ast.parse(content)
            except SyntaxError as e:
                problems.append(f"{path}: SyntaxError on line {e.lineno}: {e.msg}")
        elif path.endswith(".json"):
            try:
                json.loads(content)
            except ValueError as e:
                problems.append(f"{path}: invalid JSON ({e})")
    return problems


def route(llm: LLMClient, message: str, history: list, summary: str) -> dict:
    user = (
        f"Last delivered work: {summary or 'none'}\n\n"
        f"Recent conversation:\n{_history_text(history)}\n\nLatest user message:\n{message}"
    )
    r = llm.chat_json("router", ROUTER_SYS, user, required=("intent",), max_tokens=500)

    intent = r["intent"] if r.get("intent") in INTENTS else "new_ticket"
    questions = r.get("questions", [])
    if not isinstance(questions, list):
        questions = []

    # Normalize and cap model output so the UI only presents up to five questions.
    clean_questions = []
    for item in questions[:5]:
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            continue
        options = item.get("options", [])
        if not isinstance(options, list):
            options = []
        clean_questions.append({
            "question": str(item["question"]).strip(),
            "options": [str(option).strip() for option in options[:4] if str(option).strip()],
        })

    resume_intent = r.get("resume_intent")
    if resume_intent not in ("new_ticket", "change_request"):
        resume_intent = "change_request" if summary else "new_ticket"

    return {
        "intent": intent,
        "resume_intent": resume_intent,
        "lang": "es" if r.get("lang") == "es" else "en",
        "reason": str(r.get("reason", ""))[:200],
        "questions": clean_questions,
    }


def make_plan(llm: LLMClient, request: str, lang: str, prior_plan, files: dict, open_issues: list) -> dict:
    context = ""
    if prior_plan:
        context = (
            f"Previous plan:\n{json.dumps(prior_plan, ensure_ascii=False)}\n\n"
            f"Current files:\n{render_files(files, 8000)}\n\n"
        )
        if open_issues:
            context += "Unresolved reviewer issues:\n" + "\n".join(open_issues) + "\n\n"

    user = f"{context}Language for text fields: {LANG_NAMES[lang]}\nRequest:\n{request}"
    retry_user = (
        f"{user}\n\nYour previous response did not satisfy the required JSON schema. "
        'Return valid JSON with ALL keys: "title", "language", "assumptions", "questions", '
        '"tasks", and "acceptance_criteria". Do not ask the user questions; use reasonable '
        "assumptions for non-critical details. Include at least one task and one acceptance criterion."
    )

    try:
        plan = llm.chat_json(
            "planner", PLANNER_SYS, user,
            required=("tasks", "acceptance_criteria"), max_tokens=1400,
        )
    except LLMError:
        plan = llm.chat_json(
            "planner", PLANNER_SYS, retry_user,
            required=("tasks", "acceptance_criteria"), max_tokens=1400,
        )

    plan.setdefault("title", "Untitled")
    plan.setdefault("language", "Python")
    plan.setdefault("assumptions", [])
    # Clarification is handled only by the router, never by the planner.
    plan["questions"] = []
    return plan


def write_code(llm: LLMClient, plan: dict, files: dict, feedback, attempt: int):
    user = f"Plan:\n{json.dumps(plan, ensure_ascii=False)}\n\nCurrent files:\n{render_files(files, 12000)}"
    if feedback:
        user += f"\n\nReviewer feedback to fix (attempt {attempt}):\n{feedback}"
    for _ in range(2):
        text = llm.chat_text("coder", CODER_SYS, user, max_tokens=3500, temperature=0.2)
        new_files = parse_files(text)
        new_files = parse_files(text)

    if new_files:
        import logging

        logger = logging.getLogger(__name__)

        logger.info(
            "Coder parsed %d file(s): %s",
            len(new_files),
            {
                path: {
                    "characters": len(content),
                    "lines": len(content.splitlines()),
                    "ends_with_newline": content.endswith("\n"),
                }
                for path, content in new_files.items()
            },
        )

        notes = re.search(r"^NOTES:\s*(.+)$", text, re.M)

        return new_files, (notes.group(1).strip() if notes else "")
    raise LLMError("the Coder returned no parseable files")


def review(llm: LLMClient, plan: dict, files: dict, lang: str) -> dict:
    review_files = {
        path: content
        for path, content in files.items()
        if path.rsplit("/", 1)[-1].lower() != "readme.md"
    }

    compact_plan = {
        "title": plan.get("title", ""),
        "tasks": plan.get("tasks", []),
        "acceptance_criteria": plan.get("acceptance_criteria", []),
        "assumptions": plan.get("assumptions", []),
    }

    user = (
        f"Language for text fields: {LANG_NAMES[lang]}\n\n"
        f"Plan and acceptance criteria:\n"
        f"{json.dumps(compact_plan, ensure_ascii=False)}\n\n"
        "Automated syntax checks: passed.\n\n"
        "Review the implementation against the acceptance criteria. "
        "Inspect the actual source files and tests provided below. "
        "Do not report a file as missing if it is present. "
        "Do not invent requirements. If the provided context is "
        "insufficient to verify a criterion, identify that limitation "
        "rather than claiming the implementation is defective. "
        "Return only actionable implementation issues and a concise "
        "approval decision.\n\n"
        f"Files:\n{render_files(review_files, 9000)}"
    )

    r = llm.chat_json(
        "reviewer",
        REVIEWER_SYS,
        user,
        required=("approved",),
        max_tokens=1000,
    )

    def clean_issues(key: str) -> list[str]:
        value = r.get(key, [])

        if isinstance(value, (str, dict)):
            value = [value]

        if not isinstance(value, list):
            return []

        result = []

        for item in value:
            if isinstance(item, dict):
                # Extract useful details instead of passing a raw dict
                # representation to the Coder.
                fields = (
                    "file",
                    "line",
                    "issue",
                    "description",
                    "problem",
                    "change_needed",
                    "fix",
                    "suggestion",
                )

                parts = [
                    f"{field}: {item[field]}"
                    for field in fields
                    if item.get(field) not in (None, "")
                ]

                message = "; ".join(parts) if parts else str(item)
            else:
                message = str(item).strip()

            if message:
                result.append(message)

        return result

    code_issues = clean_issues("code_issues")
    documentation_issues = clean_issues("documentation_issues")

    # Compatibility with older Reviewer response formats.
    legacy_issues = clean_issues("issues")
    if legacy_issues and not code_issues and not documentation_issues:
        code_issues = legacy_issues

    approved_value = r.get("approved")
    approved = (
        approved_value is True
        or str(approved_value).lower() == "true"
    )

    approved = (
        approved
        and not code_issues
        and not documentation_issues
    )

    return {
        "approved": approved,
        "code_issues": code_issues,
        "documentation_issues": documentation_issues,
        "issues": code_issues + documentation_issues,
        "feedback": str(r.get("feedback", "")).strip(),
    }

def document_project(
    llm: LLMClient,
    plan: dict,
    files: dict,
    lang: str,
    feedback: list[str] | None = None,
) -> str:
    doc_files = {
        path: content
        for path, content in files.items()
        if path.rsplit("/", 1)[-1].lower() != "readme.md"
    }
    doc_files["README.md"] = "(generated by the Documentation Agent)"

    commands = get_deterministic_commands(doc_files)
    tree = generate_file_tree(doc_files)

    feedback_text = "\n".join(feedback or []) or "(none)"

    user = (
        f"Language: {LANG_NAMES[lang]}\n\n"
        f"Plan and acceptance criteria:\n{json.dumps(plan, ensure_ascii=False)}\n\n"
        f"Deterministic file tree:\n{tree}\n\n"
        f"Deterministic commands:\n{json.dumps(commands, ensure_ascii=False)}\n\n"
        f"Project files, including executable tests:\n{render_files(doc_files, 4500)}\n\n"
        f"Reviewer documentation feedback to address:\n{feedback_text}\n\n"
        "Return a complete README.md. Do not omit sections just because a command is unavailable."
    )

    text = llm.chat_text(
        "documentarian", DOCUMENTARIAN_SYS, user,
        max_tokens=1000, temperature=0.2,
    )

    def extract_readme(response: str) -> str | None:
        match = re.search(
            r"(?ms)^### README_CONTENT_START\s*\n(.*?)^### README_CONTENT_END\s*$",
            response.strip(),
        )
        if match:
            return match.group(1).strip() + "\n"
        return None

    readme = extract_readme(text)
    if readme is None:
        retry = (
            f"{user}\n\n"
            "Your previous response omitted the required markers. "
            "Return the complete README again, with no text outside this exact envelope:\n"
            "### README_CONTENT_START\n"
            "README content\n"
            "### README_CONTENT_END"
        )
        text = llm.chat_text(
            "documentarian", DOCUMENTARIAN_SYS, retry,
            max_tokens=1000, temperature=0.1,
        )
        readme = extract_readme(text)

    # Algunos modelos ignoran el sobre. Conserva su respuesta para entregar el README
    # en lugar de detener el pipeline y perder el resto de los archivos.
    if readme is None and text.strip():
        readme = text.strip() + "\n"

    if not readme:
        raise LLMError("the Documentation Agent returned an empty README")

    return readme


def answer(llm: LLMClient, message: str, history: list, plan, files: dict, open_issues: list) -> str:
    file_tree = generate_file_tree(files)
    commands = get_deterministic_commands(files)
    
    user = (
        f"Conversation:\n{_history_text(history)}\n\n"
        f"Last plan: {json.dumps(plan, ensure_ascii=False) if plan else 'none'}\n\n"
        f"Deterministic File Tree:\n{file_tree}\n\n"
        f"Deterministic Commands:\n{json.dumps(commands, ensure_ascii=False)}\n\n"
        f"Delivered files:\n{render_files(files, 8000)}\n\n"
        f"Unresolved issues: {open_issues or 'none'}\n\nUser message:\n{message}"
    )
    return llm.chat_text("assistant", ASSISTANT_SYS, user, max_tokens=1000, temperature=0.3)


def handle_chat_message(llm: LLMClient, message: str, history: list, summary: str, plan, files: dict, open_issues: list) -> str:
    """Función de orquestación principal que maneja la intención de aclaración y rutas."""
    routing = route(llm, message, history, summary)
    intent = routing["intent"]
    
    # 2. Manejar la intención de aclaración directamente sin disparar el ticket/planner
    if intent == "clarification":
        clarification_prompt = (
            f"The user request was deemed ambiguous. Ask concise clarifying questions to gather necessary details.\n"
            f"User message: {message}"
        )
        return llm.chat_text("assistant", ASSISTANT_SYS, clarification_prompt, max_tokens=400, temperature=0.3)
        
    elif intent in ("question", "chat"):
        return answer(llm, message, history, plan, files, open_issues)
        
    # Si es new_ticket o change_request, aquí continuaría tu pipeline normal...
    return routing


def documentation_checks(plan: dict, files: dict) -> list[str]:
    """Comprueba que README.md exista y documente los comandos detectados."""
    readme = next(
        (
            content for path, content in files.items()
            if path.rsplit("/", 1)[-1].lower() == "readme.md"
        ),
        None,
    )
    if readme is None:
        return ["README.md is missing."]

    commands = get_deterministic_commands(files)
    required_commands = commands["install"] + commands["run"] + commands["test"]
    missing = [command for command in required_commands if command.lower() not in readme.lower()]

    if missing:
        return ["README.md must document these detected commands: " + ", ".join(missing)]
    return []