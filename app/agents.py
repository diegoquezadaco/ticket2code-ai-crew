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
    "For Python, use only the standard library unless specified and include unittest coverage "
    "for every acceptance criterion. When existing files or review feedback are provided, output "
    "the full contents of every changed file; omitted files remain unchanged. Never modify README.md."
)

REVIEWER_SYS = (
    "Review delivered files against the plan and acceptance criteria. Report blocking issues only, "
    "classifying each as code_issues (implementation/executable tests) or documentation_issues "
    "(README, usage, examples, written test explanations). Approve only if both lists are empty. "
    "Write feedback in the requested language. "
    'Return ONLY JSON: {"approved":bool,"code_issues":[str],'
    '"documentation_issues":[str],"feedback":"actionable instructions"}'
)

DOCUMENTARIAN_SYS = (
    "Own README.md only. Never modify source code or tests. Use only the plan, supplied files, "
    "deterministic file tree, and commands. Never invent features, dependencies, commands, or test "
    "results. Document requirements, structure, setup, available run/test commands, usage, and "
    "existing test cases. State when commands are unavailable. Apply documentation feedback and "
    "write in the requested language. Output the complete README between these markers:\n"
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
    blocks, used = [], 0
    for path, content in files.items():
        block = f"### FILE: {path}\n```\n{content}\n```"
        if used + len(block) > limit:
            blocks.append(f"### FILE: {path}\n(omitted, too long)")
            continue
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks) or "(no files)"


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
        if new_files:
            notes = re.search(r"^NOTES:\s*(.+)$", text, re.M)
            return new_files, (notes.group(1).strip() if notes else "")
        user += "\n\nYour previous reply did not follow the required `### FILE:` format. Output the files again in that format."
    raise LLMError("the Coder returned no parseable files")


def review(llm: LLMClient, plan: dict, files: dict, lang: str) -> dict:
    user = (
        f"Language for text fields: {LANG_NAMES[lang]}\n"
        f"Plan:\n{json.dumps(plan, ensure_ascii=False)}\n\n"
        f"Automated syntax checks: passed\n\nFiles:\n{render_files(files, 3500)}"
    )
    r = llm.chat_json(
        "reviewer", REVIEWER_SYS, user,
        required=("approved",), max_tokens=500,
    )

    def clean_issues(key: str) -> list[str]:
        value = r.get(key, [])
        return (
            [str(item).strip() for item in value if str(item).strip()]
            if isinstance(value, list)
            else []
        )

    code_issues = clean_issues("code_issues")
    documentation_issues = clean_issues("documentation_issues")

    # Compatibilidad con respuestas antiguas del Reviewer.
    legacy_issues = clean_issues("issues")
    if legacy_issues and not code_issues and not documentation_issues:
        code_issues = legacy_issues

    approved = (
        (r.get("approved") is True or str(r.get("approved")).lower() == "true")
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