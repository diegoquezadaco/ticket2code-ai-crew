"""The agent roles. Each one is a prompt plus a small function with a strict input/output contract."""
from __future__ import annotations

import ast
import json
import re
import difflib

from .llm import LLMClient, LLMError

LANG_NAMES = {"es": "Spanish", "en": "English"}
INTENTS = ("new_ticket", "change_request", "question", "chat", "clarification")
PROJECT_EXTRACT_COMMAND = "unzip project.zip -d project && cd project"

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
    "Implement the plan completely. Preserve existing behavior outside "
    "the requested change. Add tests that meaningfully cover the "
    "acceptance criteria without duplicating tests. Do not add extra "
    "features, abstractions, or files unless necessary. "
    "Keep notes concise. Never modify files that do not need changes. "
    "When asked for SEARCH/REPLACE edit blocks, output only those blocks "
    "in the exact requested format, containing only the necessary "
    "changes. Do not output complete files in edit-block mode. "
    "Otherwise, output each complete file as `### FILE: path` followed "
    "by one fenced code block containing the full file. Output nothing "
    "else except optional `NOTES: ...`."
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
    " of truth for setup, installation, testing, and run instructions. "
    "Include an explicit first step telling users to download project.zip from the chat,"
    "then open a terminal where it was saved and extract it into a project folder with"
    f" this exact command: {PROJECT_EXTRACT_COMMAND}. Continue setup instructions from "
    "inside that extracted project folder. "
    "Do not invent or modify commands. Document only files and functionality present in the accepted implementation. "
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

_EDIT_BLOCK_RE = re.compile(
    r"(?ms)^### EDIT:[ \t]*(?P<path>[^\n]+)\n"
    r"<<<<<<< SEARCH\n"
    r"(?P<search>.*?)\n=======\n"
    r"(?P<replace>.*?)\n>>>>>>> REPLACE"
)


def apply_edit_blocks(
    text: str,
    files: dict,
) -> tuple[dict | None, list[str], str | None]:
    """
    Apply exact SEARCH/REPLACE blocks to existing files.

    Returns:
        updated files, changed paths, error message

    If any edit is invalid, no partial result is returned.
    """
    matches = list(_EDIT_BLOCK_RE.finditer(text or ""))

    if not matches:
        return None, [], "No valid SEARCH/REPLACE edit blocks were returned."

    updated = dict(files)
    changed_paths = []

    for match in matches:
        path = match.group("path").strip().replace("\\", "/")
        search_text = match.group("search")
        replacement_text = match.group("replace")

        if (
            not path
            or path.startswith("/")
            or ".." in path.split("/")
            or path not in updated
        ):
            return None, [], f"Invalid or unknown edit path: {path!r}"

        if not search_text:
            return None, [], f"Empty SEARCH block for {path}."

        occurrences = updated[path].count(search_text)

        if occurrences != 1:
            return (
                None,
                [],
                f"SEARCH block for {path} matched {occurrences} times; "
                "exactly one match is required.",
            )

        updated[path] = updated[path].replace(
            search_text,
            replacement_text,
            1,
        )

        changed_paths.append(path)

    return updated, sorted(set(changed_paths)), None

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


def write_code(
    llm: LLMClient,
    plan: dict,
    files: dict,
    feedback,
    attempt: int,
):
    import logging

    logger = logging.getLogger(__name__)

    compact_plan = {
        "title": plan.get("title", ""),
        "tasks": [
            task.get("title", "")
            if isinstance(task, dict)
            else str(task)
            for task in plan.get("tasks", [])
        ],
        "acceptance_criteria": plan.get("acceptance_criteria", []),
    }

    plan_context = json.dumps(
        compact_plan if attempt >= 2 else plan,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    feedback_text = (
        "\n\nReviewer or static-check feedback to fix:\n"
        + "\n".join(feedback)
        if feedback
        else ""
    )

    # ---------------------------------------------------------
    # Retry attempts: ask for SEARCH / REPLACE edit blocks.
    # ---------------------------------------------------------
    if attempt >= 2 and files:
        edit_user = (
            f"Plan:\n{plan_context}\n\n"
            f"Current files:\n{render_files(files, 12000)}"
            f"{feedback_text}\n\n"
            "Modify the existing files to address the feedback. "
            "Return ONLY SEARCH/REPLACE edit blocks; do not return "
            "complete files or Markdown code fences.\n\n"
            "Use this exact format for each edit:\n"
            "### EDIT: path/to/file\n"
            "<<<<<<< SEARCH\n"
            "exact existing lines to find\n"
            "=======\n"
            "replacement lines\n"
            ">>>>>>> REPLACE\n\n"
            "Rules:\n"
            "- SEARCH must match the existing file text exactly.\n"
            "- Keep SEARCH blocks as small as possible.\n"
            "- Include enough surrounding lines to make each match unique.\n"
            "- Change only what is necessary to fix the reported issues.\n"
            "- Do not output a block for an unchanged file.\n"
            "- Do not invent files or existing code.\n"
            "- If no edit is necessary, explain this in a concise NOTES line."
        )

        edit_text = llm.chat_text(
            "coder",
            CODER_SYS,
            edit_user,
            max_tokens=1200,
            temperature=0.2,
        )

        updated_files, changed_paths, edit_error = apply_edit_blocks(
            edit_text,
            files,
        )

        if updated_files is not None:
            new_files = {
                path: updated_files[path]
                for path in changed_paths
            }

            if new_files:
                logger.info(
                    "Coder applied edits to %d file(s): %s",
                    len(new_files),
                    changed_paths,
                )

                notes = re.search(
                    r"^NOTES:\s*(.+)$",
                    edit_text,
                    re.M,
                )

                return (
                    new_files,
                    notes.group(1).strip() if notes else "",
                )

            edit_error = (
                "The edit response contained no applicable file changes."
            )

        logger.warning(
            "Coder edit application failed; falling back to a full rewrite: %s",
            edit_error,
        )

    # ---------------------------------------------------------
    # First attempt, or fallback when edits cannot be applied:
    # request complete files.
    # ---------------------------------------------------------
    user = (
        f"Plan:\n{plan_context}\n\n"
        f"Current files:\n{render_files(files, 12000)}"
        f"{feedback_text}\n\n"
        "Output each complete file as `### FILE: path` followed by "
        "one fenced code block containing the full file. "
        "Output nothing else except optional `NOTES: ...`."
    )

    for format_attempt in range(2):
        text = llm.chat_text(
            "coder",
            CODER_SYS,
            user,
            max_tokens=2800,
            temperature=0.2,
        )

        new_files = parse_files(text)

        if new_files:
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

            return (
                new_files,
                notes.group(1).strip() if notes else "",
            )

        if format_attempt == 0:
            user += (
                "\n\nYour previous response did not follow the required "
                "`### FILE:` format. Return the complete files again "
                "using that exact format."
            )

    raise LLMError("the Coder returned no parseable files")

def review(
    llm: LLMClient,
    plan: dict,
    files: dict,
    lang: str,
    attempt: int = 1,
    previous_files: dict | None = None,
    previous_issues: list[str] | None = None,
) -> dict:
    review_files = {
        path: content
        for path, content in files.items()
        if path.rsplit("/", 1)[-1].lower() != "readme.md"
    }

    # Keep only task titles and acceptance criteria.
    # Task details and assumptions are unnecessary for this review.
    compact_plan = {
        "title": plan.get("title", ""),
        "tasks": [
            task.get("title", "")
            if isinstance(task, dict)
            else str(task)
            for task in plan.get("tasks", [])
        ],
        "acceptance_criteria": plan.get("acceptance_criteria", []),
    }

    plan_json = json.dumps(
        compact_plan,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    if attempt >= 2 and previous_files is not None:
        # Build a unified diff rather than resending complete files.
        diff_parts = []

        for path in sorted(set(previous_files) | set(review_files)):
            old_content = previous_files.get(path)
            new_content = review_files.get(path)

            if old_content == new_content:
                continue

            diff_parts.extend(
                difflib.unified_diff(
                    (old_content or "").splitlines(keepends=True),
                    (new_content or "").splitlines(keepends=True),
                    fromfile=f"a/{path}" if old_content is not None else "/dev/null",
                    tofile=f"b/{path}" if new_content is not None else "/dev/null",
                )
            )

        diff_text = "".join(diff_parts) or "(No source changes detected.)"

        previous_issues_json = json.dumps(
            previous_issues or [],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        implementation_context = (
            "Review attempt: "
            f"{attempt}\n\n"
            "Previous issues that must be verified as fixed:\n"
            f"{previous_issues_json}\n\n"
            "Changes since the previous attempt:\n"
            f"{diff_text}"
        )

    else:
        implementation_context = (
            "Automated syntax checks: passed.\n\n"
            "Inspect the actual source files and tests provided below.\n\n"
            f"Files:\n{render_files(review_files, 9000)}"
        )

    user = (
        f"Language for text fields: {LANG_NAMES[lang]}\n\n"
        "Task titles and acceptance criteria:\n"
        f"{plan_json}\n\n"
        f"{implementation_context}\n\n"
        "Review the implementation against the acceptance criteria. "
        "On retry attempts, focus on whether the previous issues are fixed "
        "and whether the supplied changes introduce new issues. "
        "Do not report a file as missing if it is present in the supplied "
        "context. Do not invent requirements. If the supplied context is "
        "insufficient to verify a criterion, state that limitation "
        "concisely. Do not reproduce source code. "
        "Return only actionable implementation issues and a concise "
        "approval decision."
    )

    r = llm.chat_json(
        "reviewer",
        REVIEWER_SYS,
        user,
        required=("approved",),
        max_tokens=500,
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
        "Write the download-and-extract instructions in the requested language. "
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

    readme_lower = readme.lower()
    mentions_download = "download" in readme_lower or "descarg" in readme_lower
    mentions_continuation = (
        "continue" in readme_lower or "contin" in readme_lower
    )
    if (
        not mentions_download
        or "project.zip" not in readme_lower
        or PROJECT_EXTRACT_COMMAND.lower() not in readme_lower
        or not mentions_continuation
    ):
        if lang == "es":
            readme += (
                "\n## Descargar y descomprimir el proyecto\n\n"
                "Descarga `project.zip` desde el chat y guárdalo en la "
                "ubicación donde quieras crear el proyecto. Abre una terminal "
                "en esa ubicación y ejecuta:\n\n"
                f"```bash\n{PROJECT_EXTRACT_COMMAND}\n```\n\n"
                "Después, continúa con los pasos de instalación, pruebas y "
                "ejecución de este README desde la carpeta del proyecto.\n"
            )
        else:
            readme += (
                "\n## Download and extract the project\n\n"
                "Download `project.zip` from the chat and save it where you "
                "want to create the project. Open a terminal there and run:\n\n"
                f"```bash\n{PROJECT_EXTRACT_COMMAND}\n```\n\n"
                "Then continue with the installation, test, and run steps "
                "in this README from the project folder.\n"
            )

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
    """Check that the README documents project extraction and detected commands."""
    readme = next(
        (
            content for path, content in files.items()
            if path.rsplit("/", 1)[-1].lower() == "readme.md"
        ),
        None,
    )
    if readme is None:
        return ["README.md is missing."]

    readme_lower = readme.lower()
    mentions_download = "download" in readme_lower or "descarg" in readme_lower
    mentions_continuation = (
        "continue" in readme_lower or "contin" in readme_lower
    )
    if (
        not mentions_download
        or "project.zip" not in readme_lower
        or PROJECT_EXTRACT_COMMAND.lower() not in readme_lower
        or not mentions_continuation
    ):
        return [
            "README.md must explicitly explain downloading project.zip, "
            f"extracting it with `{PROJECT_EXTRACT_COMMAND}`, and continuing "
            "with the remaining setup steps."
        ]

    commands = get_deterministic_commands(files)
    required_commands = commands["install"] + commands["run"] + commands["test"]
    missing = [command for command in required_commands if command.lower() not in readme.lower()]

    if missing:
        return ["README.md must document these detected commands: " + ", ".join(missing)]
    return []