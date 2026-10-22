"""The agent roles. Each one is a prompt plus a small function with a strict input/output contract."""
from __future__ import annotations

import ast
import json
import re

from .llm import LLMClient, LLMError

LANG_NAMES = {"es": "Spanish", "en": "English"}
INTENTS = ("new_ticket", "change_request", "question", "chat", "clarification")

ROUTER_SYS = (
    "You route messages in a chat with a code-delivery assistant. Given the recent conversation and a "
    "summary of the last delivered work (or 'none'), classify the latest user message.\n"
    "Intents: \n"
    "- new_ticket (a clear feature request unrelated to previous work),\n"
    "- change_request (modify, extend or fix the previous work),\n"
    "- question (asks about delivered work, the plan or past decisions),\n"
    "- chat (greeting or off-topic),\n"
    "- clarification (the user's request is too ambiguous, lacks essential technical details, or requires critical choices before planning).\n"
    'Reply ONLY JSON: {"intent": "...", "lang": "es" or "en", "reason": "short"}'
)

PLANNER_SYS = (
    "You are the Planner of a software team. Turn the request into a concrete technical plan. Tickets may be "
    "short or ambiguous: do NOT ask questions. Choose the most reasonable interpretation and list it under "
    "assumptions. If previous work exists and the request is a change, plan only the delta and keep what "
    "already works. Use the language named in the ticket, otherwise Python. Write text fields in the "
    "requested language; code stays in English.\n"
    'Reply ONLY JSON: {"title": str, "language": str, "assumptions": [str], '
    '"tasks": [{"id": int, "title": str, "details": str}], "acceptance_criteria": [str]}. '
    "Use 3-6 tasks and 3-6 acceptance criteria."
)

CODER_SYS = (
    "You are the Coder. Implement the plan completely so that it runs. Output every file as a line "
    "`### FILE: path` followed by exactly one fenced code block with the full file content. Nothing else "
    "except an optional last line `NOTES: ...`. If the language is Python, use only the standard library "
    "unless the plan says otherwise and include a small unittest file covering each acceptance criterion. "
    "If current files or reviewer feedback are provided, return the full corrected version of every file "
    "that changes; files you omit are kept as they are."
)

REVIEWER_SYS = (
    "You are the Reviewer. Judge the code against the plan's tasks and acceptance criteria: correctness, "
    "edge cases named in the criteria, and bugs likely to crash. Automated syntax checks are provided. List "
    "only BLOCKING issues (a criterion not met or likely failure); leave style remarks out. Approve when no "
    "blocking issue remains. Write issues and feedback in the requested language.\n"
    'Reply ONLY JSON: {"approved": true or false, "issues": [str], "feedback": "specific, actionable '
    'instructions for the Coder"}'
)

ASSISTANT_SYS = (
    "You are the assistant of a code-delivery chat. Answer the user's message using the delivered work, the "
    "plan and the conversation. Be concise and answer in the user's language. If nothing has been delivered "
    "yet, say so and invite the user to send a ticket.\n\n"
    "CRITICAL DELIVERY INSTRUCTIONS:\n"
    "When code has been successfully delivered, approved, or when the user asks how to proceed, your response MUST include:\n"
    "1. **Árbol de archivos**: Una representación visual limpia de la estructura de carpetas y archivos entregados.\n"
    "2. **Paso a paso**: Instrucciones claras, prácticas y secuenciales para que el usuario pueda instalar, ejecutar, probar o continuar trabajando con el código en su entorno local."
)


def answer(llm: LLMClient, message: str, history: list, plan, files: dict, open_issues: list) -> str:
    # Generamos el árbol de archivos para inyectarlo en el contexto del asistente
    file_tree = generate_file_tree(files) if files else "none"
    
    user = (
        f"Conversation:\n{_history_text(history)}\n\n"
        f"Last plan: {json.dumps(plan, ensure_ascii=False) if plan else 'none'}\n\n"
        f"File Tree Structure:\n{file_tree}\n\n"
        f"Delivered files:\n{render_files(files, 8000)}\n\n"
        f"Unresolved issues: {open_issues or 'none'}\n\nUser message:\n{message}"
    )
    # Aumentamos ligeramente max_tokens para asegurar que entre el paso a paso detallado
    return llm.chat_text("assistant", ASSISTANT_SYS, user, max_tokens=1000, temperature=0.3)


def _history_text(history: list) -> str:
    return "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-6:]) or "(empty)"


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


def static_checks(files: dict) -> list:
    """Deterministic checks that run before the Reviewer (and save a model call when they fail)."""
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
    r = llm.chat_json("router", ROUTER_SYS, user, required=("intent",), max_tokens=150)
    intent = r["intent"] if r.get("intent") in INTENTS else "new_ticket"
    lang = "es" if r.get("lang") == "es" else "en"
    return {"intent": intent, "lang": lang, "reason": str(r.get("reason", ""))[:200]}


def make_plan(llm: LLMClient, request: str, lang: str, prior_plan, files: dict, open_issues: list) -> dict:
    context = ""
    if prior_plan:
        context = (
            f"Previous plan:\n{json.dumps(prior_plan, ensure_ascii=False)}\n\n"
            f"Current files:\n{render_files(files, 8000)}\n\n"
        )
        if open_issues:
            context += "Unresolved reviewer issues:\n- " + "\n- ".join(open_issues) + "\n\n"
    user = f"{context}Language for text fields: {LANG_NAMES[lang]}\nRequest:\n{request}"
    plan = llm.chat_json("planner", PLANNER_SYS, user, required=("tasks", "acceptance_criteria"), max_tokens=900)
    plan.setdefault("title", "Untitled")
    plan.setdefault("language", "Python")
    plan.setdefault("assumptions", [])
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

def generate_file_tree(files: dict) -> str:
    """Genera una representación visual tipo árbol de archivos a partir de las rutas."""
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

def review(llm: LLMClient, plan: dict, files: dict, lang: str) -> dict:
    user = (
        f"Language for text fields: {LANG_NAMES[lang]}\n"
        f"Plan:\n{json.dumps(plan, ensure_ascii=False)}\n\n"
        f"Automated syntax checks: passed\n\nFiles:\n{render_files(files)}"
    )
    r = llm.chat_json("reviewer", REVIEWER_SYS, user, required=("approved",), max_tokens=1000)
    raw_issues = r.get("issues")
    issues = [str(i).strip() for i in raw_issues if str(i).strip()] if isinstance(raw_issues, list) else []
    approved = (r.get("approved") is True or str(r.get("approved")).lower() == "true") and not issues
    feedback = str(r.get("feedback", "")).strip() or "; ".join(issues)
    return {"approved": approved, "issues": issues, "feedback": feedback}


def answer(llm: LLMClient, message: str, history: list, plan, files: dict, open_issues: list) -> str:
    user = (
        f"Conversation:\n{_history_text(history)}\n\n"
        f"Last plan: {json.dumps(plan, ensure_ascii=False) if plan else 'none'}\n\n"
        f"Delivered files:\n{render_files(files, 8000)}\n\n"
        f"Unresolved issues: {open_issues or 'none'}\n\nUser message:\n{message}"
    )
    return llm.chat_text("assistant", ASSISTANT_SYS, user, max_tokens=700, temperature=0.3)
