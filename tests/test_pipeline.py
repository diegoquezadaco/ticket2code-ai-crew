from dataclasses import replace

from app import agents
from app.config import get_settings
from app.llm import LLMError, parse_json
from app.pipeline import format_reply, run_turn
from app.session import Session

from .fakes import APPROVE, BROKEN_CODE, GOOD_CODE, PLAN, REJECT, FakeLLM, route


def run(llm, session=None, message="add coupons", retries=2):
    settings = replace(get_settings(), max_retries=retries)
    session = session or Session(id="test-session")
    return list(run_turn(session, message, llm, settings)), session


def final(events):
    return events[-1]


def test_approved_on_first_attempt():
    llm = FakeLLM({"router": [route()], "planner": [PLAN], "coder": [GOOD_CODE], "reviewer": [APPROVE]})
    events, session = run(llm)
    assert final(events)["status"] == "approved"
    assert final(events)["attempts"] == 1
    assert "coupons.py" in session.files


def test_reviewer_feedback_reaches_the_coder_on_retry():
    llm = FakeLLM({"router": [route()], "planner": [PLAN], "coder": [GOOD_CODE],
                   "reviewer": [REJECT, APPROVE]})
    events, _ = run(llm)
    coder_inputs = [u for r, u in llm.calls if r == "coder"]
    assert len(coder_inputs) == 2
    assert "missing expiry check" in coder_inputs[1]
    assert final(events)["status"] == "approved" and final(events)["attempts"] == 2


def test_exhausted_retries_are_reported_not_hidden():
    llm = FakeLLM({"router": [route()], "planner": [PLAN], "coder": [GOOD_CODE], "reviewer": [REJECT]})
    events, session = run(llm, retries=1)
    last = final(events)
    assert last["status"] == "not_approved"
    assert last["attempts"] == 2 == last["max_attempts"]
    assert "NOT APPROVED" in last["reply"] and "expiry is not checked" in last["reply"]
    assert session.open_issues == ["expiry is not checked"]
    assert any(e.get("agent") == "system" and "limit" in e["message"] for e in events)


def test_spanish_label_when_user_writes_spanish():
    llm = FakeLLM({"router": [route(lang="es")], "planner": [PLAN], "coder": [GOOD_CODE], "reviewer": [REJECT]})
    events, _ = run(llm, retries=0)
    assert "NO APROBADO" in final(events)["reply"]


def test_syntax_error_skips_reviewer_and_forces_retry():
    llm = FakeLLM({"router": [route()], "planner": [PLAN], "coder": [BROKEN_CODE, GOOD_CODE],
                   "reviewer": [APPROVE]})
    events, _ = run(llm)
    assert llm.roles_called().count("reviewer") == 1  # only the fixed version was reviewed
    assert final(events)["status"] == "approved" and final(events)["attempts"] == 2


def test_follow_up_modifies_previous_work():
    llm = FakeLLM({"router": [route(), route("change_request")], "planner": [PLAN], "coder": [GOOD_CODE],
                   "reviewer": [APPROVE]})
    session = Session(id="test-session")
    run(llm, session, "add coupons")
    run(llm, session, "also add a usage limit")
    planner_inputs = [u for r, u in llm.calls if r == "planner"]
    coder_inputs = [u for r, u in llm.calls if r == "coder"]
    assert "Previous plan" in planner_inputs[1]
    assert "coupons.py" in coder_inputs[1]  # the Coder sees the files already delivered


def test_question_does_not_trigger_the_pipeline():
    llm = FakeLLM({"router": [route(), route("question")], "planner": [PLAN], "coder": [GOOD_CODE],
                   "reviewer": [APPROVE], "assistant": ["It checks expiry."]})
    session = Session(id="test-session")
    run(llm, session, "add coupons")
    events, _ = run(llm, session, "what did you deliver?")
    assert final(events)["status"] == "answer"
    assert llm.roles_called().count("coder") == 1


def test_model_failure_is_explained_with_the_failing_stage():
    llm = FakeLLM({"router": [route()], "planner": [LLMError("HTTP 404: model not found")]})
    events, session = run(llm)
    assert final(events) == {"type": "error", "stage": "planner", "message": "HTTP 404: model not found"}
    assert session.runs[-1]["events"]  # trace is still kept


def test_unexpected_exception_becomes_an_error_event():
    llm = FakeLLM({"router": [route()], "planner": [RuntimeError("boom")]})
    events, _ = run(llm)
    assert final(events)["type"] == "error" and "RuntimeError" in final(events)["message"]


def test_parse_files_rejects_unsafe_paths():
    text = "### FILE: ../evil.py\n```python\nx=1\n```\n### FILE: ok.py\n```python\ny=2\n```"
    assert list(agents.parse_files(text)) == ["ok.py"]


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('Here:\n```json\n{"a": 1}\n```') == {"a": 1}


def test_spanish_delivery_steps_start_with_zip_download_and_extraction():
    reply = format_reply(
        lang="es",
        approved=True,
        attempts=1,
        plan={},
        files={"app.py": "print('hello')\n"},
        issues=[],
    )

    steps = reply.split("### Paso a paso para continuar:\n", 1)[1]
    assert "1. Descarga `project.zip`" in steps
    assert "2. Abre una terminal" in steps
    assert "unzip project.zip -d project && cd project" in steps
    assert "3. Create and activate a virtual environment:" in steps


def test_readme_checks_require_zip_download_and_extraction_instructions():
    files = {
        "app.py": "print('hello')\n",
        "README.md": (
            "Download project.zip, then run "
            "`unzip project.zip -d project && cd project`, then continue "
            "with the setup steps below."
        ),
    }
    assert agents.documentation_checks({}, files) == []

    files["README.md"] = "# App\n"
    issues = agents.documentation_checks({}, files)
    assert len(issues) == 1
    assert "project.zip" in issues[0]


def test_document_project_adds_download_instructions_when_missing():
    llm = FakeLLM({"documentarian": ["# Generated app"]})
    readme = agents.document_project(
        llm,
        plan={},
        files={"app.py": "print('hello')\n"},
        lang="en",
    )

    assert "Download `project.zip` from the chat" in readme
    assert "unzip project.zip -d project && cd project" in readme
    assert "Then continue with the installation" in readme
