# Ticket2Code AI Crew

Ticket2Code AI Crew turns a natural-language software ticket into a reviewed
project. A browser chat streams progress from a small multi-agent workflow,
displays the generated files individually, and offers the complete delivery as
`project.zip`.

## How it works

1. **Router** classifies the request as a new ticket, a change to the existing
   delivery, a question, ordinary chat, or a request for clarification.
2. **Planner** creates a plan and acceptance criteria. For a change request, it
   receives the previous plan, files, and unresolved review issues.
3. **Coder** generates project files. On retries, it can return exact
   `SEARCH`/`REPLACE` edits for existing files.
4. **Static checks** validate generated Python syntax and JSON before review.
5. **Reviewer** checks the implementation against the plan. Code generation,
   checks, and review repeat up to `MAX_RETRIES + 1` total attempts.
6. **Documentarian** writes a README only after implementation approval.
   Deterministic documentation checks require project ZIP extraction
   instructions and verify detected setup, test, and run commands.
7. The API streams newline-delimited JSON events to the browser. The UI shows
   live activity, generated files, and a ZIP download link.

Questions about a delivered project are answered from its conversation and
current session state without starting a new implementation run.

## Project layout

| Path | Purpose |
| --- | --- |
| `app/main.py` | FastAPI application, streamed chat endpoint, health/trace endpoints, and project ZIP downloads. |
| `app/pipeline.py` | Orchestrates routing, planning, coding, validation, review, documentation, and session updates. |
| `app/agents.py` | Agent prompts, model response parsing, file/edit handling, static checks, README generation, and documentation checks. |
| `app/llm.py` | OpenAI-compatible model clients, provider fallback, Groq rate-limit protection, retries, and request logging. |
| `app/config.py` | Loads application and provider settings from environment variables and `.env`. |
| `app/session.py` | In-memory session state and bounded session storage. |
| `app/__init__.py` | Python package marker for the application. |
| `static/index.html` | Single-page chat interface, live activity display, generated-file rendering, and ZIP link. |
| `tests/test_api.py` | FastAPI endpoint and ZIP download tests. |
| `tests/test_pipeline.py` | Agent orchestration, output formatting, and documentation behavior tests. |
| `tests/test_llm.py` | LLM request and logging tests using a fake client. |
| `tests/fakes.py` | Scripted LLM stand-in for offline tests. |
| `tests/__init__.py` | Python package marker for the test suite. |
| `.env.example` | Example Groq primary and Ollama fallback settings; copy it to `.env` and provide a primary API key or switch the primary settings to Ollama. |
| `.gitignore` | Excludes local environments, credentials, and generated Python/test cache files from Git. |
| `requirements.txt` | Runtime dependencies. |
| `requirements-dev.txt` | Runtime dependencies plus development/test packages. |
| `pytest.ini` | Pytest discovery and import-path configuration. |
| `LICENSE` | Project license. |

## Requirements

- Python 3.10 or later.
- Ollama running locally with a model pulled if you want to run the app using
  local inference.
- Alternatively, an API key and base URL for an OpenAI-compatible hosted
  provider. The application requires a configured primary provider; a fallback
  provider is optional.

## Run locally with Ollama

Install Ollama and make sure its service is running. If it is not already
running as a system service, start it in a separate terminal:

```bash
ollama serve
```

Pull a model supported by your machine. This is an example; substitute another
Ollama model if desired:

```bash
ollama pull qwen2.5-coder:7b
```

In a second terminal, prepare the application:

```bash
git clone https://github.com/diegoquezadaco/ticket2code-ai-crew.git
cd ticket2code-ai-crew
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Edit `.env` and configure Ollama as the primary provider. The placeholder
`ollama` key is accepted by Ollama's OpenAI-compatible API and is not a secret:

```dotenv
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5-coder:7b
PLANNER_MODEL=
CODER_MODEL=
REVIEWER_MODEL=
DOCUMENTARIAN_MODEL=
```

The model value must match a model installed in Ollama (`ollama list`). If the
app runs in a container or on another machine, `localhost` refers to that
container or machine; set `LLM_BASE_URL` to an address reachable from the app
instead.

Start the web application:

```bash
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Submit a ticket, watch the streamed agent activity,
and use **Descargar proyecto ZIP** to download the files from the latest
delivery.

### Optional hosted primary and local Ollama fallback

The fallback is used only after the primary provider fails. To use a hosted
primary with Ollama as fallback, configure the primary credentials and set:

```dotenv
FALLBACK_API_KEY=ollama
FALLBACK_BASE_URL=http://localhost:11434/v1
FALLBACK_MODEL=qwen2.5-coder:7b
```

The checked-in `.env.example` currently includes an Ollama fallback. Its
`FALLBACK_TIMEOUT` entry does not match the setting read by the application;
use `FALLBACK_TIMEOUT_SECONDS` to override the fallback timeout. Configuration
defaults in the table below come from `app/config.py`; `.env.example` may
override those defaults.

## Run tests

Tests use scripted fake model responses, so they do not call a hosted provider
or require Ollama to be running:

```bash
source .venv/bin/activate
python -m pytest -q
```

The repository's test suite covers API behavior, streaming events, ZIP archive
contents, pipeline outcomes, generated README instructions, and LLM request
logging.

**Current validation note:** the full suite currently reports 7 failing tests
and 15 passing tests. Failures include an empty persisted trace, scripted
fixtures that do not provide Documentarian responses, and pipeline retry/error
expectations that do not match the current behavior. Treat a passing focused
test as partial validation until these cases are reconciled.

## Configuration

All settings can be provided in `.env` or through the process environment.
Environment variables set by the deployment environment take precedence over
values loaded from `.env`.

| Variable | Purpose | Default |
| --- | --- | --- |
| `LLM_API_KEY` | Primary provider API key. For Ollama, use the placeholder `ollama`. | Empty |
| `LLM_BASE_URL` | Primary OpenAI-compatible API base URL. | `https://api.groq.com/openai/v1` |
| `LLM_MODEL` | Default model and fallback for roles without a role-specific model. | `openai/gpt-oss-20b` |
| `PLANNER_MODEL` | Optional model override for the Planner. | `LLM_MODEL` |
| `CODER_MODEL` | Optional model override for the Coder. | `LLM_MODEL` |
| `REVIEWER_MODEL` | Optional model override for the Reviewer. | `LLM_MODEL` |
| `DOCUMENTARIAN_MODEL` | Optional model override for the Documentarian. | `LLM_MODEL` |
| `FALLBACK_API_KEY` | Optional fallback provider key; defaults to `ollama` when a fallback URL and model are set. | Empty |
| `FALLBACK_BASE_URL` | Optional fallback OpenAI-compatible API base URL. | Disabled |
| `FALLBACK_MODEL` | Fallback model name. A fallback is enabled only when both its URL and model are set. | Disabled |
| `MAX_RETRIES` | Additional implementation retries after the first attempt. | `5` |
| `LLM_TIMEOUT_SECONDS` | Timeout for primary-provider requests. | `60` |
| `FALLBACK_TIMEOUT_SECONDS` | Timeout for fallback-provider requests. | `380` |
| `MAX_SESSIONS` | Maximum number of in-memory sessions retained by the process. | `400` |
| `GROQ_TPM_BUDGET` | Default estimated tokens-per-minute budget for Groq. | `5900` |
| `GROQ_TPM_BUDGET_<MODEL>` | Optional per-model Groq token budget. Replace non-alphanumeric model-name characters with underscores and use uppercase. | `GROQ_TPM_BUDGET` |
| `GROQ_WAIT_SECONDS` | Maximum wait for Groq capacity and a per-model request slot. | `45` |
| `GROQ_COOLDOWN_SECONDS` | Default Groq cooldown after a rate-limit response. | `60` |

The application disables SDK-managed retries and applies its own bounded retry
policy to Groq requests. It estimates request tokens for Groq rate-limit
protection; the estimate is not an exact tokenizer count.

## HTTP endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Serves the chat interface. |
| `GET` | `/health` | Reports application health and whether the primary LLM client is configured. |
| `POST` | `/api/chat` | Accepts a session ID and user message; streams newline-delimited JSON log and final/error events. |
| `GET` | `/api/sessions/{session_id}/trace` | Returns trace entries recorded for a session. |
| `GET` | `/api/sessions/{session_id}/download` | Downloads the latest generated session files as `project.zip`. |

The trace endpoint is present, but the current pipeline does not persist turn
events into `Session.runs`. Use the live chat stream to inspect agent activity.

Chat request JSON:

```json
{
  "session_id": "example-session-123",
  "message": "Add input validation to the signup form"
}
```

## Deployment draft

There is no container image, deployment manifest, persistent database, or
hosting-provider configuration in this repository. Adapt this checklist to
the hosting platform you choose:

1. Provision a Python 3.10+ web service from this repository.
2. Install runtime dependencies with `python -m pip install -r requirements.txt`.
3. Set the platform start command to
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT` (or the platform's
   equivalent port variable).
4. Configure `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` as platform
   environment variables. Add role-specific models or fallback settings as
   required. Store real provider credentials as platform secrets, not in the
   repository.
5. If using Ollama, host it as a separate service reachable from the web
   service and point the relevant base URL to that service. Do not configure
   `localhost` unless Ollama is in the same network namespace as the app.
6. Configure the platform health check to use `/health`.
7. Verify a complete ticket run, individual file display, ZIP download, and
   model fallback behavior in the deployed environment.

**Scaling limitation:** sessions, conversation history, generated files, and
Groq request coordination are held in process memory. A restart loses session
data, and multiple workers or replicas do not share sessions or rate-limit
state. Use a single worker for the current design; before horizontal scaling,
move session/file storage and shared rate-limit coordination to shared
services, then validate session affinity and ZIP downloads.
