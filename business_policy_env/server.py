from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse

from .environment import BusinessPolicyComplianceEnv
from .landing import build_landing_page
from .models import Action, Observation, ResetRequest, StepRequest, StepResult

app = FastAPI(
    title="Business Policy Compliance and Customer Resolution Environment",
    description="An OpenEnv-style environment for policy-aware customer support reasoning under uncertainty.",
    version="1.0.0",
)

_sessions: dict[str, BusinessPolicyComplianceEnv] = {}


def _get_or_create(session_id: str) -> BusinessPolicyComplianceEnv:
    if session_id not in _sessions:
        _sessions[session_id] = BusinessPolicyComplianceEnv()
    return _sessions[session_id]


def _session_or_default(x_session_id: str | None) -> str:
    return x_session_id or "default"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return build_landing_page(app_name=app.title, app_version=app.version)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks(x_session_id: str | None = Header(default=None)) -> dict[str, list[str]]:
    session_id = _session_or_default(x_session_id)
    env = _get_or_create(session_id)
    return env.available_tasks()


@app.post("/reset", response_model=Observation)
def reset(
    request: ResetRequest | None = None,
    x_session_id: str | None = Header(default=None),
) -> Observation:
    session_id = _session_or_default(x_session_id)
    env = _get_or_create(session_id)
    payload = request or ResetRequest()
    return env.reset(
        task_name=payload.task_name,
        scenario_id=payload.scenario_id,
        variation_seed=payload.variation_seed,
    )


@app.get("/reset", response_model=Observation)
def reset_get(
    task_name: str | None = None,
    scenario_id: str | None = None,
    variation_seed: int | None = None,
    x_session_id: str | None = Header(default=None),
) -> Observation:
    session_id = _session_or_default(x_session_id)
    env = _get_or_create(session_id)
    resolved_task = task_name if task_name in {"easy", "medium", "hard"} else None
    return env.reset(task_name=resolved_task, scenario_id=scenario_id, variation_seed=variation_seed)


@app.post("/step", response_model=StepResult)
def step(
    request: StepRequest,
    x_session_id: str | None = Header(default=None),
) -> StepResult:
    session_id = _session_or_default(x_session_id)
    if session_id not in _sessions:
        raise HTTPException(status_code=400, detail="Session not found. Call /reset first.")
    env = _sessions[session_id]
    observation, reward, done, info = env.step(request.action)
    return StepResult(observation=observation, reward=reward, done=done, info=info)


@app.get("/state")
def state(
    x_session_id: str | None = Header(default=None),
    include_ground_truth: bool = False,
) -> dict[str, Any]:
    session_id = _session_or_default(x_session_id)
    if session_id not in _sessions:
        return {"active": False, "detail": "Session not found. Call /reset first."}
    return _sessions[session_id].state(include_ground_truth=include_ground_truth)


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {
            "type": "object",
            "additionalProperties": True,
            "description": "Environment state payload returned by GET /state.",
        },
    }


@app.delete("/session")
def close_session(x_session_id: str | None = Header(default=None)) -> dict[str, str]:
    session_id = _session_or_default(x_session_id)
    env = _sessions.pop(session_id, None)
    if env is not None:
        env.close()
    return {"closed": session_id}
