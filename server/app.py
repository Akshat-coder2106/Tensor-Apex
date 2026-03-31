from __future__ import annotations

import os
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException

from business_policy_env.models import Action, Observation, ResetRequest, StepRequest, StepResult
from business_policy_env.server import _get_or_create, _session_or_default, _sessions

APP_NAME = "Business Policy Compliance and Customer Resolution Environment"
APP_DESCRIPTION = (
    "An OpenEnv-style environment for policy-aware customer support reasoning under uncertainty."
)
APP_VERSION = "1.0.0"

app = FastAPI(title=APP_NAME, description=APP_DESCRIPTION, version=APP_VERSION)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict[str, str]:
    return {"name": APP_NAME, "description": APP_DESCRIPTION}


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
    return env.reset(task_name=payload.task_name, scenario_id=payload.scenario_id)


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


@app.delete("/session")
def close_session(x_session_id: str | None = Header(default=None)) -> dict[str, str]:
    session_id = _session_or_default(x_session_id)
    env = _sessions.pop(session_id, None)
    if env is not None:
        env.close()
    return {"closed": session_id}


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


@app.post("/mcp")
def mcp(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    request_id = payload.get("id") if isinstance(payload, dict) else None
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "name": APP_NAME,
            "version": APP_VERSION,
            "mode": "simulation",
        },
    }


def main() -> None:
    """Launch the FastAPI environment server."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
