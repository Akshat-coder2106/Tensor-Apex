#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from statistics import mean
from typing import Any

import httpx
from openai import OpenAI

from business_policy_env.models import Action, Observation

TASK_NAME = "all"
BENCHMARK = "business-policy-compliance"
MAX_STEPS = 16
MAX_TOTAL_REWARD = 10.0
SUCCESS_SCORE_THRESHOLD = 0.6
MAX_RUNTIME_SECONDS = 18 * 60
DEFAULT_ENV_URL = "https://aaditya-03-tensor-apex-openenv.hf.space"

ACTION_REQUIREMENTS: dict[str, str | None] = {
    "categorize": "category",
    "set_priority": "priority",
    "draft_response": "response_text",
    "escalate": "escalation_reason",
    "mark_spam": None,
    "request_info": "clarifying_question",
    "flag_fraud": "fraud_reason",
    "snooze": "snooze_hours",
    "consult_specialist": "specialist_question",
}

SYSTEM_PROMPT = """You are a policy-aware customer support agent.
Read the observation carefully and choose the next best action.
Respond with JSON only. Do not include markdown, explanations outside the JSON object, or extra keys.
"""


def _empty_summary() -> dict[str, dict[str, float]]:
    return {
        "easy": {"mean": 0.0, "min": 0.0, "max": 0.0},
        "medium": {"mean": 0.0, "min": 0.0, "max": 0.0},
        "hard": {"mean": 0.0, "min": 0.0, "max": 0.0},
    }


def _bool_str(value: bool) -> str:
    return str(value).lower()


def _log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    action_clean = action.replace("\n", " ").replace("\r", " ")[:100]
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={_bool_str(done)} error={error_value}",
        flush=True,
    )


def _log_end(task: str, success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards) or "0.00"
    print(
        f"[END] task={task} score={score:.2f} steps={steps} success={_bool_str(success)} rewards={rewards_text}",
        flush=True,
    )


def _summary_score(summary: dict[str, dict[str, float]], selected_tasks: list[str]) -> float:
    values = [
        float(summary.get(task_name, {}).get("mean", 0.0))
        for task_name in selected_tasks
    ]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _load_dotenv() -> None:
    base_dir = Path(__file__).resolve().parent
    candidate_paths = [base_dir / ".env", base_dir.parent / ".env"]
    for dotenv_path in candidate_paths:
        if not dotenv_path.exists():
            continue
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
        return


def _extract_text(content: str | list[Any] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match is None:
        raise ValueError("No JSON object found in model response.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON response is not an object.")
    return parsed


def _coerce_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
    action_type = str(payload.get("action_type", "")).strip()
    if action_type not in ACTION_REQUIREMENTS:
        raise ValueError(f"Unsupported action_type: {action_type!r}")

    coerced: dict[str, Any] = {
        "action_type": action_type,
        "reasoning": str(payload.get("reasoning", "Model-selected action.")),
    }
    required_field = ACTION_REQUIREMENTS[action_type]
    if required_field is None:
        return coerced

    if required_field not in payload:
        raise ValueError(f"Missing required field: {required_field}")

    if required_field == "snooze_hours":
        coerced[required_field] = int(payload[required_field])
    else:
        coerced[required_field] = payload[required_field]
    return coerced


def _observation_payload(observation: Observation) -> dict[str, Any]:
    return {
        "thread": [
            {
                "direction": message.direction,
                "sender_name": message.sender_name,
                "sender_email": message.sender_email,
                "timestamp": message.timestamp.isoformat(),
                "subject": message.subject,
                "body": message.body,
            }
            for message in observation.thread
        ],
        "policy_rules": observation.policy_rules,
        "account_flags": observation.account_flags,
        "action_history": [
            {
                "step_index": record.step_index,
                "action_type": record.action.action_type,
                "reasoning": record.action.reasoning,
            }
            for record in observation.action_history
        ],
        "issue_age_hours": observation.issue_age_hours,
        "available_action_types": list(ACTION_REQUIREMENTS.keys()),
        "scenario_id": observation.scenario_id,
        "difficulty": observation.difficulty,
        "current_email": {
            "subject": observation.current_email.subject,
            "body": observation.current_email.body,
        },
        "sender_tier": observation.sender_tier,
        "refund_amount": observation.refund_amount,
        "policy_version": observation.policy_version,
        "specialist_feedback": observation.specialist_feedback,
        "attachment_present": observation.attachment_present,
        "attachment_summary": observation.attachment_summary,
        "attachment_signals": observation.attachment_signals,
        "agent_notes": observation.agent_notes,
        "task_objective": observation.task_objective,
        "clarification_received": observation.clarification_received,
        "emails_remaining": observation.emails_remaining,
        "episode_phase": observation.episode_phase,
        "steps_taken": observation.steps_taken,
        "max_steps": observation.max_steps,
    }


def _safe_default_action(observation: Observation) -> Action:
    completed_types = {record.action.action_type for record in observation.action_history}
    if "categorize" not in completed_types:
        return Action(
            action_type="categorize",
            reasoning="Fallback action chooses a neutral route so the episode can continue safely.",
            category="customer_success",
        )
    if "set_priority" not in completed_types:
        return Action(
            action_type="set_priority",
            reasoning="Fallback action sets a neutral default priority.",
            priority="medium",
        )
    return Action(
        action_type="draft_response",
        reasoning="Fallback action sends a short status update without making unsafe commitments.",
        response_text="We are reviewing the details now and will follow up with the next safe step shortly.",
    )


class OpenAIEnvironmentAgent:
    def __init__(self) -> None:
        client = OpenAI(
            base_url=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=os.environ.get("HF_TOKEN", "") or "missing-token",
        )
        model = os.environ.get("MODEL_NAME", "openai/gpt-4.1-mini")
        self._client = client.with_options(timeout=12.0, max_retries=0)
        self._model = model
        self._llm_available = True

    @property
    def model_name(self) -> str:
        return self._model

    def next_action(self, observation: Observation) -> Action:
        if not self._llm_available:
            return _safe_default_action(observation)

        prompt_payload = {
            "instruction": (
                "Return exactly one JSON object with keys action_type, reasoning, and the one required field "
                "for that action. Never return arrays, markdown fences, or prose outside the JSON object."
            ),
            "required_schema_examples": {
                "categorize": {
                    "action_type": "categorize",
                    "reasoning": "Route the ticket to the correct queue.",
                    "category": "billing",
                }
            },
            "observation": _observation_payload(observation),
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=True)},
        ]

        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                max_tokens=400,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except Exception:
            try:
                completion = self._client.chat.completions.create(
                    model=self._model,
                    temperature=0.0,
                    max_tokens=400,
                    messages=messages,
                )
            except Exception:
                self._llm_available = False
                return _safe_default_action(observation)

        raw_text = _extract_text(completion.choices[0].message.content if completion.choices else "")
        try:
            parsed = _extract_json_object(raw_text)
            coerced = _coerce_action_payload(parsed)
            return Action.model_validate(coerced)
        except Exception:
            return _safe_default_action(observation)


class HttpEnvironmentClient:
    def __init__(self, *, base_url: str, session_id: str) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=20.0)
        self._headers = {"X-Session-Id": session_id}

    def close(self) -> None:
        try:
            self._client.delete("/session", headers=self._headers)
        except Exception:
            pass
        self._client.close()

    def tasks(self) -> dict[str, list[str]]:
        response = self._client.get("/tasks", headers=self._headers)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Invalid /tasks response payload.")
        return {
            key: [str(item) for item in value]
            for key, value in payload.items()
            if isinstance(value, list)
        }

    def reset(
        self,
        *,
        task_name: str | None = None,
        scenario_id: str | None = None,
        variation_seed: int | None = None,
    ) -> Observation:
        payload: dict[str, Any] = {}
        if task_name is not None:
            payload["task_name"] = task_name
        if scenario_id is not None:
            payload["scenario_id"] = scenario_id
        if variation_seed is not None:
            payload["variation_seed"] = variation_seed
        response = self._client.post("/reset", headers=self._headers, json=payload)
        response.raise_for_status()
        return Observation.model_validate(response.json())

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        payload = {"action": action.model_dump(mode="json")}
        response = self._client.post("/step", headers=self._headers, json=payload)
        response.raise_for_status()
        body = response.json()
        observation = Observation.model_validate(body.get("observation", {}))
        reward = float(body.get("reward", 0.0))
        done = bool(body.get("done", False))
        info = body.get("info", {})
        if not isinstance(info, dict):
            info = {}
        return observation, reward, done, info


def _run_scenario(
    env: HttpEnvironmentClient,
    agent: OpenAIEnvironmentAgent,
    *,
    scenario_id: str,
    variation_seed: int,
    deadline: float,
    step_counter: int,
) -> tuple[float, bool, list[float], int]:

    if time.monotonic() >= deadline:
        _log_step(step=step_counter, action="timeout", reward=0.0, done=True, error="runtime_limit_exceeded")
        return 0.0, True, [], step_counter + 1

    try:
        observation = env.reset(scenario_id=scenario_id, variation_seed=variation_seed)
    except Exception as exc:
        _log_step(step=step_counter, action="reset", reward=0.0, done=True, error=str(exc))
        print(f"inference warning: reset_failed scenario={scenario_id} error={exc}", file=sys.stderr)
        return 0.0, False, [], step_counter + 1

    final_score = 0.0
    rewards: list[float] = []
    step_limit = min(MAX_STEPS, max(1, observation.max_steps + 1))

    for _ in range(step_limit):
        if time.monotonic() >= deadline:
            _log_step(step=step_counter, action="timeout", reward=0.0, done=True, error="runtime_limit_exceeded")
            return max(0.0, min(1.0, round(final_score, 4))), True, rewards, step_counter + 1

        action_error: str | None = None
        try:
            action = agent.next_action(observation)
        except Exception as exc:
            action_error = str(exc)
            action = _safe_default_action(observation)

        try:
            observation, reward, done, info = env.step(action)
        except Exception as exc:
            _log_step(step=step_counter, action=action.action_type, reward=0.0, done=True, error=str(exc))
            return max(0.0, min(1.0, round(final_score, 4))), False, rewards, step_counter + 1

        rewards.append(float(reward))
        _log_step(
            step=step_counter,
            action=action.action_type,
            reward=float(reward),
            done=bool(done),
            error=action_error,
        )
        step_counter += 1

        raw_score = info.get("final_score")
        if isinstance(raw_score, int | float):
            final_score = float(raw_score)
        if done:
            break

    return max(0.0, min(1.0, round(final_score, 4))), False, rewards, step_counter


def run(seed: int = 42, task: str = TASK_NAME, max_scenarios: int | None = None) -> dict[str, dict[str, float]]:
    model_name = os.environ.get("MODEL_NAME", "openai/gpt-4.1-mini")
    selected_tasks = [task] if task in {"easy", "medium", "hard"} else ["easy", "medium", "hard"]
    try:
        agent = OpenAIEnvironmentAgent()
    except Exception as exc:
        print(f"inference warning: {exc}", file=sys.stderr)
        for task_name in selected_tasks:
            _log_start(task=task_name, env=BENCHMARK, model=model_name)
            _log_end(task=task_name, success=False, steps=0, score=0.0, rewards=[])
        return _empty_summary()

    base_url = os.environ.get("ENV_URL") or os.environ.get("ENV_BASE_URL") or DEFAULT_ENV_URL
    session_id = f"inference-{seed}-{uuid.uuid4().hex[:8]}"
    env = HttpEnvironmentClient(base_url=base_url, session_id=session_id)
    deadline = time.monotonic() + MAX_RUNTIME_SECONDS
    summary: dict[str, dict[str, float]] = _empty_summary()
    completed_tasks: set[str] = set()
    started_tasks: set[str] = set()

    try:
        task_map = env.tasks()
        if not task_map:
            print("inference warning: /tasks returned no scenarios.", file=sys.stderr)
            for task_name in selected_tasks:
                _log_start(task=task_name, env=BENCHMARK, model=model_name)
                _log_end(task=task_name, success=False, steps=0, score=0.0, rewards=[])
            return summary
        for task_name in selected_tasks:
            _log_start(task=task_name, env=BENCHMARK, model=model_name)
            started_tasks.add(task_name)
            scores: list[float] = []
            task_rewards: list[float] = []
            task_step_counter = 1
            scenario_ids = list(task_map.get(task_name, []))
            if max_scenarios is not None and max_scenarios > 0:
                scenario_ids = scenario_ids[:max_scenarios]
            for scenario_id in scenario_ids:
                score, timed_out, rewards, task_step_counter = _run_scenario(
                    env,
                    agent,
                    scenario_id=scenario_id,
                    variation_seed=seed,
                    deadline=deadline,
                    step_counter=task_step_counter,
                )
                scores.append(score)
                task_rewards.extend(rewards)
                if timed_out:
                    break
            task_score = 0.0
            if scores:
                task_score = round(mean(scores), 4)
                summary[task_name] = {
                    "mean": task_score,
                    "min": round(min(scores), 4),
                    "max": round(max(scores), 4),
                }
            _log_end(
                task=task_name,
                success=task_score >= SUCCESS_SCORE_THRESHOLD,
                steps=max(0, task_step_counter - 1),
                score=task_score,
                rewards=[round(value, 4) for value in task_rewards],
            )
            completed_tasks.add(task_name)
            if time.monotonic() >= deadline:
                break
    except Exception as exc:
        print(f"inference warning: environment_http_error error={exc}", file=sys.stderr)
        for task_name in selected_tasks:
            if task_name in completed_tasks:
                continue
            if task_name not in started_tasks:
                _log_start(task=task_name, env=BENCHMARK, model=model_name)
            _log_end(task=task_name, success=False, steps=0, score=0.0, rewards=[])
    finally:
        env.close()
    return summary


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Run the OpenAI-backed agent via HTTP against the environment API.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default=TASK_NAME)
    parser.add_argument("--max-scenarios", type=int, default=None, help="Optional cap per task difficulty.")
    args = parser.parse_args()
    try:
        summary = run(seed=args.seed, task=args.task, max_scenarios=args.max_scenarios)
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"inference warning: {exc}", file=sys.stderr)
        selected_tasks = [args.task] if args.task in {"easy", "medium", "hard"} else ["easy", "medium", "hard"]
        for task_name in selected_tasks:
            _log_start(task=task_name, env=BENCHMARK, model=os.environ.get("MODEL_NAME", "openai/gpt-4.1-mini"))
            _log_end(task=task_name, success=False, steps=0, score=0.0, rewards=[])
        summary = _empty_summary()
    Path("inference_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
