#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from openai import OpenAI

from business_policy_env.environment import BusinessPolicyComplianceEnv
from business_policy_env.models import Action, Observation
from business_policy_env.tasks import scenarios_for_task

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
        "policy_shift_pending": observation.policy_shift_pending,
        "specialist_feedback": observation.specialist_feedback,
        "attachment_present": observation.attachment_present,
        "attachment_summary": observation.attachment_summary,
        "attachment_signals": observation.attachment_signals,
        "agent_notes": observation.agent_notes,
        "task_objective": observation.task_objective,
        "clarification_received": observation.clarification_received,
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
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["HF_TOKEN"],
        )
        model = os.environ["MODEL_NAME"]
        self._client = client.with_options(timeout=12.0, max_retries=0)
        self._model = model
        self._llm_available = True

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


def _run_scenario(env: BusinessPolicyComplianceEnv, agent: OpenAIEnvironmentAgent, scenario_id: str) -> float:
    try:
        observation = env.reset(scenario_id=scenario_id)
    except Exception:
        return 0.0

    final_score = 0.0
    max_turns = max(1, observation.max_steps + 1)
    for _ in range(max_turns):
        try:
            action = agent.next_action(observation)
        except Exception:
            action = _safe_default_action(observation)

        try:
            observation, _reward, done, info = env.step(action)
        except Exception:
            return final_score

        raw_score = info.get("final_score")
        if isinstance(raw_score, int | float):
            final_score = float(raw_score)
        if done:
            break

    return max(0.0, min(1.0, round(final_score, 4)))


def run(seed: int = 42) -> dict[str, dict[str, float]]:
    env = BusinessPolicyComplianceEnv(variation_seed=seed)
    try:
        agent = OpenAIEnvironmentAgent()
    except Exception as exc:
        print(f"inference warning: {exc}", file=sys.stderr)
        return _empty_summary()

    summary: dict[str, dict[str, float]] = {}
    for task_name in ("easy", "medium", "hard"):
        scores = [
            _run_scenario(env, agent, scenario.scenario_id)
            for scenario in scenarios_for_task(task_name)
        ]
        if not scores:
            summary[task_name] = {"mean": 0.0, "min": 0.0, "max": 0.0}
            continue
        summary[task_name] = {
            "mean": round(mean(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }
    return summary


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Run the OpenAI-backed agent across all environment tasks.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        summary = run(seed=args.seed)
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"inference warning: {exc}", file=sys.stderr)
        summary = _empty_summary()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
