from __future__ import annotations

import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import gradio as gr
import uvicorn

from business_policy_env.baseline import RuleBasedAgent
from business_policy_env.environment import BusinessPolicyComplianceEnv
from business_policy_env.models import Action
from business_policy_env.server import app as fastapi_app
from business_policy_env.tasks import scenario_registry


def start_api() -> None:
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860, log_level="error")


threading.Thread(target=start_api, daemon=True).start()

env = BusinessPolicyComplianceEnv()
current_obs = None
ROOT_DIR = Path(__file__).resolve().parent
DEMO_VIDEO_PATH = ROOT_DIR / "assets" / "multimodal_demo.mp4"


# ── helpers ──────────────────────────────────────────────────────────────────

def get_scenario_choices() -> list[tuple[str, str]]:
    registry = scenario_registry()
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    difficulty_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
    entries = sorted(registry.values(), key=lambda item: (difficulty_order[item.difficulty], item.scenario_id))
    return [(f"{difficulty_icon[s.difficulty]} {s.difficulty.upper()} · {s.title}", s.scenario_id) for s in entries]


def format_observation(obs) -> str:
    lines = [
        f"**Scenario:** {obs.scenario_id} [{obs.difficulty}]",
        f"**Policy version:** {obs.policy_version}",
        (
            f"**Policy shift:** pending at step {obs.policy_shift_at_step} → {obs.policy_shift_to}"
            if obs.policy_shift_pending
            else "**Policy shift:** none pending"
        ),
        f"**Phase:** {obs.episode_phase}",
        f"**Steps:** {obs.steps_taken}/{obs.max_steps}",
        f"**Issue age:** {obs.issue_age_hours:.1f}h",
        f"**Sender tier:** {obs.sender_tier}",
        f"**Refund amount:** {'$' + str(obs.refund_amount) if obs.refund_amount is not None else 'N/A'}",
        f"**Account flags:** {', '.join(obs.account_flags) if obs.account_flags else 'none'}",
        f"**Attachment present:** {'yes' if obs.attachment_present else 'no'}",
        "",
        "**Latest email:**",
        f"> Subject: {obs.current_email.subject}",
        f"> {obs.current_email.body}",
        "",
        "**Policy rules:**",
    ]
    for rule in obs.policy_rules:
        lines.append(f"- {rule}")
    if obs.attachment_present:
        lines.append("")
        lines.append("**Attachment summary (precomputed):**")
        lines.append(f"- {obs.attachment_summary or 'No summary available.'}")
        lines.append(
            f"- Signals: {', '.join(obs.attachment_signals) if obs.attachment_signals else 'none'}"
        )
    if obs.agent_notes:
        lines.append("")
        lines.append("**Agent notes:**")
        for note in obs.agent_notes[-5:]:
            lines.append(f"- {note}")
    if obs.action_history:
        lines.append("")
        lines.append("**Action history:**")
        records = list(obs.action_history)
        if len(records) == 1:
            record = records[0]
            lines.append(f"- Step {record.step_index}: `{record.action.action_type}`")
        else:
            # Compress repeated runs for readability, but always show the latest step explicitly.
            historical = records[:-1]
            last_record = records[-1]
            idx = 0
            while idx < len(historical):
                run_start = idx
                run_action_type = historical[idx].action.action_type
                while (
                    idx + 1 < len(historical)
                    and historical[idx + 1].action.action_type == run_action_type
                ):
                    idx += 1
                run_end = idx
                run_count = run_end - run_start + 1
                start_step = historical[run_start].step_index
                end_step = historical[run_end].step_index
                if run_count == 1:
                    lines.append(f"- Step {start_step}: `{run_action_type}`")
                else:
                    lines.append(
                        f"- Step {start_step}-{end_step}: `{run_action_type}` x {run_count} (repeated)"
                    )
                idx += 1

            lines.append(
                f"- Step {last_record.step_index}: `{last_record.action.action_type}` (latest)"
            )
    return "\n".join(lines)


def _attachment_path_from_state(state_payload: dict[str, Any]) -> str | None:
    ground_truth = state_payload.get("ground_truth")
    if not isinstance(ground_truth, dict):
        return None
    snapshot = ground_truth.get("snapshot")
    if not isinstance(snapshot, dict):
        return None
    attachment_path = snapshot.get("attachment_path")
    if not attachment_path:
        return None
    resolved = (ROOT_DIR / str(attachment_path)).resolve()
    if not resolved.exists():
        return None
    return str(resolved)


def _signals_badges_html(signals: list[str]) -> str:
    if not signals:
        return "<span style='color:#64748b;'>No attachment signals.</span>"
    badges: list[str] = []
    for idx, signal in enumerate(signals):
        lowered = signal.lower()
        if any(key in lowered for key in ["fraud", "mismatch", "tampering", "edited", "chargeback", "risk", "ato"]):
            bg, fg, border = "#fee2e2", "#991b1b", "#fecaca"
        elif any(key in lowered for key in ["duplicate", "high_value", "amount", "billing", "checkout", "blocked"]):
            bg, fg, border = "#ffedd5", "#9a3412", "#fed7aa"
        elif any(key in lowered for key in ["error", "mobile", "ui", "code"]):
            bg, fg, border = "#dbeafe", "#1e3a8a", "#bfdbfe"
        else:
            # Deterministic fallback split for visual variety without low-contrast text.
            if idx % 2 == 0:
                bg, fg, border = "#dcfce7", "#166534", "#bbf7d0"
            else:
                bg, fg, border = "#e0f2fe", "#075985", "#bae6fd"
        badges.append(
            "<span style='display:inline-block;margin:2px 6px 2px 0;padding:4px 10px;"
            f"border-radius:999px;background:{bg};color:{fg};border:1px solid {border};"
            "font-weight:700;font-size:12px;'>"
            f"{signal}"
            "</span>"
        )
    return "".join(badges)


def _attachment_panel_values(obs, state_payload: dict[str, Any]) -> tuple[str | None, str, str]:
    if not obs.attachment_present:
        return None, "No attachment in this scenario.", _signals_badges_html([])

    image_path = _attachment_path_from_state(state_payload)
    summary = obs.attachment_summary or "No attachment summary available."
    if image_path is None:
        summary = f"{summary}\n\n(Attachment file is referenced but not present on disk.)"
    return image_path, summary, _signals_badges_html(list(obs.attachment_signals))


def _same_path(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return a == b


def _manual_image_update_for_step(scenario_image: str | None, current_image: str | None) -> Any:
    # If user has uploaded a custom image, keep it stable across steps.
    if current_image and (scenario_image is None or not _same_path(current_image, scenario_image)):
        return gr.update()
    if scenario_image is not None:
        return scenario_image
    return gr.update()


def _demo_attachment_placeholder_html(attachment_present: bool, has_attachment_file: bool) -> str:
    if attachment_present:
        note = "Multimodal scenario loaded. Here you can place your image preview."
        status = "Attachment expected"
        accent = "#0369a1"
    else:
        note = "No attachment in this scenario. Here you can place your image preview."
        status = "No attachment"
        accent = "#64748b"
    if attachment_present and not has_attachment_file:
        note += " (Attachment file path exists in metadata, but image file is missing on disk.)"
    return (
        "<div style='height:260px;border:2px dashed #cbd5e1;border-radius:12px;"
        "display:flex;align-items:center;justify-content:center;padding:18px;"
        "text-align:center;background:#f8fafc;'>"
        "<div>"
        f"<div style='font-size:13px;font-weight:700;color:{accent};margin-bottom:8px;'>{status}</div>"
        "<div style='font-size:14px;color:#334155;'>"
        f"{note}"
        "</div>"
        "</div>"
        "</div>"
    )


def _scoreboard_from_info(obs, info: dict[str, Any], *, done: bool) -> str:
    rb = info.get("reward_breakdown", {}) if isinstance(info, dict) else {}
    scenario = scenario_registry().get(obs.scenario_id)
    cost_budget = float(rb.get("cost_budget", scenario.cost_budget if scenario else 1.0))
    cost_spent = float(rb.get("cost_spent", 0.0))
    final_score = float(info.get("final_score") or 0.0) if isinstance(info, dict) else 0.0
    return _build_scoreboard(
        step=obs.steps_taken,
        max_steps=obs.max_steps,
        done=done,
        final_score=final_score,
        components=dict(info.get("component_scores", {})) if isinstance(info, dict) else {},
        cost_spent=cost_spent,
        cost_budget=cost_budget,
        policy_version=obs.policy_version,
        failure_modes=list(info.get("failure_modes", [])) if isinstance(info, dict) else [],
        evaluation_metrics=dict(info.get("evaluation_metrics", {})) if isinstance(info, dict) else {},
    )


def _action_label(action: Action) -> str:
    parts = [f"`{action.action_type}`"]
    if action.category:
        parts.append(f"→ **{action.category}**")
    if action.priority:
        parts.append(f"→ **{action.priority}**")
    if action.escalation_reason:
        parts.append(f"— _{action.escalation_reason}_")
    if action.fraud_reason:
        parts.append(f"— _{action.fraud_reason}_")
    if action.clarifying_question:
        parts.append(f"— _{action.clarifying_question}_")
    if action.response_text:
        preview = action.response_text[:80] + ("…" if len(action.response_text) > 80 else "")
        parts.append(f'— "{preview}"')
    if action.snooze_hours:
        parts.append(f"→ {action.snooze_hours}h")
    return " ".join(parts)


# ── Manual tab ────────────────────────────────────────────────────────────────

def reset_episode(scenario_id: str) -> tuple[str, str, str, dict[str, Any], str | None, str, str]:
    global current_obs
    current_obs = env.reset(scenario_id=scenario_id)
    state_payload = env.state(include_ground_truth=True)
    attachment_image, attachment_summary, attachment_signals = _attachment_panel_values(current_obs, state_payload)
    scoreboard_html = _scoreboard_from_info(current_obs, {}, done=False)
    return (
        format_observation(current_obs),
        scoreboard_html,
        "Episode reset. Ready for actions.",
        gr.update(interactive=True),
        attachment_image,
        attachment_summary,
        attachment_signals,
    )


def take_action(
    action_type: str,
    category: str,
    priority: str,
    response_text: str,
    escalation_reason: str,
    clarifying_question: str,
    fraud_reason: str,
    snooze_hours: int,
    specialist_question: str,
    reasoning: str,
    current_attachment_image: str | None,
) -> tuple[str, str, str, Any, str, str]:
    # Returns observation markdown, component scores, status text, image path, summary text, and signal badges.
    # Keeping return payload deterministic for easier judge walkthroughs.
    global current_obs
    if current_obs is None:
        return (
            "Reset the environment first.",
            _build_scoreboard(
                step=0,
                max_steps=1,
                done=False,
                final_score=0.0,
                components={},
                cost_spent=0.0,
                cost_budget=1.0,
                policy_version="v1",
                failure_modes=[],
                evaluation_metrics={},
            ),
            "No active episode.",
            gr.update(),
            "No attachment in this scenario.",
            _signals_badges_html([]),
        )
    try:
        action = Action(
            action_type=action_type,
            reasoning=reasoning or "Manual action from Gradio UI.",
            category=category or None,
            priority=priority or None,
            response_text=response_text or None,
            escalation_reason=escalation_reason or None,
            clarifying_question=clarifying_question or None,
            fraud_reason=fraud_reason or None,
            snooze_hours=snooze_hours or None,
            specialist_question=specialist_question or None,
        )
    except Exception as exc:
        state_payload = env.state(include_ground_truth=True)
        attachment_image, attachment_summary, attachment_signals = _attachment_panel_values(current_obs, state_payload)
        return (
            format_observation(current_obs),
            _scoreboard_from_info(current_obs, {}, done=False),
            f"Invalid action: {exc}",
            _manual_image_update_for_step(attachment_image, current_attachment_image),
            attachment_summary,
            attachment_signals,
        )

    current_obs, reward, done, info = env.step(action)
    status = f"Reward: {reward:.4f} | Done: {done}"
    if info.get("policy_violations"):
        status += f" | ⚠️ Policy violations: {', '.join(info['policy_violations'])}"
    if info.get("policy_event"):
        status += f" | 🔄 {info['policy_event']}"
    rb = info.get("reward_breakdown", {})
    if "cost_spent" in rb and "cost_budget" in rb:
        status += f" | Cost: {rb['cost_spent']}/{rb['cost_budget']}"
    if done:
        status += f" | 🏁 Final score: {info.get('final_score', 0):.4f}"
    state_payload = env.state(include_ground_truth=True)
    attachment_image, attachment_summary, attachment_signals = _attachment_panel_values(current_obs, state_payload)
    return (
        format_observation(current_obs),
        _scoreboard_from_info(current_obs, info, done=done),
        status,
        _manual_image_update_for_step(attachment_image, current_attachment_image),
        attachment_summary,
        attachment_signals,
    )


# ── Demo tab ──────────────────────────────────────────────────────────────────

_DEMO_SCENARIOS: list[tuple[str, str]] = [
    ("🔴 HARD · Hidden fraud with delayed-detection risk", "hard_hidden_fraud_delayed_detection"),
    ("🔴 HARD · Invoice screenshot fraud (multimodal)", "hard_invoice_screenshot_fraud"),
    ("🔴 HARD · Duplicate charge statement (multimodal)", "hard_duplicate_charge_statement"),
    ("🔴 HARD · Forged invoice policy-gaming (multimodal)", "hard_forged_invoice_policy_gaming"),
    ("🔴 HARD · FinServ hidden velocity abuse loop", "hard_finserv_hidden_velocity_refund_loop"),
    ("🟡 MEDIUM · Adversarial duplicate-refund claim", "medium_adversarial_refund_already_processed"),
    ("🟡 MEDIUM · Ambiguous screenshot ticket", "medium_ambiguous_screenshot"),
    ("🟡 MEDIUM · HR payroll duplicate-adjustment conflict", "medium_hr_payroll_duplicate_adjustment_claim"),
    ("🟡 MEDIUM · FinServ KYC refund-keyword trap", "medium_finserv_kyc_keyword_trap"),
    ("🟢 EASY · VIP refund over $500 threshold", "easy_vip_refund"),
    ("🟢 EASY · SLA breach — aged billing ticket", "easy_sla_breach"),
    ("🟢 EASY · Legal threat triggers escalation", "easy_legal_threat"),
    ("🟡 MEDIUM · Ambiguous charge — clarification required", "medium_charge_or_bug"),
    ("🟡 MEDIUM · Policy-gaming disguised as urgency", "medium_policy_gaming_subtle"),
    ("🔴 HARD · VIP refund + legal pressure + SLA breach", "hard_vip_refund_lawyer"),
    ("🔴 HARD · Chargeback fraud detection", "hard_fraud_chargeback"),
    ("🔴 HARD · Sarcastic multilingual leaderboard trap", "hard_sarcastic_multilingual_trap"),
    ("🔴 HARD · FinServ sarcastic refund keyword trap", "hard_finserv_sarcastic_refund_keyword_trap"),
    ("🔴 HARD · HR policy-gaming duplicate-adjustment chain", "hard_hr_policy_gaming_duplicate_adjustment_chain"),
    ("🔴 HARD · Long-horizon escalation chain (18 steps)", "hard_long_horizon_escalation_chain"),
    ("🔴 HARD · Three conflicting policy signals", "hard_three_signal_precedence"),
]

_DIFFICULTY_EMOJI = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}

_VIOLATION_TIPS: dict[str, str] = {
    "Priority must be at least": "💡 Policy requires higher priority for this sender tier or issue age.",
    "escalation before resolution": "💡 High-value refund or legal threat must be escalated first.",
    "flag_fraud before resolution": "💡 Fraud indicators detected — flag fraud before routing.",
    "Category must be": "💡 Suspended accounts must route to billing.",
}


def _violation_hint(violations: list[str]) -> str:
    for v in violations:
        for key, tip in _VIOLATION_TIPS.items():
            if key in v:
                return tip
    return ""


def run_demo(
    scenario_id: str,
    step_delay: float,
    show_reasoning: bool,
) -> Generator[tuple[str, str, str, str, str], None, None]:
    """Stream demo log lines and a live scoreboard as the rule agent plays."""

    demo_env = BusinessPolicyComplianceEnv()
    agent = RuleBasedAgent()
    obs = demo_env.reset(scenario_id=scenario_id)
    initial_state = demo_env.state(include_ground_truth=True)
    attachment_image, attachment_summary, attachment_signals = _attachment_panel_values(obs, initial_state)
    attachment_placeholder = _demo_attachment_placeholder_html(obs.attachment_present, bool(attachment_image))

    scenario = scenario_registry()[scenario_id]
    diff_emoji = _DIFFICULTY_EMOJI.get(scenario.difficulty, "⚪")
    policy_shift_note = (
        f" → shifting to {obs.policy_shift_to} at step {obs.policy_shift_at_step}" if obs.policy_shift_pending else ""
    )

    log_lines: list[str] = [
        f"## {diff_emoji} {scenario.title}",
        f"> **Objective:** {scenario.objective}",
        f"> **Policy:** {obs.policy_version}{policy_shift_note}",
        f"> **Max steps:** {obs.max_steps} | **Cost budget:** ${scenario.cost_budget:.2f}",
        "",
        f"**📧 Ticket ({obs.sender_tier.upper()} | age {obs.issue_age_hours:.1f}h)**",
        f"> *{obs.current_email.subject}*",
        f"> {obs.current_email.body}",
        "",
        "---",
        "### 🤖 Agent decisions",
        "",
    ]

    scoreboard = ""
    yield "\n".join(log_lines), scoreboard, attachment_placeholder, attachment_summary, attachment_signals

    done = False
    step = 0
    final_score = 0.0
    component_scores: dict[str, float] = {}
    evaluation_metric_scores: dict[str, float] = {}
    previous_action_type: str | None = None
    previous_draft_response: str | None = None
    repeated_action_streak = 0
    budget_warning_logged = False

    while not done:
        action = agent.next_action(obs)
        obs, reward, done, info = demo_env.step(action)
        step += 1
        if action.action_type == previous_action_type:
            repeated_action_streak += 1
        else:
            repeated_action_streak = 1
        previous_action_type = action.action_type

        policy_event = info.get("policy_event") or ""
        violations = info.get("policy_violations", [])
        components = info.get("component_scores", {})
        metrics = info.get("evaluation_metrics", {})
        rb = info.get("reward_breakdown", {})
        failure_modes = info.get("failure_modes", [])

        # Build step block
        reward_icon = "✅" if reward >= 0 else "❌"
        repeated_draft = (
            action.action_type == "draft_response"
            and bool(action.response_text)
            and action.response_text == previous_draft_response
        )
        if repeated_draft:
            headline = (
                f"**Step {step}** — `draft_response` ♻️ same fallback reply as previous step "
                f"{reward_icon} `{reward:+.4f}`"
            )
        else:
            headline = f"**Step {step}** — {_action_label(action)} {reward_icon} `{reward:+.4f}`"
        step_lines: list[str] = [headline]
        if show_reasoning and action.reasoning:
            step_lines.append(f"  *Reasoning: {action.reasoning}*")
        if repeated_draft:
            step_lines.append(
                "  🧩 Repetition here is a known rule-agent limitation; "
                "stronger LLM agents should vary response quality."
            )
        if violations:
            step_lines.append(f"  ⚠️ **Policy violation:** {violations[0]}")
            hint = _violation_hint(violations)
            if hint:
                step_lines.append(f"  {hint}")
        if policy_event:
            step_lines.append(f"  🔄 **{policy_event}**")
        if "snooze_sla_penalty" in rb and rb["snooze_sla_penalty"] < 0:
            step_lines.append("  ⏰ **SLA crossed during snooze!**")
        if obs.specialist_feedback and obs.specialist_feedback not in [
            line for line in log_lines if "Specialist" in line
        ]:
            step_lines.append(f"  🧑‍💼 **Specialist:** _{obs.specialist_feedback}_")
        if repeated_action_streak == 2:
            step_lines.append(
                "  ⚠️ Rule agent exhausted its strategy — repeating fallback response. "
                "An LLM agent would vary response quality here."
            )
        if (
            not budget_warning_logged
            and rb.get("cost_spent", 0.0) > rb.get("cost_budget", 999.0)
        ):
            step_lines.append("  💸 Budget exceeded — efficiency penalty will apply at episode end.")
            budget_warning_logged = True

        log_lines.extend(step_lines)
        log_lines.append("")
        if action.action_type == "draft_response" and action.response_text:
            previous_draft_response = action.response_text

        # Update scoreboard
        if components:
            component_scores = components
        if metrics:
            evaluation_metric_scores = metrics
        if done:
            final_score = info.get("final_score", 0.0) or 0.0

        scoreboard = _build_scoreboard(
            step=step,
            max_steps=obs.max_steps,
            done=done,
            final_score=final_score,
            components=component_scores,
            cost_spent=rb.get("cost_spent", 0.0),
            cost_budget=rb.get("cost_budget", scenario.cost_budget),
            policy_version=obs.policy_version,
            failure_modes=failure_modes,
            evaluation_metrics=evaluation_metric_scores,
        )
        state_payload = demo_env.state(include_ground_truth=True)
        attachment_image, attachment_summary, attachment_signals = _attachment_panel_values(obs, state_payload)
        attachment_placeholder = _demo_attachment_placeholder_html(obs.attachment_present, bool(attachment_image))
        yield "\n".join(log_lines), scoreboard, attachment_placeholder, attachment_summary, attachment_signals
        if not done:
            time.sleep(step_delay)

    # Final summary
    log_lines += [
        "---",
        "### 🏁 Episode complete",
        f"**Final score: `{final_score:.4f}`**",
        _score_verdict(final_score, scenario.difficulty),
    ]
    if component_scores:
        log_lines.append("")
        log_lines.append("**Component breakdown:**")
        for k, v in component_scores.items():
            bar = _mini_bar(v)
            log_lines.append(f"- `{k}`: {bar} `{v:.3f}`")
    ideal_summary = _ideal_agent_summary(scenario_id)
    if ideal_summary:
        log_lines.append("")
        log_lines.append("**What an ideal agent would do differently:**")
        for item in ideal_summary:
            log_lines.append(f"- {item}")

    final_state = demo_env.state(include_ground_truth=True)
    attachment_image, attachment_summary, attachment_signals = _attachment_panel_values(obs, final_state)
    attachment_placeholder = _demo_attachment_placeholder_html(obs.attachment_present, bool(attachment_image))
    yield "\n".join(log_lines), scoreboard, attachment_placeholder, attachment_summary, attachment_signals


def _mini_bar(v: float, width: int = 10) -> str:
    clamped = max(0.0, min(1.0, v))
    filled = int(round(clamped * width))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def _score_verdict(score: float, difficulty: str) -> str:
    if difficulty == "easy":
        if score >= 0.85:
            return "🏆 Excellent — rule baseline clears this tier reliably."
        if score >= 0.6:
            return "✅ Solid — basic policy compliance achieved."
        return "⚠️ Below baseline — policy or routing error occurred."
    if difficulty == "medium":
        if score >= 0.65:
            return "🏆 Strong — ambiguity handled correctly."
        if score >= 0.4:
            return "✅ Partial — clarification or response quality lacking."
        return "⚠️ Weak — likely skipped request_info on ambiguous ticket."
    # hard
    if score >= 0.7:
        return "🏆 Impressive — complex thread resolved with policy and history awareness."
    if score >= 0.45:
        return "✅ Moderate — some components missing (specialist, adversarial, history)."
    return "⚠️ Low — hard tier requires specialist coordination and adversarial reasoning."


def _ideal_agent_summary(scenario_id: str) -> list[str]:
    scenario_specific: dict[str, list[str]] = {
        "medium_adversarial_refund_already_processed": [
            "Explicitly cite prior refund evidence and avoid promising a duplicate payout.",
            "Request a transaction reference check while keeping tone firm and customer-safe.",
            "Choose escalation only if evidence is inconsistent after clarification.",
        ],
        "hard_hidden_fraud_delayed_detection": [
            "Flag fraud earlier from subtle intent signals instead of waiting for explicit confirmation.",
            "Escalate with a risk-focused rationale and communicate containment steps.",
            "Provide a concrete timeline and next action ownership in the response.",
        ],
        "hard_long_horizon_escalation_chain": [
            "Use specialist input early and keep updates structured with ownership and cadence.",
            "Avoid repetitive filler responses and progress the case each step.",
            "Balance policy compliance with budget efficiency across the full trajectory.",
        ],
    }
    default_summary = [
        "Ground responses in thread-specific facts instead of generic fallback language.",
        "Avoid repeated identical actions and adapt strategy as new signals appear.",
        "Improve clarity on timeline, ownership, and next verifiable step.",
    ]
    return scenario_specific.get(scenario_id, default_summary)


def _build_scoreboard(
    step: int,
    max_steps: int,
    done: bool,
    final_score: float,
    components: dict[str, float],
    cost_spent: float,
    cost_budget: float,
    policy_version: str,
    failure_modes: list[str],
    evaluation_metrics: dict[str, float],
) -> str:
    def _bar(value: float, *, color: str) -> str:
        clamped = max(0.0, min(1.0, float(value)))
        width = int(round(clamped * 100))
        return (
            "<div style='width:100%;height:10px;background:#334155;border-radius:999px;overflow:hidden;'>"
            f"<div style='width:{width}%;height:10px;background:{color};border-radius:999px;'></div>"
            "</div>"
        )

    metric_order = [
        "policy_score",
        "efficiency",
        "latency",
        "customer_quality",
        "risk_management",
        "adversarial_resilience",
        "memory_score_component",
        "attachment_utilization",
        "multimodal_fraud",
    ]
    metric_labels = {
        "policy_score": "Policy score",
        "efficiency": "Efficiency",
        "latency": "Latency",
        "customer_quality": "Customer quality",
        "risk_management": "Risk management",
        "adversarial_resilience": "Adversarial resilience",
        "memory_score_component": "Memory usage",
        "attachment_utilization": "Attachment utilization",
        "multimodal_fraud": "Multimodal fraud",
    }
    metric_colors = {
        "policy_score": "#0f766e",
        "efficiency": "#1d4ed8",
        "latency": "#0f766e",
        "customer_quality": "#be123c",
        "risk_management": "#b45309",
        "adversarial_resilience": "#7c3aed",
        "memory_score_component": "#334155",
        "attachment_utilization": "#0369a1",
        "multimodal_fraud": "#b91c1c",
    }

    step_ratio = 0.0 if max_steps <= 0 else min(1.0, step / max_steps)
    cost_ratio = 0.0 if cost_budget <= 0 else min(1.0, cost_spent / cost_budget)
    score_label = f"{final_score:.4f}" if done else f"{float(evaluation_metrics.get('score', 0.0)):.4f}"
    score_caption = "Final score" if done else "Current score"

    parts: list[str] = [
        (
            "<div style='border:1px solid #334155;border-radius:14px;padding:14px;background:#1e293b;"
            "color:#e2e8f0 !important;'>"
        ),
        "<div style='font-weight:800;font-size:16px;color:#f1f5f9;margin-bottom:10px;'>📊 Live Scoreboard</div>",
        (
            "<div style='display:flex;align-items:center;justify-content:space-between;"
            "padding:10px 12px;border-radius:10px;background:#0f172a;border:1px solid #1e293b;'>"
            f"<div style='font-size:12px;font-weight:700;color:#e2e8f0 !important;'>{score_caption}</div>"
            f"<div style='font-size:22px;font-weight:800;color:#f8fafc;'>{score_label}</div>"
            "</div>"
        ),
        "<div style='margin-top:12px;'>",
        (
            "<div style='font-size:12px;font-weight:700;color:#f1f5f9 !important;"
            "margin-bottom:4px;'>Steps</div>"
        ),
        _bar(step_ratio, color="#334155"),
        (
            f"<div style='font-size:11px;font-weight:600;color:#e2e8f0 !important;"
            f"margin-top:4px;'>{step} / {max_steps}</div>"
        ),
        "</div>",
        "<div style='margin-top:10px;'>",
        (
            "<div style='font-size:12px;font-weight:700;color:#f1f5f9 !important;"
            "margin-bottom:4px;'>Cost budget usage</div>"
        ),
        _bar(cost_ratio, color="#b45309"),
        (
            f"<div style='font-size:11px;font-weight:600;color:#e2e8f0 !important;margin-top:4px;'>"
            f"${cost_spent:.3f} / ${cost_budget:.2f}</div>"
        ),
        "</div>",
        (
            f"<div style='font-size:12px;font-weight:700;color:#f1f5f9 !important;margin-top:10px;'>"
            f"Policy version: <b style='color:#7dd3fc;'>{policy_version}</b></div>"
        ),
    ]

    if evaluation_metrics:
        parts.append(
            "<div style='margin-top:14px;font-weight:800;font-size:13px;color:#f1f5f9 !important;'>"
            "Evaluation metrics</div>"
        )
        for metric in metric_order:
            value = float(evaluation_metrics.get(metric, 0.0))
            label = metric_labels[metric]
            color = metric_colors[metric]
            parts.extend(
                [
                    "<div style='margin-top:8px;'>",
                    (
                        "<div style='display:flex;justify-content:space-between;font-size:12px;"
                        "font-weight:700;color:#e2e8f0 !important;margin-bottom:4px;'>"
                        f"<span>{label}</span><span>{value:.3f}</span></div>"
                    ),
                    _bar(value, color=color),
                    "</div>",
                ]
            )

        customer_quality = float(evaluation_metrics.get("customer_quality", 0.0))
        if customer_quality <= 0.05:
            parts.append(
                "<div style='margin-top:10px;padding:8px 10px;border-radius:8px;"
                "background:#1c1917;border:1px solid #78350f;font-size:12px;color:#fed7aa;'>"
                "ℹ️ Low customer quality is expected for the rule-based baseline. "
                "Reasoning-capable LLM agents should score materially higher here."
                "</div>"
            )

    if failure_modes:
        chips = " ".join(
            (
                "<span style='display:inline-block;margin:3px 5px 0 0;padding:4px 8px;"
                "border-radius:999px;background:#450a0a;color:#fca5a5;font-size:11px;font-weight:600;'>"
                f"{mode}</span>"
            )
            for mode in failure_modes
        )
        parts.append(
            "<div style='margin-top:12px;font-weight:800;font-size:13px;color:#f1f5f9 !important;'>"
            "Failure modes</div>"
        )
        parts.append(f"<div style='margin-top:2px;'>{chips}</div>")

    if components:
        parts.append(
            "<details style='margin-top:12px;'><summary style='cursor:pointer;color:#94a3b8;"
            "font-weight:700;'>Component details</summary>"
        )
        for key, value in components.items():
            parts.append(
                "<div style='font-size:11px;font-weight:600;color:#94a3b8 !important;margin-top:4px;'>"
                f"{key}: {float(value):.3f}</div>"
            )
        parts.append("</details>")

    parts.append("</div>")
    return "".join(parts)


# ── Gradio layout ─────────────────────────────────────────────────────────────

with gr.Blocks(title="Business Policy Compliance Environment", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "## 🏢 Business Policy Compliance & Customer Resolution Environment\n"
        "_Policy-aware agent evaluation — 54 scenarios · adversarial mechanics · "
        "specialist escalation · multimodal signals_"
    )

    with gr.Tabs():

        # ── Tab 1: Demo ───────────────────────────────────────────────────────
        with gr.Tab("🎬 Demo — Watch the agent"):
            gr.Markdown(
                "Select a scenario and watch the **rule-based agent** solve it step by step. "
                "The agent uses keyword heuristics and policy rules — no LLM. "
                "Hard scenarios expose its limits clearly."
            )

            with gr.Row():
                demo_scenario_dd = gr.Dropdown(
                    choices=_DEMO_SCENARIOS,
                    value="hard_hidden_fraud_delayed_detection",
                    label="Scenario",
                    scale=3,
                )
                step_delay_slider = gr.Slider(
                    minimum=0.2,
                    maximum=3.0,
                    value=0.8,
                    step=0.1,
                    label="Step delay (seconds)",
                    scale=1,
                )

            gr.Markdown("**Quick multimodal jump:**")
            with gr.Row():
                jump_invoice_btn = gr.Button("Invoice screenshot fraud", size="sm")
                jump_statement_btn = gr.Button("Duplicate charge statement", size="sm")
                jump_ambiguous_btn = gr.Button("Ambiguous screenshot ticket", size="sm")

            with gr.Row():
                show_reasoning_cb = gr.Checkbox(
                    value=True,
                    label="Show agent reasoning",
                    scale=1,
                )
                run_demo_btn = gr.Button(
                    "▶ Run demo",
                    variant="primary",
                    scale=2,
                )
                stop_demo_btn = gr.Button(
                    "⏹ Stop",
                    variant="stop",
                    scale=1,
                )

            with gr.Row():
                with gr.Column(scale=2):
                    demo_log = gr.Markdown(
                        label="Episode log",
                        value="_Press **Run demo** to start._",
                    )
                with gr.Column(scale=1):
                    demo_attachment_box = gr.HTML(
                        label="Attachment placeholder",
                        value=_demo_attachment_placeholder_html(False, False),
                    )
                    demo_attachment_summary = gr.Textbox(
                        label="Attachment summary (what the agent sees)",
                        interactive=False,
                        lines=4,
                        value="No attachment in this scenario.",
                    )
                    demo_attachment_signals = gr.HTML(
                        value=_signals_badges_html([]),
                        label="Attachment signals",
                    )
                    demo_scoreboard = gr.HTML(
                        label="Live scoreboard",
                        value=_build_scoreboard(
                            step=0,
                            max_steps=1,
                            done=False,
                            final_score=0.0,
                            components={},
                            cost_spent=0.0,
                            cost_budget=1.0,
                            policy_version="v1",
                            failure_modes=[],
                            evaluation_metrics={},
                        ),
                    )

            run_demo_btn.click(
                fn=run_demo,
                inputs=[demo_scenario_dd, step_delay_slider, show_reasoning_cb],
                outputs=[
                    demo_log,
                    demo_scoreboard,
                    demo_attachment_box,
                    demo_attachment_summary,
                    demo_attachment_signals,
                ],
            )

            jump_invoice_btn.click(fn=lambda: "hard_invoice_screenshot_fraud", outputs=[demo_scenario_dd])
            jump_statement_btn.click(fn=lambda: "hard_duplicate_charge_statement", outputs=[demo_scenario_dd])
            jump_ambiguous_btn.click(fn=lambda: "medium_ambiguous_screenshot", outputs=[demo_scenario_dd])

            if DEMO_VIDEO_PATH.exists():
                gr.Markdown("### Multimodal walkthrough video")
                gr.Video(
                    value=str(DEMO_VIDEO_PATH),
                    label="Recorded demo: attachment-driven fraud handling",
                    interactive=False,
                )
            else:
                gr.Markdown(
                    "_Optional: place `assets/multimodal_demo.mp4` to show a short recorded walkthrough here._"
                )

            gr.Markdown(
                "---\n"
                "**Difficulty guide:**  🟢 Easy — clear policy signals  |  "
                "🟡 Medium — ambiguous, requires clarification  |  "
                "🔴 Hard — multi-turn, adversarial, specialist needed"
            )

        # ── Tab 2: Manual judge UI ────────────────────────────────────────────
        with gr.Tab("🕹️ Manual — Step through yourself"):
            gr.Markdown(
                "_Select a scenario, reset the episode, then take actions one step at a time._"
            )

            with gr.Row():
                scenario_dd = gr.Dropdown(
                    choices=get_scenario_choices(),
                    label="Scenario",
                    value=None,
                )
                reset_btn = gr.Button("Reset episode", variant="primary")

            obs_display = gr.Markdown(label="Observation")
            with gr.Row():
                manual_attachment_image = gr.Image(
                    label="Attachment evidence",
                    type="filepath",
                    interactive=True,
                    sources=["upload", "clipboard"],
                    height=300,
                )
                with gr.Column():
                    manual_attachment_summary = gr.Textbox(
                        label="Attachment summary (what the agent sees)",
                        interactive=False,
                        lines=5,
                        value="No attachment in this scenario.",
                    )
                    manual_attachment_signals = gr.HTML(
                        value=_signals_badges_html([]),
                        label="Attachment signals",
                    )

            with gr.Row():
                action_type = gr.Dropdown(
                    choices=[
                        "categorize",
                        "set_priority",
                        "draft_response",
                        "escalate",
                        "mark_spam",
                        "request_info",
                        "flag_fraud",
                        "snooze",
                        "consult_specialist",
                    ],
                    label="Action type",
                )
                reasoning = gr.Textbox(
                    label="Reasoning (not graded)",
                    placeholder="Why are you taking this action?",
                )

            with gr.Row():
                category = gr.Dropdown(
                    choices=["billing", "technical_support", "returns", "legal", "customer_success", "spam"],
                    label="Category",
                )
                priority = gr.Dropdown(
                    choices=["low", "medium", "high", "urgent"],
                    label="Priority",
                )

            with gr.Row():
                response_text = gr.Textbox(label="Response text (for draft_response)")
                escalation_reason = gr.Textbox(label="Escalation reason (for escalate)")
                clarifying_question = gr.Textbox(label="Clarifying question (for request_info)")

            with gr.Row():
                fraud_reason = gr.Textbox(label="Fraud reason (for flag_fraud)")
                snooze_hours = gr.Number(label="Snooze hours (for snooze)", precision=0, value=0)
                specialist_question = gr.Textbox(label="Specialist question (for consult_specialist)")

            step_btn = gr.Button("Take action", variant="secondary")
            manual_scoreboard = gr.HTML(
                label="Live scoreboard",
                value=_build_scoreboard(
                    step=0,
                    max_steps=1,
                    done=False,
                    final_score=0.0,
                    components={},
                    cost_spent=0.0,
                    cost_budget=1.0,
                    policy_version="v1",
                    failure_modes=[],
                    evaluation_metrics={},
                ),
            )
            status_display = gr.Textbox(label="Status / reward", interactive=False)

            reset_btn.click(
                reset_episode,
                inputs=[scenario_dd],
                outputs=[
                    obs_display,
                    manual_scoreboard,
                    status_display,
                    step_btn,
                    manual_attachment_image,
                    manual_attachment_summary,
                    manual_attachment_signals,
                ],
            )
            step_btn.click(
                take_action,
                inputs=[
                    action_type,
                    category,
                    priority,
                    response_text,
                    escalation_reason,
                    clarifying_question,
                    fraud_reason,
                    snooze_hours,
                    specialist_question,
                    reasoning,
                    manual_attachment_image,
                ],
                outputs=[
                    obs_display,
                    manual_scoreboard,
                    status_display,
                    manual_attachment_image,
                    manual_attachment_summary,
                    manual_attachment_signals,
                ],
            )

        # ── Tab 3: Environment info ───────────────────────────────────────────
        with gr.Tab("📖 Environment info"):
            gr.Markdown("""
## Action space
| Action | Required field | Purpose |
|--------|---------------|---------|
| `categorize` | `category` | Route ticket to the right team |
| `set_priority` | `priority` | Set SLA urgency level |
| `draft_response` | `response_text` | Send reply to customer |
| `escalate` | `escalation_reason` | Escalate to senior team |
| `mark_spam` | — | Discard as spam |
| `request_info` | `clarifying_question` | Ask customer for clarification |
| `flag_fraud` | `fraud_reason` | Flag suspicious activity |
| `snooze` | `snooze_hours` | Defer ticket (SLA risk if >72h total) |
| `consult_specialist` | `specialist_question` | Pull in specialist review |

## Policy sets
**v1** — Refunds >$500 escalate · VIP = high/urgent · Age >72h = urgent · Legal = escalate · Suspended → billing

**v2** (v1 plus) — Premier = same-day · Fraud indicators → `flag_fraud` first

## Reward structure
| Signal | Value |
|--------|-------|
| Valid action | +0.05 |
| Policy violation | −0.20 |
| SLA crossed during snooze | −0.10 |
| Fraud missed at episode end | −0.15 |
| Efficiency bonus (≤ half steps) | +0.10 |
| Cost adjustment | ±0.12 max |
| Redundancy penalty | −0.05 per repeat |

## Baseline scores (rule agent, 54 scenarios)
| Difficulty | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Easy | 0.91 | 0.14 | 0.70 | 1.00 |
| Medium | 0.71 | 0.12 | 0.49 | 0.86 |
| Hard | 0.49 | 0.15 | 0.16 | 0.75 |

## API endpoints
`GET /health` · `GET /tasks` · `POST /reset` · `POST /step` · `GET /state` · `DELETE /session`

Session header: `X-Session-Id` (optional, defaults to `"default"`)
""")


demo.launch(server_port=7861, share=False)
