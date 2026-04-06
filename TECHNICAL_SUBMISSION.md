# Tensor-Apex: Business Policy Compliance and Customer Resolution Environment
## Complete Technical Submission — End-to-End Model Explanation

---

## The Gap We Are Filling

Every existing agent evaluation benchmark makes the same implicit assumption: if the agent says the right thing, it did the right thing. This assumption fails immediately in production support operations.

A real support agent can write a polished, empathetic reply while simultaneously: routing the ticket to the wrong department, missing a fraud signal buried in account history, failing to escalate a $700 refund that policy requires to go to senior review, and claiming "we have already escalated this" when no escalation action was ever taken. Under existing benchmarks, this agent scores well. Under Tensor-Apex, it scores poorly — because Tensor-Apex evaluates what the agent **did**, not just what it **said**.

Current benchmarks fail along three specific axes:

**No multi-step reasoning requirement.** Most benchmarks are single-turn: one input, one output, one score. Real support operations are multi-turn workflows where early decisions constrain later options, where policy may shift mid-thread, and where the order of actions matters as much as the content of any individual action. A correct escalation that happens after an illegal draft response still violates policy sequence requirements.

**No adversarial pressure.** Benchmark inputs are honest. Real customers are not always honest. Users claim refunds that were already processed, fabricate urgency to bypass fraud review, use emotionally charged or multilingual sarcasm to obscure their actual intent, and embed "refund" keywords in messages that are actually about technical crashes. Benchmarks that do not model this produce agents that are trivially manipulable in deployment.

**No action grounding verification.** Benchmarks reward language quality independently of operational correctness. An agent that fabricates having taken an action — "we have already processed your refund and escalated to our specialist team" — when neither action was executed in the environment gets full response quality credit. Tensor-Apex explicitly detects and penalizes this pattern.

Tensor-Apex was built to fix all three failures simultaneously across enterprise customer support plus adjacent compliance-heavy verticals (HR operations and financial-services operations), where policy complexity is real and behavioral evaluation requirements are not yet served by existing public benchmarks.
In benchmarking terms, the target is the "enterprise operations" gap: not toy chat tasks, but policy-constrained workflows where mistakes have operational and compliance cost.

---

## What the Environment Models

Think of Tensor-Apex as a compact simulation of a real support operations desk. The agent plays the role of a frontline support decision-maker. It receives customer email threads with sender metadata, account signals, refund context, issue age, and active policy rules. It must execute a sequence of actions — routing, prioritization, escalation, fraud flagging, clarification requests, specialist consultation, response drafting — while operating under four simultaneous constraints:

**Policy constraints.** Two policy versions are active: v1 and v2. v1 requires escalation for refunds over $500, urgent priority for issues older than 72 hours, immediate escalation for legal threats, and billing routing for suspended accounts. v2 adds same-day response requirements for premier accounts and mandatory fraud flagging before any resolution action when fraud indicators are present. In some scenarios the policy shifts from v1 to v2 at a specific step mid-episode, requiring the agent to detect and adapt.

**Partial observability.** In adversarial scenarios, fraud risk signals (`fraud_risk`, `ato_watch`, `chargeback_risk`) are hidden from the agent's observation surface and stored in internal-only fields. The agent sees the customer's email but not the account risk flags. It must infer risk from linguistic signals — phrases like "bypass review", "multiple new cards", "urgent test transactions" — and act accordingly before it has confirmation.

**Operational efficiency.** Every action has a cost: `draft_response` costs 0.08, `escalate` 0.06, `consult_specialist` 0.05, `categorize` 0.03. Each scenario has a budget (0.20 for easy, 0.24 for medium, 0.25 for hard). Spending the budget on five identical draft responses to fill remaining steps is penalized. Solving the ticket correctly in fewer steps is rewarded.

**Communication grounding.** Every response the agent writes is checked not only for keyword coverage but for whether it references the specific facts of this ticket, maintains a coherent structure with ownership and timeline commitments, and remains consistent with the actions actually taken. An agent cannot claim it escalated if it did not escalate.

---

## Model Performance — What the Numbers Show

The rule-based baseline provides the floor. It uses keyword pattern matching and hardcoded policy rules — no language model, no reasoning.

| Difficulty | Rule Baseline | Expected LLM Range | What this means |
|---|---|---|---|
| Easy (10 scenarios) | 0.91 ± 0.14 | 0.90–0.98 | Basic policy following; almost saturated |
| Medium (16 scenarios) | 0.71 ± 0.12 | 0.75–0.88 | Ambiguity handling + response quality; meaningful gap |
| Hard (28 scenarios) | 0.49 ± 0.15 | 0.60–0.80 | Adversarial reasoning + specialist coordination; clear separation |

The hard tier is where the benchmark discriminates. A rule-based system scores lower because it cannot: infer hidden fraud from text alone, detect refund traps from conversation history, produce thread-grounded responses, coordinate specialist escalation correctly, or handle sarcastic multilingual mixed-intent. An LLM agent with genuine multi-step reasoning capability should score materially higher.

---

## Core Design — Four Evaluation Signals

To make the grader comprehensible at a glance, everything reduces to four core signals. The 10-component hard grader is internally decomposed from these four principles:

**1. Correctness** — Did the agent take the right actions in the right sequence? Category, priority, escalation, fraud flag, specialist consultation — all must match policy requirements and scenario ground truth.

**2. Policy Compliance** — Did the agent follow active rules throughout the episode, including after mid-episode policy drift? Policy violations are penalized immediately (-0.20 per step) and capped at episode end (compliance score cannot exceed 0.70 if any violation occurred).

**3. Response Quality** — Is the response grounded in the specific ticket, structured with ownership and a forward plan, consistent with what the agent actually did, and free of keyword stuffing? This is evaluated as a hybrid of four sub-signals.

**4. Adversarial Robustness** — Did the agent resist manipulation? Was it fooled into processing a duplicate refund? Did it correctly identify hidden fraud? Did it avoid routing to billing when the customer said "refund" but the actual issue was a technical crash?

Every grader component maps to one of these four signals. Judges can interpret any score by asking: which of the four did the agent fail on?

---

## Architecture — How Every Component Works

### Scenario Engine (`data_generation.py`)

`ScenarioTemplate` drives realism with:
- `hide_fraud_signals`
- `adversarial_pattern`
- `delayed_fraud_step_threshold`
- `min_steps_before_completion`
- `style_noise` + `emotional_tone`
- `specialist_decision`
- `thread_directions`

`ScenarioFactory` uses deterministic seed-based generation, timestamp construction, stylization, hidden-flag handling, and ground-truth assembly.
The adversarial pattern taxonomy (`refund_already_processed`, `policy_gaming_refund_pressure`, `keyword_refund_trap`, `sarcastic_mixed_intent`, `delayed_escalation_chain`) is grounded in publicly documented customer-support and trust/safety failure modes, then encoded deterministically for reproducible scoring.

### Environment Dynamics (`environment.py`)

Key mechanics:
- `_materialize_variant`: anti-memorization jitter with threshold-preserving transforms
- `_adaptive_task_name`: rolling curriculum
- `_action_cost`: explicit operational cost model
- `_maybe_apply_policy_shift`: deterministic mid-episode policy drift
- `_completion_reached`: required actions + min steps + score threshold
- `agent_notes` accumulator: each accepted action appends a compact reasoning note for memory-aware diagnostics
- `_reasoning_depth`: deterministic shallow/moderate/deep label from cross-reference terms + action diversity
- phase machine: `initial → awaiting_clarification → post_clarification → resolving → complete`

### Policy Engine (`policies.py`)

- `compute_policy_expectations`
- `check_policy_violations` (immediate step penalties)
- `policies_satisfied` (terminal compliance)

### Grading Architecture (`tasks.py`)

Anti-gaming text stack:
- `_thread_focus_terms`
- `_thread_grounding_score`
- `_response_structure_score`
- `_response_action_consistency_score`
- `_hybrid_response_score`
- `_semantic_variants` synonym expansion to reduce paraphrase brittleness in keyword-linked components (for example refund/reimburse/credit families)

Fraud logic:
- `_fraud_score` gives proportional credit for late fraud flagging: `threshold / detection_step`

Adversarial pattern scoring:
- `refund_already_processed`
- `policy_gaming_refund_pressure`
- `keyword_refund_trap`
- `sarcastic_mixed_intent`
- `hidden_fraud_signal`
- `delayed_escalation_chain`

Hard grader weights (sum = 1.0):
- `adversarial_resilience`: 0.18
- `history_acknowledgment`: 0.14
- `specialist_coordination`: 0.15
- `response_completeness`: 0.10
- `fraud_handling`: 0.10
- `escalation_accuracy`: 0.10
- `clarification_strategy`: 0.06
- `policy_compliance`: 0.06
- `customer_quality`: 0.07
- `temporal_reasoning`: 0.04

Calibration evidence is enforced in tests, including:
- `test_hard_response_prefers_thread_grounded_reply`
- `test_keyword_stuffing_is_penalized_vs_balanced_response`
- `test_hard_response_action_consistency_detects_false_claims`
- `test_hard_response_action_consistency_detects_category_claim_mismatch`

### Reward Shaping (`rewards.py`)

Immediate shaping:
- valid action signal
- partial score
- policy penalty
- snooze SLA penalty
- running cost penalty

Terminal shaping:
- efficiency bonus
- redundancy penalty
- fraud missed penalty
- delayed fraud penalty (proportional by lateness)
- early misroute penalty
- context ignorance penalty when history exists but response ignores it
- cross-partition bonus when response grounds both thread-history and attachment signals
- cost adjustment
- policy history penalty

---

## Infrastructure and Reproducibility

- Session isolation via `X-Session-Id`
- Episode-scoped SQLite action logging extracted to `db.py` (`ActionLogger`) with `episode_id` surfaced in step info
- Deterministic baseline CLI with seed control: `baseline.py --seed 42` (default)
- OpenEnv contract validation script
- Docker smoke script for runtime proof
- 40 targeted tests including async concurrency, proportional delayed-fraud reward checks, action-claim consistency, seeded reproducibility, cross-vertical registration, hard-baseline ceiling checks, memory diagnostics, and episode logging invariants

Published deterministic baselines (`--seed 42`):
- Easy: `0.91 ± 0.14`
- Medium: `0.71 ± 0.12`
- Hard: `0.49 ± 0.15`

---

## Episode Lifecycle

1. `POST /reset`
2. repeated `POST /step`
3. per-step policy checks, shifts, rewards, metrics, failure modes
4. terminal grading + final breakdown

---

## Why the Complexity Is Justified

Each major component closes a concrete gaming loophole:
- no thread grounding → policy parroting wins
- no action consistency checks → fabricated action claims win
- no stuffing penalty → repetition gaming wins
- no delayed fraud timing → late fraud flagging scores too high
- no policy-history cap → policy violations can be fully washed out
- no adversarial scoring → manipulation resistance is invisible
- no min-step constraints → hard tasks can be fast-closed
- no early-misroute penalty → wrong first route has little cost

The complexity is purposeful and test-backed.

---

## End-to-End Value Proposition

Tensor-Apex evaluates whether an agent can operate correctly inside a constrained, policy-governed, adversarially pressured support workflow — not merely produce polished support language.

It rewards grounded, policy-compliant, timely, and coordinated behavior.
It penalizes operationally unsafe or manipulative failure modes.
It returns interpretable diagnostics that separate correctness, compliance, communication quality, and adversarial robustness.

That is the deployment-relevant question this benchmark is built to answer.
