# Agent Guide

This environment rewards policy-safe, grounded actions over surface-level phrasing.

- Prioritize `issue_age_hours`, `sender_tier`, and `account_flags` before drafting responses.
- Treat ambiguous medium/hard tickets as clarification-first unless evidence is sufficient.
- Avoid claiming actions in `response_text` unless those actions were actually taken.
- Use attachment signals (`attachment_summary`, `attachment_signals`) when present.
- Watch policy changes through active `policy_rules` and `policy_version` (shift timing is hidden).
- Keep action sequences efficient: unnecessary repeats increase cost and reduce reward.
