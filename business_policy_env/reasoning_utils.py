from __future__ import annotations

from typing import Literal

ReasoningDepth = Literal["shallow", "moderate", "deep"]

_CROSS_REFERENCE_TERMS: tuple[str, ...] = (
    "previous",
    "earlier",
    "thread",
    "history",
    "already",
    "attachment",
    "mentioned",
    "follow-up",
)


def reasoning_depth_label(*, text: str, entry_count: int, unique_action_types: int) -> ReasoningDepth:
    if entry_count < 2:
        return "shallow"

    lowered = text.lower()
    cross_refs = sum(1 for term in _CROSS_REFERENCE_TERMS if term in lowered)

    if cross_refs >= 3 and unique_action_types >= 3:
        return "deep"
    if cross_refs >= 1 or unique_action_types >= 2:
        return "moderate"
    return "shallow"

