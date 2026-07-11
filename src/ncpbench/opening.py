"""External fact extraction for a narrator-generated opening."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Protocol

from ncpbench.models import PendingFactUpdates
from ncpbench.model_output import (
    ModelCallError,
    ModelOutputError,
    call_with_retries,
    parse_json_object,
)
from ncpbench.narrator import NarratorResponse, OpeningRequest


ChatMessage = Mapping[str, str]


class OpeningAuditClient(Protocol):
    def complete(self, messages: Sequence[ChatMessage], *, stage: str) -> str:
        """Return the raw evaluator response for opening fact extraction."""


class OpeningEvaluationError(RuntimeError):
    """Raised when opening fact extraction does not return valid JSON."""


@dataclass(frozen=True)
class OpeningEvaluator:
    """Extract deferred fact updates from text produced by a narrator."""

    auditor: OpeningAuditClient

    def evaluate(
        self, request: OpeningRequest, response: NarratorResponse
    ) -> PendingFactUpdates:
        prompt = render_opening_fact_extract_prompt(
            player_role=request.player_role,
            opening_text=response.text,
            pre_turn_facts_text=_format_active_facts(request),
        )
        try:
            return call_with_retries(
                lambda: self.auditor.complete(
                    ({"role": "user", "content": prompt},),
                    stage="opening_fact_extract",
                ),
                _parse_fact_updates,
                stage="opening_fact_extract",
            )
        except ModelCallError as exc:
            raise OpeningEvaluationError(str(exc)) from exc


def render_opening_fact_extract_prompt(
    *, player_role: str, opening_text: str, pre_turn_facts_text: str
) -> str:
    """Render the frozen evaluator prompt used after opening generation."""

    return files("ncpbench.prompts").joinpath("fact_update.txt").read_text(
        encoding="utf-8"
    ).format(
        player_role=player_role or "player",
        pre_turn_facts=pre_turn_facts_text or "(No explicit fact data available)",
        response_text=opening_text or "(No narrative content available)",
    )


def _format_active_facts(request: OpeningRequest) -> str:
    lines = [f"- {fact.id}: {fact.text}" for fact in request.active_facts if fact.active]
    return "\n".join(lines) or "(none)"


def _parse_fact_updates(raw: str) -> PendingFactUpdates:
    payload = parse_json_object(raw)
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ModelOutputError("reason must be a non-empty string")
    return PendingFactUpdates(
        add_facts=_string_list(payload.get("add_facts"), "add_facts"),
        negate_fact_ids=_string_list(payload.get("negate_facts"), "negate_facts"),
    )


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ModelOutputError(f"{label} must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ModelOutputError(f"{label}[{index}] must be a non-empty string")
        result.append(item.strip())
    return tuple(result)
