"""Plain narrator used for the NCP-Bench paper baseline."""

from __future__ import annotations

from importlib.resources import files
from typing import Mapping, Protocol, Sequence

from ncpbench.model_output import call_with_retries, parse_text_object
from ncpbench.narrator import (
    Narrator,
    NarratorRequest,
    NarratorResponse,
    OpeningRequest,
    render_opening_prompt,
)


class TextGenerationClient(Protocol):
    """A method-owned model client; NCP-Bench never supplies this dependency."""

    def complete(self, messages: Sequence[Mapping[str, str]], *, stage: str) -> str:
        """Return the raw completion for one method-owned generation request."""


class BaselineNarrator(Narrator):
    """The paper's plain context-conditioned narrator, without evaluator access."""

    def __init__(self, client: TextGenerationClient) -> None:
        self._client = client

    def open(self, request: OpeningRequest) -> NarratorResponse:
        messages = ({"role": "user", "content": render_opening_prompt(request)},)
        text = call_with_retries(
            lambda: self._client.complete(messages, stage="opening"),
            parse_text_object,
            stage="opening",
        )
        return NarratorResponse(text=text)

    def respond(self, request: NarratorRequest) -> NarratorResponse:
        prompt = build_generation_prompt(request)
        messages = ({"role": "user", "content": prompt},)
        text = call_with_retries(
            lambda: self._client.complete(messages, stage="method_response_generate"),
            parse_text_object,
            stage="method_response_generate",
        )
        return NarratorResponse(text=text)


def build_generation_prompt(request: NarratorRequest) -> str:
    """Render the paper prompt from the public narrator request only."""

    return _prompt_template().format(
        player_role=request.player_role or "玩家",
        current_node=_current_node(request),
        narrative_commitments=_format_commitments(request),
        pre_turn_facts=_format_facts(request),
        pending_fact_updates=_format_pending_updates(request),
        system_response_history=_format_system_response_history(_format_history(request)),
        user_input=request.player_input,
    )


def _prompt_template() -> str:
    return files("ncpbench_reference_baseline.prompts").joinpath("method_response_generate_user.txt").read_text(
        encoding="utf-8"
    )


def _current_node(request: NarratorRequest) -> str:
    if not request.trajectory:
        return "(None)"
    current_index = 0
    for index, node in enumerate(request.trajectory):
        if node.occurred:
            current_index = index
    node = request.trajectory[current_index]
    return "\n".join(
        (
            f"- node_id: {node.id}",
            f" | description: {node.description}",
            f" | trigger: {node.trigger_event}",
            f" | delta: {node.key_delta}",
        )
    )


def _format_commitments(request: NarratorRequest) -> str:
    if not request.commitments:
        return "(无)"
    lines: list[str] = []
    for commitment in request.commitments:
        lines.extend(
            (
                f"- {commitment.id}: type={commitment.kind} | status={commitment.status}",
                f"  description: {commitment.description}",
                f"  satisfaction_condition: {commitment.satisfaction_condition}",
                f"  violation_condition: {commitment.violation_condition}",
            )
        )
    return "\n".join(lines)


def _format_facts(request: NarratorRequest) -> str:
    if not request.active_facts:
        return "(none)"
    return "\n".join(f"- {fact.id}: {fact.text}" for fact in request.active_facts)


def _format_pending_updates(request: NarratorRequest) -> str:
    fact_lookup = {fact.id: fact.text for fact in request.active_facts}
    add_lines = [f"- {text}" for text in request.pending_fact_updates.add_facts if text.strip()]
    negate_lines: list[str] = []
    for fact_id in request.pending_fact_updates.negate_fact_ids:
        normalized_id = fact_id.strip()
        if not normalized_id:
            continue
        text = fact_lookup.get(normalized_id)
        negate_lines.append(f"- {normalized_id}: {text}" if text else f"- {normalized_id}")
    add_block = "\n".join(add_lines) if add_lines else "(none)"
    negate_block = "\n".join(negate_lines) if negate_lines else "(none)"
    return f"add_facts:\n{add_block}\nnegate_facts:\n{negate_block}"


def _format_history(request: NarratorRequest) -> str:
    lines: list[str] = []
    for turn in request.history:
        if turn.turn_id >= 0 and turn.narrator_response is None:
            continue
        if turn.turn_id == -1:
            lines.extend(("Opening:", turn.narrator_response or "(No opening narrative)"))
            continue
        lines.extend(
            (
                f"Turn {turn.turn_id}:",
                f"- Player: {turn.player_input or '(None)'}",
                f"- System: {turn.narrator_response or '(None)'}",
            )
        )
    return "\n".join(lines)


def _format_system_response_history(history_text: str) -> str:
    stripped = history_text.strip()
    if not stripped or stripped == "(The story has just begun)":
        return "(No prior system responses)"

    lines, chunks, index = stripped.splitlines(), [], 0
    while index < len(lines):
        line = lines[index].strip()
        if line == "Opening:":
            opening: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("Turn "):
                if lines[index].strip():
                    opening.append(lines[index].strip())
                index += 1
            if opening:
                chunks.extend(("Opening System Response:", *opening))
            continue
        if line.startswith("Turn "):
            label = line
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("Turn "):
                current = lines[index].strip()
                if current.startswith("- System:"):
                    chunks.extend((f"{label} System Response:", current[len("- System:") :].strip() or "(None)"))
                    break
                index += 1
            continue
        index += 1
    return "\n".join(chunks) if chunks else "(No prior system responses)"
