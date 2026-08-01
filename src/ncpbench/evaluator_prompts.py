"""Prompt rendering owned by the evaluator, never by narrator methods."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ncpbench.evaluator import (
        CommitmentCheckRequest,
        FactUpdate,
        TrajectoryCheckRequest,
        TurnAuditContext,
    )


@dataclass(frozen=True)
class EvaluationPrompts:
    """The fixed prompt texts that define the NCP-Bench evaluator protocol."""

    fact_update: str
    conflict_check: str
    conflict_double_check: str
    trajectory_check: str
    status_check: str


def load_evaluation_prompts() -> EvaluationPrompts:
    """Load the frozen evaluator prompts shipped with NCP-Bench."""

    prompt_dir = files("ncpbench.prompts")
    return EvaluationPrompts(
        fact_update=prompt_dir.joinpath("fact_update.txt").read_text(encoding="utf-8"),
        conflict_check=prompt_dir.joinpath("conflict_check.txt").read_text(encoding="utf-8"),
        conflict_double_check=prompt_dir.joinpath("conflict_double_check.txt").read_text(encoding="utf-8"),
        trajectory_check=prompt_dir.joinpath("trajectory_check.txt").read_text(encoding="utf-8"),
        status_check=prompt_dir.joinpath("status_check.txt").read_text(encoding="utf-8"),
    )


def render_fact_update(prompts: EvaluationPrompts, context: "TurnAuditContext", response_text: str) -> str:
    return prompts.fact_update.format(
        player_role=context.player_role or "player",
        pre_turn_facts=_format_facts(context),
        response_text=response_text or "(No narrative content available)",
    )


def render_conflict_check(
    prompts: EvaluationPrompts,
    context: "TurnAuditContext",
    response_text: str,
    fact_update: "FactUpdate",
) -> str:
    return prompts.conflict_check.format(**_conflict_variables(context, response_text, fact_update))


def render_conflict_double_check(
    prompts: EvaluationPrompts,
    context: "TurnAuditContext",
    response_text: str,
    fact_update: "FactUpdate",
    initial_conflicts_json: str,
) -> str:
    variables = _conflict_variables(context, response_text, fact_update)
    variables["initial_conflicts_json"] = initial_conflicts_json or "[]"
    return prompts.conflict_double_check.format(**variables)


def render_trajectory_check(prompts: EvaluationPrompts, request: "TrajectoryCheckRequest") -> str:
    current_node = _current_trajectory_node(request)
    return prompts.trajectory_check.format(
        player_role=request.player_role.strip() or "player",
        facts=_format_checker_facts(request.active_facts),
        history_text=_format_checker_history(request.history),
        current_node=_format_checker_node(current_node),
    )


def render_status_check(prompts: EvaluationPrompts, request: "CommitmentCheckRequest") -> str:
    return prompts.status_check.format(
        player_role=request.player_role.strip() or "player",
        facts=_format_checker_facts(request.active_facts),
        history_text=_format_checker_history(request.history),
        trajectory_progress=_format_checker_trajectory(request.trajectory),
        narrative_commitments=_format_checker_commitments(request),
    )


def _format_facts(context: "TurnAuditContext") -> str:
    if not context.active_facts:
        return "(none)"
    return "\n".join(f"- {fact.id}: {fact.text}" for fact in context.active_facts)


def _conflict_variables(
    context: "TurnAuditContext", response_text: str, fact_update: "FactUpdate"
) -> dict[str, str]:
    return {
        "player_role": context.player_role or "player",
        "system_response_history": _format_history(context),
        "user_input": context.player_input or "(None)",
        "response_text": response_text,
        "pre_turn_facts": _format_facts(context),
        "candidate_fact_updates": _format_fact_update(context, fact_update),
        "narrative_commitments": _format_commitments(context),
        "trajectory_progress": _format_trajectory(context),
        "current_node": _format_current_node(context),
    }


def _format_history(context: "TurnAuditContext") -> str:
    chunks: list[str] = []
    for turn in context.history:
        if turn.narrator_response is None:
            continue
        label = (
            "Opening System Response:"
            if turn.turn_id == -1
            else f"Turn {turn.turn_id} System Response:"
        )
        chunks.extend((label, turn.narrator_response))
    return "\n".join(chunks) or "(No prior system responses)"


def _format_fact_update(context: "TurnAuditContext", update: "FactUpdate") -> str:
    lookup = {fact.id: fact.text for fact in context.active_facts}
    added = "\n".join(f"- {fact}" for fact in update.add_facts) or "(none)"
    negated = "\n".join(
        f"- {fact_id}: {lookup[fact_id]}" if fact_id in lookup else f"- {fact_id}"
        for fact_id in update.negate_fact_ids
    ) or "(none)"
    return f"add_facts:\n{added}\nnegate_facts:\n{negated}"


def _format_commitments(context: "TurnAuditContext") -> str:
    if not context.commitments:
        return "(无)"
    return "\n".join(
        "\n".join(
            [
                f"- {commitment.id}: type={commitment.kind} | status={commitment.status}",
                f"  description: {commitment.description}",
                f"  satisfaction_condition: {commitment.satisfaction_condition}",
                f"  violation_condition: {commitment.violation_condition}",
            ]
        )
        for commitment in context.commitments
    )


def _format_trajectory(context: "TurnAuditContext") -> str:
    if not context.trajectory:
        return "(No trajectory progress available)"
    return "\n".join(_format_node(node) for node in context.trajectory)


def _format_current_node(context: "TurnAuditContext") -> str:
    current = None
    for node in context.trajectory:
        if not node.occurred:
            current = node
            break
    if current is None and context.trajectory:
        current = context.trajectory[-1]
    if current is None:
        return "(None)"
    return "\n".join(
        [
            f"- node_id: {current.id}",
            f" | description: {current.description}",
            f" | trigger: {current.trigger_event}",
            f" | delta: {current.key_delta}",
        ]
    )


def _format_node(node: object) -> str:
    status = "occurred" if node.occurred else "pending"
    return "\n".join(
        [
            f"- {node.id}: status={status}",
            f" | description: {node.description}",
            f" | trigger: {node.trigger_event}",
            f" | delta: {node.key_delta}",
        ]
    )


def _format_checker_facts(facts: object) -> str:
    if not facts:
        return "(No explicit fact data available)"
    lines = []
    for fact in facts:
        fact_id = str(getattr(fact, "id", "") or "").strip() or "f_?"
        text = str(getattr(fact, "text", "") or "").strip() or "(empty)"
        lines.append(f"- {fact_id}: {text}")
    return "\n".join(lines)


def _format_checker_history(history: object) -> str:
    if not history:
        return "(The story has just begun)"
    lines = []
    for turn in history:
        turn_id = getattr(turn, "turn_id", None)
        player_input = str(getattr(turn, "player_input", "") or "").strip()
        narrator_response = str(getattr(turn, "narrator_response", "") or "").strip()
        if turn_id == -1:
            lines.extend(("Opening:", narrator_response or "(No opening narrative)"))
            continue
        lines.extend(
            (
                f"Turn {turn_id}:",
                f"- Player: {player_input or '(None)'}",
                f"- System: {narrator_response or '(None)'}",
            )
        )
    return "\n".join(lines)


def _current_trajectory_node(request: "TrajectoryCheckRequest") -> object | None:
    for node in request.trajectory:
        if not node.occurred:
            return node
    return None


def _format_checker_node(node: object | None) -> str:
    if node is None:
        return "(None)"
    node_id = str(getattr(node, "id", "") or "").strip() or "s_?"
    trigger = str(getattr(node, "trigger_event", "") or "").strip() or "(No trigger event)"
    delta = str(getattr(node, "key_delta", "") or "").strip()
    lines = [f"- {node_id}:"]
    lines.append(f" | trigger: {trigger}")
    if delta:
        lines.append(f" | delta: {delta}")
    return "\n".join(lines)


def _format_checker_trajectory(trajectory: object) -> str:
    if not trajectory:
        return "(No trajectory progress available)"
    lines = []
    for node in trajectory:
        node_id = str(getattr(node, "id", "") or "").strip() or "s_?"
        description = str(getattr(node, "description", "") or "").strip() or "(No description)"
        status = "occurred" if bool(getattr(node, "occurred", False)) else "pending"
        lines.extend((f"- {node_id}: status={status}", f" | description: {description}"))
    return "\n".join(lines) if lines else "(No trajectory progress available)"


def _format_checker_commitments(request: "CommitmentCheckRequest") -> str:
    if not request.commitments:
        return "(No commitments)"
    lines = []
    for commitment in request.commitments:
        lines.extend(
            (
                f"- {commitment.id}: type={commitment.kind} | status={commitment.status}",
                f"  description: {commitment.description}",
                f"  satisfaction_condition: {commitment.satisfaction_condition}",
            )
        )
    return "\n".join(lines)
