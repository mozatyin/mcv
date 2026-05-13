"""Action Router — converts differential diagnosis into structured ActionSpecs.

PM-Soul calls route_diagnosis() to get a prioritized list of ActionSpecs,
each targeting a specific actor (Code-Soul, ELTM, or PM-Soul itself).
Routing is deterministic — no LLM calls, pure pattern matching.

Design principles:
  - Every diagnosis maps to at least one ActionSpec (no silent drops)
  - PM-Soul is the supervisor/fallback for unroutable items
  - Closed-loop: every ActionSpec carries enough context for replay verification
"""
from __future__ import annotations

from user_soul.models import (
    ActionSpec, GradedPlaytestFeedback, PlaytestFeedback, PlaytestIssue,
)
from user_soul.game_knowledge import (
    DiagnosisCategory, DiagnosisItem, DifferentialDiagnosis,
    KnowledgeTier, TierResult,
)


_DIAGNOSIS_TO_ACTIONS: dict[DiagnosisCategory, list[tuple[str, str]]] = {
    DiagnosisCategory.CODE_BUG: [
        ("code-soul", "fix"),
    ],
    DiagnosisCategory.FLOW_FRICTION: [
        ("code-soul", "fix"),
    ],
    DiagnosisCategory.TUTORIAL_NEEDED: [
        ("eltm", "add_help"),
    ],
    DiagnosisCategory.DISCOVERABILITY: [
        ("eltm", "redesign"),
    ],
    DiagnosisCategory.HELP_SKIPPABLE: [
        ("pm-soul", "decide"),
    ],
    DiagnosisCategory.INCONCLUSIVE: [
        ("pm-soul", "triage"),
    ],
}


def route_diagnosis(feedback: GradedPlaytestFeedback) -> list[ActionSpec]:
    """Convert a GradedPlaytestFeedback into prioritized ActionSpecs.

    Each DiagnosisItem becomes one or more ActionSpecs. Additionally,
    per-tier issues that weren't captured by the diagnosis matrix get
    routed via fallback rules.
    """
    specs: list[ActionSpec] = []

    if feedback.diagnosis is None:
        return _route_flat_feedback(feedback)

    diagnosis: DifferentialDiagnosis = feedback.diagnosis

    for diag in diagnosis.diagnoses:
        actions = _DIAGNOSIS_TO_ACTIONS.get(
            diag.category,
            [("pm-soul", "triage")],
        )
        for owner, action_type in actions:
            evidence_lines = [
                f"{tier}: {obs}" for tier, obs in diag.evidence.items()
            ]
            specs.append(ActionSpec(
                owner=owner,
                action_type=action_type,
                severity=diag.severity,
                description=diag.description,
                evidence=evidence_lines,
                payload=_build_payload(diag, feedback),
                source_tier="",
                diagnosis_category=diag.category.value,
            ))

    # Supplement: severe per-tier issues not covered by diagnosis patterns
    covered_descriptions = {s.description for s in specs}
    for tier_val, fb in feedback.tier_feedbacks.items():
        for issue in fb.issues:
            if issue.severity == "P0" and issue.description not in covered_descriptions:
                specs.append(_issue_to_action_spec(issue, tier_val))

    specs.sort(key=lambda s: {"P0": 0, "P1": 1, "P2": 2}.get(s.severity, 3))
    return specs


def route_flat_feedback(feedback: PlaytestFeedback) -> list[ActionSpec]:
    """Route a single-tier PlaytestFeedback (backward compatible)."""
    return _route_flat_feedback(feedback)


def _route_flat_feedback(feedback) -> list[ActionSpec]:
    """Fallback routing for non-graded feedback."""
    specs: list[ActionSpec] = []
    for issue in feedback.issues:
        specs.append(_issue_to_action_spec(issue, ""))
    specs.sort(key=lambda s: {"P0": 0, "P1": 1, "P2": 2}.get(s.severity, 3))
    return specs


def _issue_to_action_spec(issue: PlaytestIssue, source_tier: str) -> ActionSpec:
    """Convert a single PlaytestIssue to an ActionSpec using category routing."""
    category = issue.category or "unknown"

    if category == "code_bug":
        owner, action_type = "code-soul", "fix"
    elif category == "design_issue":
        owner, action_type = "eltm", "redesign"
    elif category == "ux_friction":
        owner, action_type = "eltm", "add_help"
    else:
        owner, action_type = "pm-soul", "triage"

    return ActionSpec(
        owner=owner,
        action_type=action_type,
        severity=issue.severity,
        description=issue.description,
        evidence=issue.evidence[:3],
        payload={
            "affected_personas": issue.affected_personas,
            "category": category,
        },
        source_tier=source_tier,
    )


def _build_payload(diag: DiagnosisItem, feedback: GradedPlaytestFeedback) -> dict:
    """Build action payload with context needed for execution."""
    payload: dict = {
        "diagnosis_category": diag.category.value,
    }

    if diag.common_failure_turn is not None:
        payload["failure_turn"] = diag.common_failure_turn

    if diag.category == DiagnosisCategory.CODE_BUG:
        all_evidence = []
        for tier_val, fb in feedback.tier_feedbacks.items():
            for issue in fb.issues:
                for ev in issue.evidence[:2]:
                    all_evidence.append(f"[{tier_val}] {ev}")
        payload["all_tier_evidence"] = all_evidence[:10]

    elif diag.category == DiagnosisCategory.TUTORIAL_NEEDED:
        novice_fb = feedback.tier_feedbacks.get("novice")
        if novice_fb:
            payload["novice_give_up_reasons"] = novice_fb.suggestions[:5]
            payload["novice_score"] = novice_fb.score

    elif diag.category == DiagnosisCategory.DISCOVERABILITY:
        casual_fb = feedback.tier_feedbacks.get("casual")
        if casual_fb:
            payload["casual_friction_points"] = [
                iss.description for iss in casual_fb.issues
            ]
            payload["casual_score"] = casual_fb.score

    elif diag.category == DiagnosisCategory.FLOW_FRICTION:
        informed_fb = feedback.tier_feedbacks.get("informed")
        if informed_fb:
            payload["informed_issues"] = [
                {"severity": iss.severity, "description": iss.description}
                for iss in informed_fb.issues
            ]

    return payload


def group_by_owner(specs: list[ActionSpec]) -> dict[str, list[ActionSpec]]:
    """Group action specs by owner for dispatch."""
    groups: dict[str, list[ActionSpec]] = {}
    for spec in specs:
        if spec.owner not in groups:
            groups[spec.owner] = []
        groups[spec.owner].append(spec)
    return groups


def format_action_summary(specs: list[ActionSpec]) -> str:
    """Human-readable summary of all action specs."""
    if not specs:
        return "No actions needed — ship it."

    lines = []
    by_owner = group_by_owner(specs)

    for owner in ["code-soul", "eltm", "pm-soul"]:
        owner_specs = by_owner.get(owner, [])
        if not owner_specs:
            continue
        lines.append(f"\n{owner.upper()} ({len(owner_specs)} actions):")
        for spec in owner_specs:
            tier_tag = f" [{spec.source_tier}]" if spec.source_tier else ""
            lines.append(
                f"  [{spec.severity}] {spec.action_type}{tier_tag}: "
                f"{spec.description[:80]}"
            )

    return "\n".join(lines)
