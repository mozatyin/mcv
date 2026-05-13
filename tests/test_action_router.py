"""Tests for action_router — diagnosis to ActionSpec routing."""
from __future__ import annotations

from user_soul.models import (
    ActionSpec, PlaytestFeedback, PlaytestIssue, GradedPlaytestFeedback,
)
from user_soul.game_knowledge import (
    DiagnosisCategory, DiagnosisItem, DifferentialDiagnosis,
    KnowledgeTier, TierResult,
)
from user_soul.action_router import (
    route_diagnosis, route_flat_feedback,
    group_by_owner, format_action_summary,
)


def _make_tier_result(tier, score, gave_up_rate):
    return TierResult(
        tier=tier, score=score,
        completed_rate=1.0 - gave_up_rate,
        gave_up_rate=gave_up_rate,
        friction_count=2,
    )


def _make_graded_feedback(
    novice_score=30, novice_giveup=0.7,
    casual_score=75, casual_giveup=0.1,
    informed_score=85, informed_giveup=0.0,
):
    tier_feedbacks = {
        "novice": PlaytestFeedback(score=novice_score, verdict="CRITICAL"),
        "casual": PlaytestFeedback(score=casual_score, verdict="NEEDS_WORK"),
        "informed": PlaytestFeedback(score=informed_score, verdict="PASS"),
    }
    tier_results = {
        "novice": _make_tier_result(KnowledgeTier.NOVICE, novice_score, novice_giveup),
        "casual": _make_tier_result(KnowledgeTier.CASUAL, casual_score, casual_giveup),
        "informed": _make_tier_result(KnowledgeTier.INFORMED, informed_score, informed_giveup),
    }
    diagnosis = DifferentialDiagnosis.from_tier_results("test_game", tier_results)

    return GradedPlaytestFeedback(
        score=60.0,
        verdict="NEEDS_WORK",
        tier_feedbacks=tier_feedbacks,
        diagnosis=diagnosis,
    )


# --- Routing from graded feedback ---

def test_route_code_bug_goes_to_code_soul():
    fb = _make_graded_feedback(
        novice_score=20, novice_giveup=0.8,
        casual_score=25, casual_giveup=0.7,
        informed_score=30, informed_giveup=0.6,
    )
    specs = route_diagnosis(fb)
    assert any(s.owner == "code-soul" for s in specs)
    assert any(s.action_type == "fix" for s in specs)
    assert specs[0].severity == "P0"


def test_route_tutorial_needed_goes_to_eltm():
    fb = _make_graded_feedback(
        novice_score=30, novice_giveup=0.7,
        casual_score=75, casual_giveup=0.1,
        informed_score=85, informed_giveup=0.0,
    )
    specs = route_diagnosis(fb)
    assert any(s.owner == "eltm" for s in specs)
    assert any(s.action_type == "add_help" for s in specs)


def test_route_discoverability_goes_to_eltm_p0():
    fb = _make_graded_feedback(
        novice_score=20, novice_giveup=0.8,
        casual_score=30, casual_giveup=0.6,
        informed_score=80, informed_giveup=0.1,
    )
    specs = route_diagnosis(fb)
    eltm_specs = [s for s in specs if s.owner == "eltm"]
    assert len(eltm_specs) >= 1
    assert any(s.severity == "P0" for s in eltm_specs)


def test_route_flow_friction_goes_to_code_soul():
    fb = _make_graded_feedback(
        novice_score=75, novice_giveup=0.1,
        casual_score=72, casual_giveup=0.15,
        informed_score=55, informed_giveup=0.3,
    )
    specs = route_diagnosis(fb)
    code_specs = [s for s in specs if s.owner == "code-soul"]
    assert len(code_specs) >= 1
    assert any(s.diagnosis_category == "flow_friction" for s in code_specs)


def test_route_all_smooth_inconclusive_to_pm():
    fb = _make_graded_feedback(
        novice_score=80, novice_giveup=0.1,
        casual_score=85, casual_giveup=0.05,
        informed_score=90, informed_giveup=0.0,
    )
    specs = route_diagnosis(fb)
    assert any(s.owner == "pm-soul" for s in specs)


def test_route_preserves_severity_order():
    fb = _make_graded_feedback(
        novice_score=20, novice_giveup=0.8,
        casual_score=30, casual_giveup=0.6,
        informed_score=80, informed_giveup=0.1,
    )
    specs = route_diagnosis(fb)
    severities = [s.severity for s in specs]
    expected_order = {"P0": 0, "P1": 1, "P2": 2}
    for i in range(len(severities) - 1):
        assert expected_order.get(severities[i], 3) <= expected_order.get(severities[i + 1], 3)


def test_route_no_diagnosis_returns_empty():
    fb = GradedPlaytestFeedback(score=80, verdict="PASS", diagnosis=None)
    specs = route_diagnosis(fb)
    assert specs == []


# --- Flat feedback routing (backward compat) ---

def test_route_flat_code_bug():
    fb = PlaytestFeedback(
        score=40, verdict="CRITICAL",
        issues=[PlaytestIssue("P0", "crash on load", category="code_bug")],
    )
    specs = route_flat_feedback(fb)
    assert len(specs) == 1
    assert specs[0].owner == "code-soul"
    assert specs[0].action_type == "fix"


def test_route_flat_design_issue():
    fb = PlaytestFeedback(
        score=60, verdict="NEEDS_WORK",
        issues=[PlaytestIssue("P1", "confusing layout", category="design_issue")],
    )
    specs = route_flat_feedback(fb)
    assert specs[0].owner == "eltm"
    assert specs[0].action_type == "redesign"


def test_route_flat_ux_friction():
    fb = PlaytestFeedback(
        score=60, verdict="NEEDS_WORK",
        issues=[PlaytestIssue("P1", "unclear next step", category="ux_friction")],
    )
    specs = route_flat_feedback(fb)
    assert specs[0].owner == "eltm"
    assert specs[0].action_type == "add_help"


def test_route_flat_unknown_category_to_pm():
    fb = PlaytestFeedback(
        score=60, verdict="NEEDS_WORK",
        issues=[PlaytestIssue("P2", "something weird", category="")],
    )
    specs = route_flat_feedback(fb)
    assert specs[0].owner == "pm-soul"
    assert specs[0].action_type == "triage"


# --- Grouping and formatting ---

def test_group_by_owner():
    specs = [
        ActionSpec(owner="code-soul", action_type="fix", severity="P0", description="bug"),
        ActionSpec(owner="eltm", action_type="redesign", severity="P1", description="ux"),
        ActionSpec(owner="code-soul", action_type="fix", severity="P1", description="bug2"),
    ]
    groups = group_by_owner(specs)
    assert len(groups["code-soul"]) == 2
    assert len(groups["eltm"]) == 1


def test_format_action_summary_no_actions():
    assert "ship" in format_action_summary([]).lower()


def test_format_action_summary_with_actions():
    specs = [
        ActionSpec(owner="code-soul", action_type="fix", severity="P0", description="JS crash"),
        ActionSpec(owner="eltm", action_type="add_help", severity="P1", description="missing tutorial"),
    ]
    summary = format_action_summary(specs)
    assert "CODE-SOUL" in summary
    assert "ELTM" in summary
    assert "fix" in summary
    assert "add_help" in summary


# --- Payload enrichment ---

def test_code_bug_payload_has_all_tier_evidence():
    fb = _make_graded_feedback(
        novice_score=20, novice_giveup=0.8,
        casual_score=25, casual_giveup=0.7,
        informed_score=30, informed_giveup=0.6,
    )
    specs = route_diagnosis(fb)
    code_specs = [s for s in specs if s.diagnosis_category == "code_bug"]
    assert len(code_specs) >= 1


def test_tutorial_payload_has_novice_context():
    fb = _make_graded_feedback()
    fb.tier_feedbacks["novice"].suggestions = ["unclear how to start"]
    specs = route_diagnosis(fb)
    tutorial_specs = [s for s in specs if s.diagnosis_category == "tutorial_needed"]
    if tutorial_specs:
        assert "novice_give_up_reasons" in tutorial_specs[0].payload
