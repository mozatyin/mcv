"""Tests for game_knowledge — knowledge-graded virtual testing."""
from __future__ import annotations

from user_soul.game_knowledge import (
    GameKnowledge, KnowledgeTier, brief_for_tier,
    DifferentialDiagnosis, TierResult, DiagnosisItem, DiagnosisCategory,
    distribute_personas_across_tiers, minimum_personas_for_graded_test,
    _is_strategy, _is_interaction_constraint, _infer_category,
    _find_common_failure_turns,
)


# --- GameKnowledge extraction ---

_TTT_GDD = {
    "game_name": "Tic-Tac-Toe",
    "formal_rules": {
        "game_loop_type": "turn-based",
        "objective": "Get three of your symbols in a row",
        "turn_structure": "Click an empty cell to place your mark",
        "win_conditions": [
            "Three in a row horizontally",
            "Three in a column vertically",
            "Three in a diagonal",
            "Draw if board is full",
        ],
        "special_rules": [
            "Cannot place symbol in occupied cell",
            "X always goes first",
            "Game ends immediately when win condition is met",
            "The center cell (position 4) is strategically advantageous",
            "Corner cells are the second-best strategic positions",
            "A fork is when a player has two ways to win",
            "Players must make a move each turn",
        ],
    },
    "render_hints": {"board_type": "grid"},
    "game_modes": [{"name": "vs AI"}, {"name": "PvP"}],
}


def test_from_gdd_parses_basics():
    k = GameKnowledge.from_gdd(_TTT_GDD)
    assert k.game_name == "Tic-Tac-Toe"
    assert k.objective == "Get three of your symbols in a row"
    assert k.game_loop_type == "turn-based"
    assert len(k.win_conditions) == 4


def test_from_gdd_separates_strategy():
    k = GameKnowledge.from_gdd(_TTT_GDD)
    assert len(k.strategy_hints) >= 2
    for hint in k.strategy_hints:
        assert _is_strategy(hint)
    for rule in k.interaction_rules:
        assert not _is_strategy(rule)


def test_from_gdd_identifies_interaction_rules():
    k = GameKnowledge.from_gdd(_TTT_GDD)
    assert any("Cannot" in r for r in k.interaction_rules)
    assert any("first" in r.lower() for r in k.interaction_rules)


def test_from_gdd_empty():
    k = GameKnowledge.from_gdd({})
    assert k.game_name == ""
    assert k.objective == ""
    assert k.strategy_hints == []


def test_from_gdd_fallback_to_game_rules():
    gdd = {
        "game_rules": {
            "objective": "Win the game",
            "turn_structure": "Take turns",
        }
    }
    k = GameKnowledge.from_gdd(gdd)
    assert k.objective == "Win the game"


# --- Category inference ---

def test_infer_category_grid():
    assert _infer_category({"render_hints": {"board_type": "grid"}}, "turn-based") == "board game"


def test_infer_category_card():
    assert _infer_category({"render_hints": {"board_type": "card"}}, "") == "card game"


def test_infer_category_fallback():
    assert _infer_category({}, "turn-based") == "turn-based game"
    assert _infer_category({}, "") == "game"


# --- Strategy/interaction classification ---

def test_is_strategy():
    assert _is_strategy("The center cell is strategically advantageous")
    assert _is_strategy("A fork is when a player has two ways to win")
    assert not _is_strategy("Cannot place symbol in occupied cell")
    assert not _is_strategy("X always goes first")


def test_is_interaction_constraint():
    assert _is_interaction_constraint("Cannot place symbol in occupied cell")
    assert _is_interaction_constraint("X always goes first")
    assert _is_interaction_constraint("Players must make a move each turn")


# --- Tier briefings ---

def test_brief_novice_no_rules():
    k = GameKnowledge.from_gdd(_TTT_GDD)
    brief = brief_for_tier(k, KnowledgeTier.NOVICE)
    assert "NEVER played" in brief
    assert "board game" in brief
    assert "Get three" not in brief
    assert "Cannot place" not in brief


def test_brief_casual_has_objective_no_constraints():
    k = GameKnowledge.from_gdd(_TTT_GDD)
    brief = brief_for_tier(k, KnowledgeTier.CASUAL)
    assert "Get three" in brief
    assert "Click an empty cell" in brief
    assert "Cannot place" not in brief


def test_brief_informed_has_all_rules_no_strategy():
    k = GameKnowledge.from_gdd(_TTT_GDD)
    brief = brief_for_tier(k, KnowledgeTier.INFORMED)
    assert "Get three" in brief
    assert "Cannot place" in brief or "first" in brief
    assert "strategically" not in brief.lower()
    assert "fork" not in brief.lower()


def test_brief_tiers_have_increasing_game_knowledge():
    k = GameKnowledge.from_gdd(_TTT_GDD)
    novice = brief_for_tier(k, KnowledgeTier.NOVICE)
    casual = brief_for_tier(k, KnowledgeTier.CASUAL)
    informed = brief_for_tier(k, KnowledgeTier.INFORMED)
    # NOVICE has more INSTRUCTIONS (cognitive load constraints) but less GAME KNOWLEDGE
    # CASUAL has objective + turn structure
    # INFORMED has all rules
    assert "Objective" not in novice
    assert "Objective" in casual
    assert "Interaction rules" not in casual
    assert "Interaction rules" in informed
    assert len(casual) < len(informed)


# --- Persona distribution ---

def test_distribute_even():
    personas = ["a", "b", "c", "d", "e", "f"]
    dist = distribute_personas_across_tiers(personas)
    assert len(dist["novice"]) == 2
    assert len(dist["casual"]) == 2
    assert len(dist["informed"]) == 2


def test_distribute_remainder():
    personas = ["a", "b", "c", "d", "e"]
    dist = distribute_personas_across_tiers(personas)
    assert len(dist["novice"]) == 2
    assert len(dist["casual"]) == 2
    assert len(dist["informed"]) == 1


def test_distribute_minimum():
    personas = ["a", "b", "c"]
    dist = distribute_personas_across_tiers(personas)
    assert all(len(v) == 1 for v in dist.values())


def test_distribute_empty():
    dist = distribute_personas_across_tiers([])
    assert all(len(v) == 0 for v in dist.values())


def test_minimum_personas():
    assert minimum_personas_for_graded_test() == 3
    assert minimum_personas_for_graded_test(2) == 2


# --- TierResult properties ---

def test_tier_result_failed():
    r = TierResult(tier=KnowledgeTier.NOVICE, score=30, completed_rate=0.3,
                   gave_up_rate=0.7, friction_count=5)
    assert r.failed
    assert not r.struggled
    assert not r.smooth


def test_tier_result_struggled():
    r = TierResult(tier=KnowledgeTier.CASUAL, score=60, completed_rate=0.7,
                   gave_up_rate=0.3, friction_count=3)
    assert not r.failed
    assert r.struggled
    assert not r.smooth


def test_tier_result_smooth():
    r = TierResult(tier=KnowledgeTier.INFORMED, score=85, completed_rate=0.9,
                   gave_up_rate=0.1, friction_count=1)
    assert not r.failed
    assert not r.struggled
    assert r.smooth


# --- Differential Diagnosis ---

def _make_tier_result(tier, score, gave_up_rate, friction_count=2):
    return TierResult(
        tier=tier, score=score,
        completed_rate=1.0 - gave_up_rate,
        gave_up_rate=gave_up_rate,
        friction_count=friction_count,
    )


def test_diagnosis_all_fail_code_bug():
    results = {
        "novice": _make_tier_result(KnowledgeTier.NOVICE, 20, 0.8),
        "casual": _make_tier_result(KnowledgeTier.CASUAL, 25, 0.7),
        "informed": _make_tier_result(KnowledgeTier.INFORMED, 30, 0.6),
    }
    d = DifferentialDiagnosis.from_tier_results("test", results)
    assert any(di.category == DiagnosisCategory.CODE_BUG for di in d.diagnoses)
    assert d.diagnoses[0].owner == "code-soul"
    assert d.diagnoses[0].severity == "P0"


def test_diagnosis_only_novice_fails_tutorial():
    results = {
        "novice": _make_tier_result(KnowledgeTier.NOVICE, 30, 0.7),
        "casual": _make_tier_result(KnowledgeTier.CASUAL, 75, 0.1),
        "informed": _make_tier_result(KnowledgeTier.INFORMED, 85, 0.0),
    }
    d = DifferentialDiagnosis.from_tier_results("test", results)
    assert any(di.category == DiagnosisCategory.TUTORIAL_NEEDED for di in d.diagnoses)
    assert any(di.owner == "eltm" for di in d.diagnoses)


def test_diagnosis_novice_casual_fail_discoverability():
    results = {
        "novice": _make_tier_result(KnowledgeTier.NOVICE, 20, 0.8),
        "casual": _make_tier_result(KnowledgeTier.CASUAL, 30, 0.6),
        "informed": _make_tier_result(KnowledgeTier.INFORMED, 80, 0.1),
    }
    d = DifferentialDiagnosis.from_tier_results("test", results)
    assert any(di.category == DiagnosisCategory.DISCOVERABILITY for di in d.diagnoses)
    assert any(di.severity == "P0" for di in d.diagnoses)


def test_diagnosis_informed_struggles_flow_friction():
    results = {
        "novice": _make_tier_result(KnowledgeTier.NOVICE, 75, 0.1),
        "casual": _make_tier_result(KnowledgeTier.CASUAL, 72, 0.15),
        "informed": _make_tier_result(KnowledgeTier.INFORMED, 55, 0.3),
    }
    d = DifferentialDiagnosis.from_tier_results("test", results)
    assert any(di.category == DiagnosisCategory.FLOW_FRICTION for di in d.diagnoses)
    assert any(di.owner == "code-soul" for di in d.diagnoses)


def test_diagnosis_partial_run():
    results = {
        "novice": _make_tier_result(KnowledgeTier.NOVICE, 50, 0.4),
    }
    d = DifferentialDiagnosis.from_tier_results("test", results)
    assert "Partial" in d.summary


def test_diagnosis_all_smooth_inconclusive():
    results = {
        "novice": _make_tier_result(KnowledgeTier.NOVICE, 80, 0.1),
        "casual": _make_tier_result(KnowledgeTier.CASUAL, 85, 0.05),
        "informed": _make_tier_result(KnowledgeTier.INFORMED, 90, 0.0),
    }
    d = DifferentialDiagnosis.from_tier_results("test", results)
    assert any(di.category == DiagnosisCategory.INCONCLUSIVE for di in d.diagnoses)


# --- Common failure turns ---

def test_find_common_failure_turns():
    turns = _find_common_failure_turns([2, 5], [3, 8], [2, 9])
    assert 2 in turns or 3 in turns


def test_find_common_failure_turns_empty():
    assert _find_common_failure_turns([], [], []) == []


# --- Summary ---

def test_diagnosis_summary_has_tiers():
    results = {
        "novice": _make_tier_result(KnowledgeTier.NOVICE, 30, 0.7),
        "casual": _make_tier_result(KnowledgeTier.CASUAL, 75, 0.1),
        "informed": _make_tier_result(KnowledgeTier.INFORMED, 85, 0.0),
    }
    d = DifferentialDiagnosis.from_tier_results("test", results)
    assert "NOVICE" in d.summary
    assert "CASUAL" in d.summary
    assert "INFORMED" in d.summary
    assert "Diagnoses:" in d.summary
