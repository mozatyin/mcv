"""Tests for playtest_bridge — User-Soul ↔ Code-Soul integration."""
from __future__ import annotations
import json
from user_soul.models import (
    AgentProfile, PlaytestFeedback, PlaytestIssue,
)
from user_soul.playtest_bridge import (
    _agent_to_persona,
    _make_llm_caller,
    _group_frictions,
    _classify_issues,
    _translate_friction_report,
    extract_game_rules,
)


class _FakeBackend:
    def __init__(self):
        self.last_call = None

    def text(self, prompt, **kw):
        self.last_call = ("text", prompt)
        return '{"action": "click", "selector": "#btn", "reason": "test"}'

    def vision(self, prompt, images, **kw):
        self.last_call = ("vision", prompt, len(images))
        return '{"action": "click", "selector": "#btn", "reason": "test"}'


def _make_agent(name="test", story="小明，25岁程序员"):
    return AgentProfile(
        agent_id=name,
        archetype_name="Tester",
        trait_vector={"engagement": 5.0},
        background_story=story,
    )


def test_agent_to_persona_converts():
    agent = _make_agent("agent_001", "小明，25岁深圳程序员，爱玩手游")
    persona = _agent_to_persona(agent)
    assert persona.name == "agent_001"
    assert "小明" in persona.profile
    assert len(persona.goals) > 0


def test_llm_caller_routes_vision_when_image():
    backend = _FakeBackend()
    caller = _make_llm_caller(backend)
    import base64
    img_b64 = base64.b64encode(b"fake_png").decode()
    caller("system", "user", img_b64, "key")
    assert backend.last_call[0] == "vision"


def test_llm_caller_routes_text_when_no_image():
    backend = _FakeBackend()
    caller = _make_llm_caller(backend)
    caller("system", "user", None, "key")
    assert backend.last_call[0] == "text"


def test_group_frictions_clusters_by_kind():
    class FakeEvent:
        def __init__(self, kind, detail, turn=0):
            self.kind = kind
            self.detail = detail
            self.turn = turn

    class FakeResult:
        def __init__(self, persona, events):
            self.persona = persona
            self.friction_events = events
            self.gave_up = False
            self.completed = True

    results = [
        FakeResult("p1", [
            FakeEvent("dead_end", "no DOM change"),
            FakeEvent("dead_end", "no DOM change"),
            FakeEvent("wrong_action", "Timeout 2000ms"),
        ]),
        FakeResult("p2", [
            FakeEvent("give_up", "stuck"),
        ]),
    ]
    grouped = _group_frictions(results)
    assert "dead_end" in grouped
    assert len(grouped["dead_end"]) == 2
    assert "selector_timeout" in grouped
    assert "give_up" in grouped


def test_classify_issues_severity():
    grouped = {
        "dead_end": [
            {"persona": "p1", "turn": 3, "detail": "no change", "kind": "dead_end"},
            {"persona": "p2", "turn": 5, "detail": "no change", "kind": "dead_end"},
            {"persona": "p3", "turn": 7, "detail": "no change", "kind": "dead_end"},
        ],
        "give_up": [
            {"persona": "p1", "turn": 10, "detail": "frustrated", "kind": "give_up"},
            {"persona": "p2", "turn": 8, "detail": "lost", "kind": "give_up"},
        ],
        "error": [
            {"persona": "p1", "turn": 1, "detail": "JS error", "kind": "error"},
        ],
    }
    issues = _classify_issues(grouped)
    severities = [i.severity for i in issues]
    assert "P0" in severities
    assert issues[0].severity == "P0"


def test_playtest_feedback_model():
    fb = PlaytestFeedback(
        score=65.0,
        verdict="NEEDS_WORK",
        issues=[
            PlaytestIssue("P0", "crash on load", category="code_bug"),
            PlaytestIssue("P1", "confusing nav", category="ux_friction"),
            PlaytestIssue("P2", "minor text", category="design_issue"),
        ],
    )
    assert fb.has_blockers
    assert len(fb.p0_issues) == 1
    assert len(fb.p1_issues) == 1
    assert fb.verdict == "NEEDS_WORK"


def test_playtest_feedback_pass():
    fb = PlaytestFeedback(score=85.0, verdict="PASS")
    assert not fb.has_blockers
    assert fb.p0_issues == []


def test_translate_friction_report_minimal():
    class FakePersonaResult:
        def __init__(self, persona):
            self.persona = persona
            self.friction_events = []
            self.completed = True
            self.gave_up = False

    class FakeReport:
        def __init__(self):
            self.score = 90.0
            self.per_persona_results = [FakePersonaResult("p1")]
            self.summary = "clean run"

    agents = [_make_agent()]
    fb = _translate_friction_report(FakeReport(), agents, None)
    assert fb.verdict == "PASS"
    assert fb.score == 90.0
    assert fb.issues == []


def test_extract_game_rules_from_gdd():
    gdd = {
        "formal_rules": {
            "objective": "Get three in a row",
            "turn_structure": "Click an empty cell to place your mark",
            "win_conditions": ["Three in a row", "Three in a column"],
            "special_rules": [
                "Cannot place symbol in occupied cell",
                "X always goes first",
                "Players must make a move each turn",
            ],
        },
        "game_modes": [
            {"name": "vs AI"},
            {"name": "PvP"},
        ],
    }
    rules = extract_game_rules(gdd)
    assert "Get three in a row" in rules
    assert "Click an empty cell" in rules
    assert "Cannot place symbol in occupied cell" in rules
    assert "vs AI" in rules


def test_extract_game_rules_empty_gdd():
    assert extract_game_rules({}) == ""
    assert extract_game_rules({"formal_rules": {}}) == ""


def test_llm_caller_injects_game_rules():
    backend = _FakeBackend()
    rules = "Objective: Get three in a row\nHow to play: Click empty cells"
    caller = _make_llm_caller(backend, game_rules=rules)
    caller("system prompt", "user prompt", None, "key")
    assert backend.last_call[0] == "text"
    prompt_sent = backend.last_call[1]
    assert "Get three in a row" in prompt_sent
    assert "Click empty cells" in prompt_sent


def test_llm_caller_no_rules_leaves_system_unchanged():
    backend = _FakeBackend()
    caller = _make_llm_caller(backend, game_rules="")
    caller("system prompt", "user prompt", None, "key")
    prompt_sent = backend.last_call[1]
    assert prompt_sent.startswith("system prompt")
