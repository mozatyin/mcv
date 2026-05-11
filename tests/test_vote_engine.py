from user_soul.engines.vote import VoteEngine
from user_soul.models import AgentProfile, DecisionResult, FeatureAAR


class _FakeBackend:
    def text(self, prompt, **kw):
        return '{"choice": "Must-Have", "reasoning": "core feature"}'
    def vision(self, prompt, images, **kw):
        return ""


def _make_persona():
    return AgentProfile(
        agent_id="a1", archetype_name="Casual",
        trait_vector={"skill": 3.0}, background_story="小明，25岁学生")


def test_classify_returns_decision():
    engine = VoteEngine(_FakeBackend())
    result = engine.classify(
        question="Kano category?",
        options=["Must-Have", "Delighter"],
        context="chess game", personas=[_make_persona()])
    assert isinstance(result, DecisionResult)
    assert result.value == "Must-Have"


def test_score_returns_number():
    class _ScoreBackend:
        def text(self, prompt, **kw):
            return '{"score": 7.5, "reasoning": "good"}'
        def vision(self, prompt, images, **kw):
            return ""
    engine = VoteEngine(_ScoreBackend())
    result = engine.score("How important?", 0, 10, "chess", [_make_persona()])
    assert isinstance(result, DecisionResult)
    assert isinstance(result.value, float)
    assert 0 <= result.value <= 10


def test_classify_multi_persona():
    call_count = 0
    class _AlternatingBackend:
        def text(self, prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                return '{"choice": "Delighter", "reasoning": "nice"}'
            return '{"choice": "Must-Have", "reasoning": "core"}'
        def vision(self, prompt, images, **kw):
            return ""
    personas = [_make_persona() for _ in range(3)]
    engine = VoteEngine(_AlternatingBackend())
    result = engine.classify("Kano?", ["Must-Have", "Delighter"], "chess", personas)
    assert result.value == "Must-Have"
    assert result.confidence > 0.5


def test_aarrr_returns_feature_list():
    class _AARRRBackend:
        def text(self, prompt, **kw):
            import json
            return json.dumps([{
                "feature_id": "f1",
                "archetype_votes": {"Casual": {"acquisition": 0.8, "activation": 0.7, "retention": 0.6, "revenue": 0.3, "referral": 0.2}},
                "mean": {"acquisition": 0.8, "activation": 0.7, "retention": 0.6, "revenue": 0.3, "referral": 0.2}
            }])
        def vision(self, prompt, images, **kw):
            return ""
    engine = VoteEngine(_AARRRBackend())
    from user_soul.models import Archetype
    archs = [Archetype("Casual", 1.0, "casual player", {}, "小明")]
    result = engine.aarrr("chess game", [{"id": "f1", "name": "matchmaking"}], archs)
    assert len(result) == 1
    assert isinstance(result[0], FeatureAAR)
    assert result[0].acquisition == 0.8
