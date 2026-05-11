from user_soul.engines.persona import PersonaEngine
from user_soul.models import PersonaStructure, AgentProfile


class _FakeBackend:
    def text(self, prompt, **kw):
        return '{"population_label":"gamers","product_context":"chess","trait_dimensions":[{"name":"skill","description":"d","low_label":"l","high_label":"h","distribution":"normal","mean":5,"std":2,"source":"space1"}],"archetypes":[{"name":"Casual","frequency":0.6,"description":"casual player","background_story":"小明，25岁","trait_constraints":{"skill":[1,5]}},{"name":"Hardcore","frequency":0.4,"description":"serious player","background_story":"老王，40岁","trait_constraints":{"skill":[6,10]}}],"research_notes":"test"}'

    def vision(self, prompt, images, **kw):
        return ""


def test_research_returns_persona_structure():
    engine = PersonaEngine(_FakeBackend())
    result = engine.research("chess game")
    assert isinstance(result, PersonaStructure)
    assert len(result.archetypes) == 2


def test_generate_pool_returns_agent_profiles():
    engine = PersonaEngine(_FakeBackend())
    structure = engine.research("chess game")
    pool = engine.generate_pool(structure, n=5)
    assert len(pool) == 5
    assert all(isinstance(a, AgentProfile) for a in pool)


def test_get_or_create_convenience():
    engine = PersonaEngine(_FakeBackend())
    pool = engine.get_or_create("chess game", n=3)
    assert len(pool) == 3


def test_pool_respects_archetype_constraints():
    engine = PersonaEngine(_FakeBackend())
    structure = engine.research("chess game")
    pool = engine.generate_pool(structure, n=20)
    for agent in pool:
        if agent.archetype_name == "Casual":
            assert agent.trait_vector["skill"] <= 5.0
        elif agent.archetype_name == "Hardcore":
            assert agent.trait_vector["skill"] >= 6.0


def test_fallback_on_bad_json():
    class _BadBackend:
        def text(self, prompt, **kw):
            return "not json"

        def vision(self, prompt, images, **kw):
            return ""

    engine = PersonaEngine(_BadBackend())
    result = engine.research("test product")
    assert isinstance(result, PersonaStructure)
    assert len(result.archetypes) == 2
