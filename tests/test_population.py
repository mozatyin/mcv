import json
from unittest.mock import patch
from mcv.population import TraitDimension, Archetype, PersonaStructure, AgentProfile, PersonaPool, PopulationResearcher

def test_trait_dimension_fields():
    td = TraitDimension(
        name="ludo_familiarity",
        description="Real-world Ludo experience",
        low_label="never played Ludo before",
        high_label="expert Ludo player who teaches others",
        distribution="bimodal",
        mean=5.0,
        std=2.5,
        source="space2",
    )
    assert td.name == "ludo_familiarity"
    assert td.distribution == "bimodal"
    assert td.source == "space2"

def test_archetype_fields():
    arch = Archetype(
        name="Family Socializer",
        frequency=0.45,
        description="Plays Ludo with family, social motivation dominates",
        trait_constraints={"social_motivation": (6.0, 10.0), "ludo_familiarity": (3.0, 9.0)},
    )
    assert arch.frequency == 0.45
    assert arch.trait_constraints["social_motivation"] == (6.0, 10.0)

def test_persona_structure_fields():
    dims = [
        TraitDimension("social_motivation", "Social drive", "solo player", "social-first", "normal", 6.0, 2.0, "space2"),
    ]
    archs = [
        Archetype("Family Socializer", 0.60, "social player", {"social_motivation": (6.0, 10.0)}),
        Archetype("Solo Grinder",       0.40, "solo player",  {"social_motivation": (0.0, 4.0)}),
    ]
    ps = PersonaStructure(
        population_label="Arabic Ludo Players",
        product_context="Ludo mobile app for Arabic families",
        trait_dimensions=dims,
        archetypes=archs,
    )
    assert ps.population_label == "Arabic Ludo Players"
    assert len(ps.archetypes) == 2
    assert ps.research_notes == ""

def test_agent_profile_fields():
    ap = AgentProfile(
        agent_id="agent_001",
        archetype_name="Family Socializer",
        trait_vector={"social_motivation": 8.3, "ludo_familiarity": 5.1, "patience": 6.2},
    )
    assert ap.agent_id == "agent_001"
    assert ap.trait_vector["social_motivation"] == 8.3


def _make_structure() -> PersonaStructure:
    dims = [
        TraitDimension("social_motivation", "Social drive", "solo", "social", "normal", 6.0, 2.0, "space2"),
        TraitDimension("patience",           "Patience",    "quits fast", "very patient", "normal", 5.0, 2.0, "space2"),
    ]
    archs = [
        Archetype("Socializer", 0.60, "social", {"social_motivation": (6.0, 10.0)}),
        Archetype("Grinder",    0.40, "solo",   {"social_motivation": (0.0, 5.0)}),
    ]
    return PersonaStructure("Test Pop", "test product", dims, archs)

def test_pool_generates_n_agents():
    pool = PersonaPool(_make_structure())
    agents = pool.generate(30)
    assert len(agents) == 30

def test_pool_agents_have_unique_ids():
    pool = PersonaPool(_make_structure())
    agents = pool.generate(10)
    ids = [a.agent_id for a in agents]
    assert len(set(ids)) == 10

def test_pool_archetype_frequency_respected():
    """60% Socializer, 40% Grinder → at N=100, within ±15pp."""
    pool = PersonaPool(_make_structure())
    agents = pool.generate(100)
    socializers = sum(1 for a in agents if a.archetype_name == "Socializer")
    assert 45 <= socializers <= 75  # 60% ± 15pp

def test_pool_trait_constraints_respected():
    """Socializer agents must have social_motivation in [6.0, 10.0]."""
    pool = PersonaPool(_make_structure())
    agents = pool.generate(60)
    for a in agents:
        if a.archetype_name == "Socializer":
            assert 6.0 <= a.trait_vector["social_motivation"] <= 10.0
        elif a.archetype_name == "Grinder":
            assert 0.0 <= a.trait_vector["social_motivation"] <= 5.0

def test_pool_all_traits_in_0_10():
    pool = PersonaPool(_make_structure())
    agents = pool.generate(30)
    for a in agents:
        for val in a.trait_vector.values():
            assert 0.0 <= val <= 10.0

def test_pool_ids_unique_across_multiple_generate_calls():
    pool = PersonaPool(_make_structure())
    agents1 = pool.generate(5)
    agents2 = pool.generate(5)
    all_ids = [a.agent_id for a in agents1 + agents2]
    assert len(set(all_ids)) == 10

def test_behavioral_constraints_low_trait():
    dims = [TraitDimension("patience", "Patience", "quits immediately if confused", "waits indefinitely", "normal", 5.0, 2.0, "space2")]
    agent = AgentProfile("a1", "Grinder", {"patience": 1.5})
    text = agent.to_behavioral_constraints(dims)
    assert "quits immediately if confused" in text
    assert "1.5" in text

def test_behavioral_constraints_high_trait():
    dims = [TraitDimension("patience", "Patience", "quits immediately if confused", "waits indefinitely", "normal", 5.0, 2.0, "space2")]
    agent = AgentProfile("a1", "Socializer", {"patience": 9.2})
    text = agent.to_behavioral_constraints(dims)
    assert "waits indefinitely" in text

def test_behavioral_constraints_includes_anti_rationalization():
    dims = [TraitDimension("ludo_familiarity", "Ludo XP", "never played Ludo", "expert", "normal", 5.0, 2.0, "space2")]
    agent = AgentProfile("a1", "Arch", {"ludo_familiarity": 1.0})
    text = agent.to_behavioral_constraints(dims)
    # Must include the anti-rationalization rules (Chinese or English)
    assert "放弃" in text or "abandon" in text.lower()
    assert "礼貌" in text or "polite" in text.lower()

def test_behavioral_constraints_covers_all_traits():
    dims = [
        TraitDimension("patience", "Patience", "low patience", "high patience", "normal", 5.0, 2.0, "space2"),
        TraitDimension("social_motivation", "Social", "solo player", "social-first", "normal", 6.0, 2.0, "space2"),
    ]
    agent = AgentProfile("a1", "Arch", {"patience": 7.0, "social_motivation": 3.0})
    text = agent.to_behavioral_constraints(dims)
    assert "high patience" in text
    assert "solo player" in text

def test_behavioral_constraints_mid_trait():
    dims = [TraitDimension("patience", "Patience level", "quits fast", "very patient", "normal", 5.0, 2.0, "space2")]
    agent = AgentProfile("a1", "Arch", {"patience": 5.0})
    text = agent.to_behavioral_constraints(dims)
    # Mid-range should reference the dimension description, not raw snake_case name
    assert "patience level" in text.lower()
    assert "patience" in text  # dim.name=5.0 still appears in the value part


def _mock_researcher_response() -> str:
    return json.dumps({
        "population_label": "Arabic Ludo App Users",
        "product_context": "Ludo mobile app for Arabic families",
        "trait_dimensions": [
            {
                "name": "ludo_familiarity",
                "description": "Prior Ludo experience",
                "low_label": "never played Ludo, confused by rules",
                "high_label": "expert player who knows all strategies",
                "distribution": "bimodal",
                "mean": 5.0,
                "std": 2.5,
                "source": "space2",
            },
            {
                "name": "social_motivation",
                "description": "Motivation to play with others vs solo",
                "low_label": "prefers solo play, ignores social features",
                "high_label": "only plays with friends and family",
                "distribution": "right_skewed",
                "mean": 7.0,
                "std": 2.0,
                "source": "space3",
            },
        ],
        "archetypes": [
            {
                "name": "Family Socializer",
                "frequency": 0.50,
                "description": "Plays with family, social-first motivation",
                "trait_constraints": {"social_motivation": [6.0, 10.0], "ludo_familiarity": [2.0, 9.0]},
            },
            {
                "name": "Competitive Grinder",
                "frequency": 0.30,
                "description": "Plays to win, high familiarity",
                "trait_constraints": {"social_motivation": [2.0, 6.0], "ludo_familiarity": [6.0, 10.0]},
            },
            {
                "name": "Casual Dabbler",
                "frequency": 0.20,
                "description": "First-time user, low commitment",
                "trait_constraints": {"social_motivation": [3.0, 7.0], "ludo_familiarity": [0.0, 4.0]},
            },
        ],
        "research_notes": "Ludo is dominant in Arabic family gaming.",
    })

def test_researcher_returns_persona_structure():
    with patch("mcv.core._llm_call", return_value=(_mock_researcher_response(), {})):
        researcher = PopulationResearcher(api_key="test")
        result = researcher.research("A Ludo mobile app for Arabic families")
    assert isinstance(result, PersonaStructure)
    assert result.population_label == "Arabic Ludo App Users"
    assert len(result.trait_dimensions) == 2
    assert len(result.archetypes) == 3

def test_researcher_trait_dimensions_parsed():
    with patch("mcv.core._llm_call", return_value=(_mock_researcher_response(), {})):
        researcher = PopulationResearcher(api_key="test")
        result = researcher.research("A Ludo app")
    dim = result.trait_dimensions[0]
    assert dim.name == "ludo_familiarity"
    assert dim.distribution == "bimodal"
    assert dim.source == "space2"

def test_researcher_archetypes_parsed():
    with patch("mcv.core._llm_call", return_value=(_mock_researcher_response(), {})):
        researcher = PopulationResearcher(api_key="test")
        result = researcher.research("A Ludo app")
    arch = result.archetypes[0]
    assert arch.name == "Family Socializer"
    assert arch.frequency == 0.50
    assert arch.trait_constraints["social_motivation"] == (6.0, 10.0)

def test_researcher_frequencies_sum_to_1():
    with patch("mcv.core._llm_call", return_value=(_mock_researcher_response(), {})):
        researcher = PopulationResearcher(api_key="test")
        result = researcher.research("A Ludo app")
    total = sum(a.frequency for a in result.archetypes)
    assert abs(total - 1.0) < 0.01

def test_researcher_fallback_on_invalid_json():
    with patch("mcv.core._llm_call", return_value=("not valid json at all", {})):
        researcher = PopulationResearcher(api_key="test")
        result = researcher.research("some product")
    assert isinstance(result, PersonaStructure)
    assert result.research_notes == "fallback — LLM parsing failed"
    assert len(result.archetypes) == 2


def test_prepare_with_pool_sets_pool():
    from mcv.user_simulator import UserSimulator
    from mcv.domain_configs import AppDomainConfig
    from mcv.schema_extractor import EvaluationMetric
    sim = UserSimulator("test user", AppDomainConfig, api_key="test")
    agents = PersonaPool(_make_structure()).generate(6)
    metrics = [EvaluationMetric("day1_return", "bool", "会回来吗？")]
    sim.prepare_with_pool(product="test product", pool=agents, locked_metrics=metrics)
    assert sim._agent_pool == agents
    assert sim._metrics == metrics

def test_simulate_with_pool_uses_behavioral_constraints():
    """Each session prompt should contain anti-rationalization text from AgentProfile."""
    from mcv.user_simulator import UserSimulator
    from mcv.domain_configs import AppDomainConfig
    from mcv.schema_extractor import EvaluationMetric
    from unittest.mock import patch
    sim = UserSimulator("test user", AppDomainConfig, api_key="test")
    agents = PersonaPool(_make_structure()).generate(6)
    metrics = [EvaluationMetric("day1_return", "bool", "会回来吗？")]
    sim.prepare_with_pool(product="test product", pool=agents, locked_metrics=metrics)
    captured_prompts = []
    def fake_llm(prompt, api_key, **kwargs):
        captured_prompts.append(prompt)
        return "day1_return: yes", {}
    with patch("mcv.core._llm_call", side_effect=fake_llm):
        sim.simulate(n_runs=6)
    assert len(captured_prompts) == 6
    # All prompts should contain anti-rationalization text
    for p in captured_prompts:
        assert "放弃" in p or "abandon" in p.lower()

def test_simulate_with_pool_cycles_when_n_exceeds_pool():
    """If n_runs > len(pool), cycles through pool."""
    from mcv.user_simulator import UserSimulator
    from mcv.domain_configs import AppDomainConfig
    from mcv.schema_extractor import EvaluationMetric
    from unittest.mock import patch
    sim = UserSimulator("test user", AppDomainConfig, api_key="test")
    agents = PersonaPool(_make_structure()).generate(6)
    metrics = [EvaluationMetric("day1_return", "bool", "会回来吗？")]
    sim.prepare_with_pool(product="test product", pool=agents, locked_metrics=metrics)
    with patch("mcv.core._llm_call", return_value=("day1_return: yes", {})):
        sim.simulate(n_runs=10)
    assert len(sim._session_results) == 10
