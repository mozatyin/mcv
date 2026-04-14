from mcv.population import TraitDimension, Archetype, PersonaStructure, AgentProfile, PersonaPool

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
