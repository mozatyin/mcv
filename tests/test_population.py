from mcv.population import TraitDimension, Archetype, PersonaStructure, AgentProfile

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

def test_agent_profile_fields():
    ap = AgentProfile(
        agent_id="agent_001",
        archetype_name="Family Socializer",
        trait_vector={"social_motivation": 8.3, "ludo_familiarity": 5.1, "patience": 6.2},
    )
    assert ap.agent_id == "agent_001"
    assert ap.trait_vector["social_motivation"] == 8.3
