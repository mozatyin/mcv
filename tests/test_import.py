import dataclasses
import inspect


def test_import_symbols():
    from mcv import PersonaDecider, DecisionResult, Persona, load_or_generate
    assert PersonaDecider is not None, "PersonaDecider not importable"
    assert DecisionResult is not None, "DecisionResult not importable"
    assert Persona is not None, "Persona not importable"
    assert load_or_generate is not None, "load_or_generate not importable"


def test_persona_fields():
    from mcv import Persona
    fields = {f.name for f in dataclasses.fields(Persona)}
    assert fields == {"id", "name", "description", "cohort", "motivations", "pain_points"}, \
        f"Persona fields mismatch: {fields}"


def test_decision_result_fields():
    from mcv import DecisionResult
    fields = {f.name for f in dataclasses.fields(DecisionResult)}
    assert {"value", "confidence", "distribution", "mode", "tokens_used", "raw_votes"} <= fields, \
        f"DecisionResult missing fields: {fields}"


def test_persona_decider_signature():
    from mcv import PersonaDecider
    sig = inspect.signature(PersonaDecider.__init__)
    params = set(sig.parameters.keys()) - {"self"}
    assert "personas" in params, "PersonaDecider missing 'personas' param"
    assert "api_key" in params, "PersonaDecider missing 'api_key' param"
    assert "mode" in params, "PersonaDecider missing 'mode' param"
