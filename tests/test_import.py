def test_import():
    import sys
    sys.path.insert(0, '/Users/michael/mcv')
    from mcv import PersonaDecider, DecisionResult, Persona, load_or_generate
    assert PersonaDecider is not None
    assert DecisionResult is not None
    assert Persona is not None
    assert load_or_generate is not None
