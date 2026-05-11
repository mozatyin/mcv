from user_soul import UserSoulClient, LLMBackend
from user_soul.models import (
    ResearchReport, DesignReviewReport, ModuleUATReport, LaunchReport,
    AgentProfile, EvaluationMetric,
)


class _FakeBackend:
    def text(self, prompt, **kw):
        if "AARRR" in prompt or "scoring" in prompt.lower():
            import json
            return json.dumps([{
                "feature_id": "f1",
                "archetype_votes": {"Casual": {"acquisition": 0.8, "activation": 0.7,
                                               "retention": 0.6, "revenue": 0.3, "referral": 0.2}},
                "mean": {"acquisition": 0.8, "activation": 0.7, "retention": 0.6,
                          "revenue": 0.3, "referral": 0.2},
            }])
        if "proceed" in prompt.lower() or "前往" in prompt:
            return "proceed: yes\nreason: clear\nfogg_issue: none"
        if "population" in prompt.lower() or "behavioral researcher" in prompt.lower():
            return ('{"population_label":"gamers","product_context":"chess",'
                    '"trait_dimensions":[{"name":"skill","description":"d","low_label":"l",'
                    '"high_label":"h","distribution":"normal","mean":5,"std":2,"source":"space1"}],'
                    '"archetypes":[{"name":"Casual","frequency":1.0,"description":"casual",'
                    '"background_story":"小明","trait_constraints":{}}],"research_notes":"insight"}')
        return "他打开了app。\nday_1_return_intent: yes\nfriction_points: none\n"

    def vision(self, prompt, images, **kw):
        import json
        if len(images) >= 2:
            return json.dumps({
                "dimensions": {"视觉精致度": {"winner": "A", "reason": "better"}},
                "overall_winner": "A",
                "overall_reason": "ours wins",
            })
        return json.dumps({"issues": [], "overall_score": "professional", "suggestions": []})


def test_client_satisfies_protocol():
    backend = _FakeBackend()
    client = UserSoulClient(backend)
    assert client is not None


def test_research():
    client = UserSoulClient(_FakeBackend())
    report = client.research("chess game", [{"id": "f1", "name": "matchmaking"}])
    assert isinstance(report, ResearchReport)


def test_review():
    client = UserSoulClient(_FakeBackend())
    screens = [
        {"screen_id": "home", "navigates_to": ["game"]},
        {"screen_id": "game", "navigates_to": []},
    ]
    report = client.review("chess game", screens, ["home", "game"])
    assert isinstance(report, DesignReviewReport)


def test_verify():
    client = UserSoulClient(_FakeBackend())
    metrics = [EvaluationMetric("day_1_return_intent", "bool", "Return?")]
    report = client.verify("chess game", metrics=metrics)
    assert isinstance(report, ModuleUATReport)


def test_launch():
    client = UserSoulClient(_FakeBackend())
    metrics = [EvaluationMetric("day_1_return_intent", "bool", "Return?")]
    report = client.launch("chess game", [b"our_png"], [("comp", b"their_png")],
                           metrics=metrics)
    assert isinstance(report, LaunchReport)


def test_create_persona_pool():
    client = UserSoulClient(_FakeBackend())
    pool = client.create_persona_pool("chess game", n=5)
    assert len(pool) == 5
    assert all(isinstance(a, AgentProfile) for a in pool)
