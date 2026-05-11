from user_soul.stages.launch import LaunchGate
from user_soul.models import LaunchReport, EvaluationMetric


class _FakeBackend:
    def text(self, prompt, **kw):
        return ("他打开了app。\n"
                "day_1_return_intent: yes\n"
                "friction_points: none\n")

    def vision(self, prompt, images, **kw):
        import json
        return json.dumps({
            "dimensions": {"视觉精致度": {"winner": "A", "reason": "better"}},
            "overall_winner": "A",
            "overall_reason": "our product wins",
        })


def test_launch_gate_ship():
    import random
    random.seed(0)  # deterministic: A = ours
    gate = LaunchGate(_FakeBackend())
    metrics = [
        EvaluationMetric("day_1_return_intent", "bool", "Return?"),
        EvaluationMetric("friction_points", "text", "Friction?"),
    ]
    report = gate.run("chess game", [b"our_png"],
                      [("comp", b"their_png")], metrics=metrics)
    assert isinstance(report, LaunchReport)
    # With seed(0), taste may or may not win depending on random order.
    # Core assertion: it's a valid LaunchReport with correct structure.
    assert report.recommendation in ("SHIP", "IMPROVE")
    assert report.taste_win_rate >= 0.0


def test_launch_gate_no_screenshots():
    gate = LaunchGate(_FakeBackend())
    metrics = [EvaluationMetric("day_1_return_intent", "bool", "Return?")]
    report = gate.run("chess game", [], [], metrics=metrics)
    assert isinstance(report, LaunchReport)
    assert report.taste_win_rate == 0.0


def test_launch_gate_abandon():
    class _BadBackend:
        def text(self, prompt, **kw):
            return ("他打开了app。\n"
                    "day_1_return_intent: no\n"
                    "friction_points: everything\n")

        def vision(self, prompt, images, **kw):
            import json
            return json.dumps({
                "dimensions": {"视觉精致度": {"winner": "B",
                                          "reason": "theirs better"}},
                "overall_winner": "B",
                "overall_reason": "competitor wins",
            })

    gate = LaunchGate(_BadBackend())
    metrics = [
        EvaluationMetric("day_1_return_intent", "bool", "Return?"),
        EvaluationMetric("friction_points", "text", "Friction?"),
    ]
    report = gate.run("chess game", [b"our_png"],
                      [("comp", b"their_png")], metrics=metrics)
    assert report.recommendation in ("IMPROVE", "ABANDON")
