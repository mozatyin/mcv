from user_soul.stages.module_uat import ModuleUAT
from user_soul.models import ModuleUATReport, EvaluationMetric


class _FakeBackend:
    def text(self, prompt, **kw):
        if "evaluation" in prompt.lower() or "metric" in prompt.lower():
            return '[{"name":"retention","type":"bool","question":"Return?"}]'
        return "他打开了app。\nretention: yes\n"

    def vision(self, prompt, images, **kw):
        return '{"issues":[],"overall_score":"professional","suggestions":[]}'


def test_module_uat_basic():
    uat = ModuleUAT(_FakeBackend())
    metrics = [EvaluationMetric("retention", "bool", "Return?")]
    report = uat.run("chess game", metrics=metrics)
    assert isinstance(report, ModuleUATReport)
    assert report.behavior is not None
    assert report.passes_gate is True


def test_module_uat_with_p0_fails_gate():
    class _P0Backend:
        def text(self, prompt, **kw):
            return "他打开了app。\nretention: yes\n"

        def vision(self, prompt, images, **kw):
            return ('{"issues":[{"severity":"P0","dimension":"布局",'
                    '"description":"crash"}],"overall_score":"amateur",'
                    '"suggestions":[]}')

    uat = ModuleUAT(_P0Backend())
    metrics = [EvaluationMetric("retention", "bool", "Return?")]
    report = uat.run("chess game", metrics=metrics, html_screenshots=[b"png"])
    assert report.passes_gate is False
    assert len(report.visual_issues) == 1
