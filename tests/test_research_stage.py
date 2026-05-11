from user_soul.stages.research import ResearchPanel
from user_soul.models import ResearchReport


class _FakeBackend:
    def text(self, prompt, **kw):
        if "AARRR" in prompt or "scoring" in prompt.lower():
            import json
            return json.dumps([{
                "feature_id": "f1",
                "archetype_votes": {"Casual": {"acquisition": 0.8, "activation": 0.7,
                                               "retention": 0.6, "revenue": 0.3, "referral": 0.2}},
                "mean": {"acquisition": 0.8, "activation": 0.7,
                         "retention": 0.6, "revenue": 0.3, "referral": 0.2},
            }])
        return ('{"population_label":"gamers","product_context":"chess",'
                '"trait_dimensions":[{"name":"skill","description":"d","low_label":"l",'
                '"high_label":"h","distribution":"normal","mean":5,"std":2,"source":"space1"}],'
                '"archetypes":[{"name":"Casual","frequency":0.6,"description":"casual",'
                '"background_story":"小明","trait_constraints":{"skill":[1,5]}},'
                '{"name":"Hardcore","frequency":0.4,"description":"serious",'
                '"background_story":"老王","trait_constraints":{"skill":[6,10]}}],'
                '"research_notes":"key insight"}')

    def vision(self, prompt, images, **kw):
        return '{"dimensions":{},"overall_winner":"A","overall_reason":"better"}'


def test_research_panel_basic():
    panel = ResearchPanel(_FakeBackend())
    report = panel.run("chess game", [{"id": "f1", "name": "matchmaking"}])
    assert isinstance(report, ResearchReport)
    assert report.persona_structure is not None
    assert len(report.feature_priorities) == 1


def test_research_panel_with_screenshots():
    panel = ResearchPanel(_FakeBackend())
    report = panel.run("chess game", [],
                       our_screenshot=b"png",
                       competitor_screenshots=[("comp", b"png2")])
    assert len(report.visual_preferences) == 1
