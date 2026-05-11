from user_soul.stages.design_review import DesignReview
from user_soul.models import DesignReviewReport


class _FakeBackend:
    def text(self, prompt, **kw):
        if "proceed" in prompt.lower() or "前往" in prompt:
            return "proceed: yes\nreason: clear\nfogg_issue: none"
        return ('{"population_label":"gamers","product_context":"chess",'
                '"trait_dimensions":[{"name":"skill","description":"d","low_label":"l",'
                '"high_label":"h","distribution":"normal","mean":5,"std":2,"source":"space1"}],'
                '"archetypes":[{"name":"Casual","frequency":1.0,"description":"casual",'
                '"background_story":"小明","trait_constraints":{}}],"research_notes":""}')

    def vision(self, prompt, images, **kw):
        import json
        if len(images) == 2:
            return json.dumps({"dimensions": {}, "overall_winner": "A",
                               "overall_reason": "better"})
        return json.dumps({"issues": [], "overall_score": "professional",
                           "suggestions": []})


def test_design_review_basic():
    review = DesignReview(_FakeBackend())
    screens = [
        {"screen_id": "home", "navigates_to": ["game"]},
        {"screen_id": "game", "navigates_to": ["result"]},
        {"screen_id": "result", "navigates_to": []},
    ]
    report = review.run("chess game", screens, ["home", "game", "result"])
    assert isinstance(report, DesignReviewReport)
    assert report.journey is not None
    assert isinstance(report.passes_gate, bool)


def test_design_review_with_screenshots():
    review = DesignReview(_FakeBackend())
    screens = [
        {"screen_id": "home", "navigates_to": ["game"]},
        {"screen_id": "game", "navigates_to": []},
    ]
    report = review.run("chess", screens, ["home", "game"],
                        wireframe_screenshots=[b"png1"],
                        competitor_screenshots=[("comp", b"png2")])
    assert len(report.layout_reviews) == 1
    assert len(report.competitor_gaps) == 1
