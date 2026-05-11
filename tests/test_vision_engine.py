import json
from user_soul.engines.vision import VisionEngine
from user_soul.models import PairwiseResult, ReviewResult


class _FakeVisionBackend:
    def text(self, prompt, **kw):
        return ""
    def vision(self, prompt, images, **kw):
        return json.dumps({
            "dimensions": {
                "视觉精致度": {"winner": "A", "reason": "cleaner layout"},
                "色彩和谐": {"winner": "B", "reason": "better contrast"},
                "信息层级": {"winner": "tie", "reason": "similar"},
                "专业感": {"winner": "A", "reason": "more polished"},
            },
            "overall_winner": "A",
            "overall_reason": "Better polish despite weaker colors",
        })


def test_pairwise_compare_returns_result():
    engine = VisionEngine(_FakeVisionBackend())
    import random; random.seed(42)
    result = engine.pairwise_compare(b"png_ours", b"png_theirs")
    assert isinstance(result, PairwiseResult)
    assert result.winner in ("ours", "theirs", "tie")


def test_pairwise_dimension_results():
    import random; random.seed(42)
    engine = VisionEngine(_FakeVisionBackend())
    result = engine.pairwise_compare(b"png_ours", b"png_theirs")
    assert len(result.dimension_results) == 4


def test_pairwise_confidence():
    import random; random.seed(42)
    engine = VisionEngine(_FakeVisionBackend())
    result = engine.pairwise_compare(b"png_ours", b"png_theirs")
    assert 0.0 <= result.confidence <= 1.0


def test_batch_compare():
    engine = VisionEngine(_FakeVisionBackend())
    results = engine.batch_compare(
        b"png_ours",
        [("comp_a", b"png_a"), ("comp_b", b"png_b")])
    assert len(results) == 2
    assert all(isinstance(r, PairwiseResult) for r in results)


class _FakeReviewBackend:
    def text(self, prompt, **kw):
        return ""
    def vision(self, prompt, images, **kw):
        return json.dumps({
            "issues": [
                {"severity": "P1", "dimension": "布局", "description": "text overflow"}
            ],
            "overall_score": "acceptable",
            "suggestions": ["Fix text overflow in header"],
        })


def test_screenshot_review():
    engine = VisionEngine(_FakeReviewBackend())
    result = engine.screenshot_review(b"png_data", context="chess game")
    assert isinstance(result, ReviewResult)
    assert result.overall_score == "acceptable"
    assert len(result.issues) == 1
