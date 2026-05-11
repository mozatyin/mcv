from user_soul.backend import LLMBackend


def test_protocol_has_text_method():
    assert hasattr(LLMBackend, 'text')


def test_protocol_has_vision_method():
    assert hasattr(LLMBackend, 'vision')


class _StubBackend:
    def text(self, prompt, *, max_tokens=512, temperature=0.0,
             model_tier="fast"):
        return "stub"

    def vision(self, prompt, images, *, max_tokens=512,
               temperature=0.0, model_tier="smart"):
        return "stub"


def test_stub_satisfies_protocol():
    backend: LLMBackend = _StubBackend()
    assert isinstance(backend, LLMBackend)
