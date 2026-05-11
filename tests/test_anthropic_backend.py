from unittest.mock import MagicMock
from user_soul.backends.anthropic import AnthropicBackend
from user_soul.backend import LLMBackend


def test_satisfies_protocol():
    backend = AnthropicBackend(api_key="sk-test")
    assert isinstance(backend, LLMBackend)


def test_text_calls_anthropic(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="hello")],
        usage=MagicMock(input_tokens=10, output_tokens=5),
    )
    monkeypatch.setattr("user_soul.backends.anthropic.anthropic.Anthropic",
                        lambda **kw: mock_client)
    backend = AnthropicBackend(api_key="sk-test")
    result = backend.text("prompt", model_tier="fast")
    assert result == "hello"
    mock_client.messages.create.assert_called_once()


def test_model_tier_fast_uses_haiku():
    backend = AnthropicBackend(api_key="sk-test")
    assert "haiku" in backend._resolve_model("fast")


def test_model_tier_smart_uses_sonnet():
    backend = AnthropicBackend(api_key="sk-test")
    assert "sonnet" in backend._resolve_model("smart")


def test_vision_builds_image_content(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="looks good")],
        usage=MagicMock(input_tokens=100, output_tokens=10),
    )
    monkeypatch.setattr("user_soul.backends.anthropic.anthropic.Anthropic",
                        lambda **kw: mock_client)
    backend = AnthropicBackend(api_key="sk-test")
    result = backend.vision("describe", [b"\x89PNG fake"])
    assert result == "looks good"
    call_kwargs = mock_client.messages.create.call_args[1]
    messages = call_kwargs["messages"]
    content = messages[0]["content"]
    assert any(c["type"] == "image" for c in content)
