import sys
sys.path.insert(0, '/Users/michael/mcv')

from mcv.domain_configs import DomainConfig, GameDomainConfig, AppDomainConfig, WebDomainConfig
from mcv.scenarios import random_context_for_domain, ScenarioContext


def test_game_domain_config_fields():
    assert GameDomainConfig.session_framing == "你开始了一局游戏"
    assert "competitive" in GameDomainConfig.emotional_states
    assert "Newcomer" in GameDomainConfig.user_roles
    assert "want_to_rank_up" in GameDomainConfig.triggers


def test_app_domain_config_fields():
    assert "stressed" in AppDomainConfig.emotional_states
    assert "Explorer" in AppDomainConfig.user_roles


def test_web_domain_config_fields():
    assert WebDomainConfig.session_framing == "你在刷新闻"
    assert "curious" in WebDomainConfig.emotional_states


def test_custom_domain_config():
    custom = DomainConfig(
        session_framing="你在体验VR",
        emotional_states=["immersed", "curious"],
        triggers=["boredom", "curiosity"],
        time_options=["evening", "weekend"],
        user_roles={"Newbie": [1, 7]},
    )
    assert custom.session_framing == "你在体验VR"
    assert "Newbie" in custom.user_roles


def test_random_context_for_domain_uses_config_options():
    ctx = random_context_for_domain(role="Newcomer", domain_config=GameDomainConfig)
    assert isinstance(ctx, ScenarioContext)
    assert ctx.emotional_state in GameDomainConfig.emotional_states
    assert ctx.trigger in GameDomainConfig.triggers
    assert ctx.usage_day in GameDomainConfig.user_roles["Newcomer"]


def test_random_context_for_domain_produces_variance():
    contexts = [random_context_for_domain(domain_config=GameDomainConfig) for _ in range(20)]
    unique = {(c.time_of_day, c.emotional_state, c.usage_day, c.trigger) for c in contexts}
    assert len(unique) >= 3
