"""Domain configurations for UserSimulator — controls session 'world'."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DomainConfig:
    session_framing: str
    emotional_states: list[str]
    triggers: list[str]
    time_options: list[str]
    user_roles: dict[str, list[int]]


GameDomainConfig = DomainConfig(
    session_framing="你开始了一局游戏",
    emotional_states=["competitive", "casual", "tilted", "bored", "hyped"],
    triggers=["want_to_rank_up", "friend_challenged_me", "kill_time", "daily_login", "revenge_match"],
    time_options=["morning_commute", "lunch_break", "evening", "late_night"],
    user_roles={
        "Newcomer": [1, 3],
        "Casual":   [3, 14],
        "Grinder":  [14, 30],
        "Veteran":  [30],
    },
)

AppDomainConfig = DomainConfig(
    session_framing="你打开了这个 app",
    emotional_states=["stressed", "calm", "bored", "excited", "sad", "anxious"],
    triggers=["habit", "work_stress", "relationship_tension", "boredom", "notification", "curiosity"],
    time_options=["morning_commute", "lunch_break", "evening_wind_down", "night"],
    user_roles={
        "Explorer":  [1, 3],
        "Skeptic":   [3, 7],
        "Habituer":  [14, 30],
        "Advocate":  [30],
    },
)

WebDomainConfig = DomainConfig(
    session_framing="你在刷新闻",
    emotional_states=["curious", "bored", "anxious", "relaxed", "rushed"],
    triggers=["morning_routine", "notification", "kill_time", "topic_interest", "breaking_news"],
    time_options=["morning_commute", "lunch_break", "evening", "late_night"],
    user_roles={
        "Casual":      [1, 7],
        "Regular":     [7, 30],
        "PowerReader": [30],
    },
)
