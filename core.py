"""Monte Carlo Voter — core types and PersonaDecider."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Persona:
    id: str
    name: str
    description: str
    cohort: str             # e.g. "College students 18-24, casual gamers"
    motivations: list[str] = field(default_factory=list)
    pain_points: list[str] = field(default_factory=list)


@dataclass
class DecisionResult:
    value: Any                          # str for classify, float for score, bool for validate
    confidence: float                   # 0.0–1.0, fraction of personas that agreed
    distribution: dict[str, float]      # option → fraction of votes
    mode: str                           # "fast" | "validated"
    tokens_used: int = 0
    raw_votes: list[Any] = field(default_factory=list)   # per-persona values


def _model_name(api_key: str) -> str:
    """Pick model. OpenRouter keys use openrouter base URL."""
    return "claude-sonnet-4-6"


def _llm_call(prompt: str, api_key: str, max_tokens: int = 512) -> tuple[str, int]:
    """Single LLM call → (response_text, tokens_used)."""
    import anthropic
    kwargs: dict = dict(
        model=_model_name(api_key),
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if api_key.startswith("sk-or-"):
        client = anthropic.Anthropic(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    else:
        client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(**kwargs)
    text = resp.content[0].text if resp.content else ""
    tokens = (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
    return text, tokens


def _safe_json(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


class PersonaDecider:
    def __init__(self, personas: list[Persona], api_key: str, mode: str = "fast"):
        self.personas = personas
        self.api_key = api_key
        self.mode = mode

    def classify(self, question: str, options: list[str], context: str, batch: list[dict[str, Any]] | None = None) -> DecisionResult | list[DecisionResult]:
        raise NotImplementedError

    def score(self, question: str, lo: float, hi: float, context: str, batch: list[dict[str, Any]] | None = None) -> DecisionResult | list[DecisionResult]:
        raise NotImplementedError

    def validate(self, assertion: str, context: str) -> DecisionResult:
        if self.mode == "validated":
            raise NotImplementedError("validated mode implemented in Task 5")
        return self._fast_validate(assertion, context)

    def _fast_validate(self, assertion: str, context: str) -> DecisionResult:
        persona = self.personas[0]
        prompt = (
            f"You are evaluating a product feature on behalf of this user persona:\n"
            f"Name: {persona.name}\n"
            f"Cohort: {persona.cohort}\n"
            f"Motivations: {', '.join(persona.motivations)}\n"
            f"Pain points: {', '.join(persona.pain_points)}\n\n"
            f"Context: {context}\n\n"
            f"Assertion: {assertion}\n\n"
            f'Reply with JSON only: {{"result": true|false, "reasoning": "one sentence"}}'
        )
        raw, tokens = _llm_call(prompt, self.api_key)
        data = _safe_json(raw)
        value = bool(data.get("result", False))
        return DecisionResult(
            value=value,
            confidence=1.0,
            distribution={"true": 1.0 if value else 0.0, "false": 0.0 if value else 1.0},
            mode=self.mode,
            tokens_used=tokens,
            raw_votes=[value],
        )
