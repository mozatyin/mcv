"""Persona generation and caching."""
from __future__ import annotations

from pathlib import Path
from mcv.core import Persona


def load_or_generate(
    state_dir: Path,
    prd_text: str,
    archetype: str,
    target_market: str,
    api_key: str,
    n: int = 5,
) -> list[Persona]:
    """Load personas from state_dir/personas.json, or generate fresh and cache."""
    raise NotImplementedError
