"""MCVClient — unified facade for MCV decision and simulation capabilities."""
from __future__ import annotations

from pathlib import Path

from mcv.core import Persona, DecisionResult, PersonaDecider
from mcv.personas import load_or_generate
from mcv.user_simulator import UserSimulator
from mcv.domain_configs import DomainConfig, build_domain_config
from mcv.report import SimulationReport, CompareReport, FeatureAAR, CoherenceReport
from mcv.schema_extractor import EvaluationMetric
from mcv.population import PopulationResearcher


_AARRR_VOTE_PROMPT = """You are scoring product features for a mobile app from the perspective of different user archetypes.

Product: {product_description}

User archetypes (each with their background story):
{archetypes_block}

Features to score:
{features_block}

For each feature, score its impact on each AARRR dimension (0.0 = no impact, 1.0 = primary driver)
from each archetype's perspective. Then compute the mean across archetypes.

Return ONLY valid JSON (no markdown):
[
  {{
    "feature_id": "...",
    "archetype_votes": {{
      "ArchetypeName": {{"acquisition": 0.0, "activation": 0.0, "retention": 0.0, "revenue": 0.0, "referral": 0.0}},
      ...
    }},
    "mean": {{"acquisition": 0.0, "activation": 0.0, "retention": 0.0, "revenue": 0.0, "referral": 0.0}}
  }},
  ...
]"""


class MCVClient:
    """Single entry point for all MCV capabilities.

    Routes to PersonaDecider (fast decisions) or UserSimulator (behavioral simulation)
    based on method called. domain_config=None triggers auto-inference.

    Usage:
        client = MCVClient(api_key=os.environ["ANTHROPIC_API_KEY"])
        report = client.simulate(product=prd, user_type="18岁玩家", goal="Day-1留存?")
        diff   = client.compare(prd_v1, prd_v2, user_type="玩家", goal="哪版更好?")
        result = client.decide("Kano?", options=["Must-Have", "Delighter"], context=prd)
    """

    def __init__(
        self,
        api_key: str,
        mode: str = "fast",
        personas: list[Persona] | None = None,
    ):
        self._api_key = api_key
        self._mode = mode
        self._personas = personas

    def simulate(
        self,
        product: str,
        user_type: str,
        goal: str,
        domain_config: DomainConfig | None = None,
        n_runs: int = 60,
        locked_metrics: list[EvaluationMetric] | None = None,
    ) -> SimulationReport:
        """Run N behavioral sessions and return aggregated SimulationReport.

        When domain_config is None, automatically infers it via build_domain_config().
        """
        cfg = domain_config or build_domain_config(product, self._api_key)
        sim = UserSimulator(user_type, cfg, api_key=self._api_key)
        sim.prepare(product=product, goal=goal, locked_metrics=locked_metrics)
        return sim.simulate(n_runs=n_runs).report()

    def compare(
        self,
        product_a: str,
        product_b: str,
        user_type: str,
        goal: str,
        label_a: str = "v_a",
        label_b: str = "v_b",
        domain_config: DomainConfig | None = None,
        n_runs: int = 30,
        locked_metrics: list[EvaluationMetric] | None = None,
    ) -> CompareReport:
        """Compare two product variants using shared scenario seeds.

        When domain_config is None, it is inferred from product_a's description.
        Both variants run with the same inferred or provided config.
        """
        cfg = domain_config or build_domain_config(product_a, self._api_key)
        sim = UserSimulator(user_type, cfg, api_key=self._api_key)
        return sim.compare(
            product_a, product_b,
            label_a=label_a, label_b=label_b,
            n_runs=n_runs, locked_metrics=locked_metrics, goal=goal,
        )

    def decide(
        self,
        question: str,
        options: list[str],
        context: str,
        product: str | None = None,
        state_dir: Path | None = None,
    ) -> DecisionResult:
        """Classify using PersonaDecider.

        Uses pre-loaded personas if provided at init.
        Otherwise auto-generates from product description (requires product= argument).
        """
        personas = self._personas
        if not personas:
            if product is None:
                raise ValueError("provide personas= at init or product= to auto-generate")
            personas = load_or_generate(
                state_dir=state_dir or Path(".mcv_state"),
                prd_text=product,
                archetype="generic app",
                target_market="app users",
                api_key=self._api_key,
                n=3,
            )
        decider = PersonaDecider(personas, api_key=self._api_key, mode=self._mode)
        return decider.classify(question=question, options=options, context=context)

    def research_aarrr(
        self,
        product_description: str,
        features: list[dict],
        objectives: dict | None = None,
    ) -> list:
        """Score features on AARRR dimensions using population-grounded archetype voting.

        Two LLM calls: (1) PopulationResearcher to build persona archetypes,
        (2) batch AARRR vote for all features across all archetypes.

        Args:
            product_description: PRD or one-paragraph product description.
            features: [{"id": str, "name": str, "description": str}, ...]
            objectives: ignored (reserved for future weighting)

        Returns:
            list[FeatureAAR] in same order as input features.
            Empty list if features is empty.
        """
        import statistics as _stats
        import json as _json
        import re as _re
        from mcv import core as _core

        if not features:
            return []

        # Step 1: Research population (1 Sonnet call)
        researcher = PopulationResearcher(self._api_key)
        try:
            structure = researcher.research(product_description)
        except Exception:
            structure = researcher._fallback(product_description)

        archetypes = structure.archetypes

        # Step 2: Build batch prompt
        archetypes_block = "\n".join(
            f"- {a.name}: {a.background_story or a.description}"
            for a in archetypes
        )
        features_block = "\n".join(
            f"- id={f['id']}: {f['name']} — {f.get('description', '')}"
            for f in features
        )
        prompt = _AARRR_VOTE_PROMPT.format(
            product_description=product_description[:800],
            archetypes_block=archetypes_block,
            features_block=features_block,
        )

        # Step 3: Single LLM vote call (1 Sonnet call)
        raw, _ = _core._llm_call(prompt, self._api_key, max_tokens=3000)

        # Parse
        m = _re.search(r'\[.*\]', raw, _re.DOTALL)
        items = []
        if m:
            try:
                items = _json.loads(m.group())
            except (ValueError, _json.JSONDecodeError):
                pass

        items_by_id = {
            item["feature_id"]: item
            for item in items
            if isinstance(item, dict) and "feature_id" in item
        }

        _DIMS = ("acquisition", "activation", "retention", "revenue", "referral")
        scored = []

        for f in features:
            fid = f["id"]
            item = items_by_id.get(fid)
            if item:
                mean = item.get("mean", {})
                arch_votes = item.get("archetype_votes", {})
                # Confidence = 1 - mean stdev across archetypes per dimension
                stdevs = []
                for dim in _DIMS:
                    vals = [v.get(dim, 0.5) for v in arch_votes.values() if isinstance(v, dict)]
                    if len(vals) >= 2:
                        stdevs.append(_stats.stdev(vals))
                confidence = round(max(0.0, 1.0 - (sum(stdevs) / len(stdevs) if stdevs else 0.0)), 4)
                scored.append(FeatureAAR(
                    feature_id=fid,
                    acquisition=round(min(1.0, max(0.0, float(mean.get("acquisition", 0.5)))), 4),
                    activation=round(min(1.0, max(0.0, float(mean.get("activation", 0.5)))), 4),
                    retention=round(min(1.0, max(0.0, float(mean.get("retention", 0.5)))), 4),
                    revenue=round(min(1.0, max(0.0, float(mean.get("revenue", 0.2)))), 4),
                    referral=round(min(1.0, max(0.0, float(mean.get("referral", 0.2)))), 4),
                    confidence=confidence,
                    archetype_votes=arch_votes,
                ))
            else:
                # Fallback: neutral scores
                scored.append(FeatureAAR(
                    feature_id=fid,
                    acquisition=0.5, activation=0.5, retention=0.5,
                    revenue=0.2, referral=0.2,
                    confidence=0.0,
                    archetype_votes={},
                ))
        return scored
