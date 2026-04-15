"""MCVClient — unified facade for MCV decision and simulation capabilities."""
from __future__ import annotations

from pathlib import Path

from mcv.core import Persona, DecisionResult, PersonaDecider
from mcv.personas import load_or_generate
from mcv.user_simulator import UserSimulator
from mcv.domain_configs import DomainConfig, build_domain_config
from mcv.report import SimulationReport, CompareReport
from mcv.schema_extractor import EvaluationMetric


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
        """Compare two product variants using shared scenario seeds."""
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
