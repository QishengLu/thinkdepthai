"""Three-layer pipeline assembly. Single interface for agent_runner integration.

v2: Dual-timing detection (process + conclusion), dimension-level dedup,
    reasoning extraction, aligned with D×R failure taxonomy.
"""

from __future__ import annotations

import logging

from .config import MiddlewareConfig
from .deficiency_detector import ConclusionDetector, ProcessDetector
from .intent_classifier import IntentClassifier
from .intervention_generator import InterventionGenerator
from .state import InvestigationState

logger = logging.getLogger(__name__)


class MiddlewarePipeline:
    """Dual-timing metacognitive middleware.

    Usage::

        pipeline = MiddlewarePipeline(model, config)

        # After each tool_node execution:
        intervention = pipeline.process_tool_calls(
            tool_calls, round_num, assistant_content
        )
        if intervention:
            messages.append(HumanMessage(content=intervention))

        # Before compress_research:
        intervention = pipeline.check_before_conclusion()
        if intervention:
            messages.append(HumanMessage(content=intervention))
            return "llm_call"  # force continue
    """

    def __init__(self, model=None, config: MiddlewareConfig | None = None):
        self.config = config or MiddlewareConfig()
        self.state = InvestigationState()
        self._pending_sqls: list[tuple[str, int]] = []
        self._intervened_dimensions: set[str] = set()
        self._checked_points: set[int] = set()

        if self.config.enabled and model is not None:
            self.classifier = IntentClassifier(model)
            self.process_detector = ProcessDetector(
                model, self.config.process_deficiencies
            )
            self.conclusion_detector = ConclusionDetector(
                model, self.config.conclusion_deficiencies
            )
            self.generator = InterventionGenerator(model)
            logger.info(
                "Middleware v2 enabled — process: %s, conclusion: %s",
                self.config.process_deficiencies,
                self.config.conclusion_deficiencies,
            )
        else:
            self.classifier = None  # type: ignore[assignment]
            self.process_detector = None  # type: ignore[assignment]
            self.conclusion_detector = None  # type: ignore[assignment]
            self.generator = None  # type: ignore[assignment]

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.classifier is not None

    # ── Public API ──────────────────────────────────────────────────────

    def process_tool_calls(
        self,
        tool_calls: list[dict],
        round_num: int,
        assistant_content: str = "",
    ) -> str | None:
        """Process one round of tool_calls. Returns intervention or None."""
        if not self.enabled:
            return None

        # Always extract reasoning (zero-cost string ops)
        self.state.extract_reasoning(tool_calls, assistant_content)

        # Buffer SQLs for batch classification
        for tc in tool_calls:
            if tc["name"] != "query_parquet_files":
                continue
            sql = tc["args"].get("query", "")
            if sql:
                self._pending_sqls.append((sql, round_num))
                self.state.raw_sqls.append(sql)
                self.state.query_count += 1

        should = self._should_check()
        logger.info(
            "MW query_count=%d, should_check=%s, checked_points=%s, intervention_count=%d",
            self.state.query_count, should, self._checked_points, self.state.intervention_count,
        )
        if not should:
            return None

        return self._run_process_cycle()

    # Fixed metacognitive prompt — covers all D×R failure dimensions
    _CONCLUSION_PROMPT = (
        "[Investigation Advisor — Pre-Conclusion Review]\n\n"
        "Before you finalize your conclusion, pause and reflect on these aspects:\n\n"
        "**Reasoning checks:**\n\n"
        "1. **Causality vs. Amplitude (R2)**: Is your root cause selection based on "
        "a causal chain (who triggered whom) or just the largest anomaly? "
        "The service with the highest error rate is often a victim, not the cause.\n\n"
        "2. **Causal Direction (R5)**: Did you correctly identify cause vs. effect? "
        "If service A calls service B and both show errors, A's failure may cause B's errors "
        "— or vice versa. Ask: \"If I remove my candidate root cause, does the other anomaly disappear?\"\n\n"
        "3. **Anchoring Bias (R6)**: Did you form an early hypothesis and then ignore "
        "contradicting evidence found later? If evidence disproved your initial candidate, "
        "do not fall back to it — a disproven hypothesis should stay disproven.\n\n"
        "4. **Survivor Bias (R7)**: Are you only analyzing data that exists? "
        "A crashed or partitioned service may have STOPPED emitting traces/logs/metrics. "
        "Compare normal vs. abnormal span counts — missing data is a signal, not silence.\n\n"
        "5. **Shared Component Trap (M2)**: If your candidate root cause is a shared "
        "component (database, message queue, gateway), does it have independent "
        "failure evidence, or is it just receiving propagated errors from an upstream caller?\n\n"
        "6. **Absence ≠ Health (M3)**: Did you dismiss any service because you found "
        "\"no anomalous data\"? No data could mean the service crashed entirely "
        "and stopped emitting telemetry — verify with restart logs or container status.\n\n"
        "7. **Over-recursion (R4)**: If you already identified an application-layer root cause, "
        "are you still digging into infrastructure (JVM, container, network) unnecessarily? "
        "The deepest layer is not always the right answer.\n\n"
        "**Data interpretation checks:**\n\n"
        "8. **Cascade Amplification (D3)**: Could your candidate root cause actually be a "
        "second/third-order effect? Request pile-ups cause DB connection exhaustion, "
        "thread saturation causes HikariPool timeouts — trace the chain upstream to the origin.\n\n"
        "9. **Counter-intuitive Metrics (D5)**: Did any metric IMPROVE during the incident? "
        "Latency dropping may mean fast failures; average improving may mean bad requests "
        "disappeared (survivor bias); normal metrics may mean requests never reached the service.\n\n"
        "Re-examine your evidence and adjust your conclusion if needed."
    )

    def check_before_conclusion(self) -> str | None:
        """Pre-conclusion check. Returns fixed metacognitive prompt once."""
        if not self.enabled:
            return None
        if not self.config.check_before_conclusion:
            return None
        if self.state.conclusion_checked:
            return None

        self.state.conclusion_checked = True
        logger.info("Pre-conclusion metacognitive prompt injected at query %d", self.state.query_count)
        return self._CONCLUSION_PROMPT

    # ── Internal ────────────────────────────────────────────────────────

    def _should_check(self) -> bool:
        if self.state.intervention_count >= self.config.max_interventions:
            return False
        # Check if we've reached any checkpoint that hasn't been checked yet
        for cp in self.config.check_points:
            if self.state.query_count >= cp and cp not in self._checked_points:
                self._checked_points.add(cp)
                return True
        return False

    def _run_process_cycle(self) -> str | None:
        """L1 (batch classify) + L2-process + L3 in one cycle."""
        # Layer 1: batch-classify pending SQLs
        if self._pending_sqls:
            try:
                results = self.classifier.classify_batch(self._pending_sqls)
                for r in results:
                    self.state.add_intent(r)
                    logger.debug(
                        "L1 classified round %d: %s", r["round"], r["intent"]
                    )
            except Exception:
                logger.warning("L1 batch classification failed", exc_info=True)
            self._pending_sqls.clear()

        # Layer 2: process detection
        context = self.state.to_prompt_context()
        try:
            detection = self.process_detector.detect(context)
        except Exception:
            logger.warning("L2-process detection failed", exc_info=True)
            return None

        return self._maybe_intervene(detection, "process")

    def _run_conclusion_cycle(self) -> str | None:
        """L1 (flush pending) + L2-conclusion + L3."""
        # Flush any remaining pending SQLs through L1
        if self._pending_sqls:
            try:
                results = self.classifier.classify_batch(self._pending_sqls)
                for r in results:
                    self.state.add_intent(r)
            except Exception:
                logger.warning("L1 flush failed", exc_info=True)
            self._pending_sqls.clear()

        context = self.state.to_prompt_context()
        try:
            detection = self.conclusion_detector.detect(context)
        except Exception:
            logger.warning("L2-conclusion detection failed", exc_info=True)
            return None

        return self._maybe_intervene(detection, "conclusion")

    def _maybe_intervene(
        self, detection: dict, phase: str
    ) -> str | None:
        """Select critical deficiency, dedup, generate intervention."""
        # Dimension-level dedup: skip already-intervened dimensions
        candidates = [
            d["id"]
            for d in detection.get("observations", [])
            if d.get("detected") and d["id"] not in self._intervened_dimensions
        ]
        if not candidates:
            logger.debug(
                "L2-%s: no actionable deficiency at query %d",
                phase,
                self.state.query_count,
            )
            return None

        # Prefer LLM's most_critical if it passes dedup
        critical = detection.get("most_critical")
        if not critical or critical not in candidates:
            critical = candidates[0]

        # Get evidence
        evidence = ""
        for d in detection.get("observations", []):
            if d.get("id") == critical and d.get("detected"):
                evidence = d.get("evidence", "")
                break

        # Layer 3: generate intervention
        context = self.state.to_prompt_context()
        try:
            intervention = self.generator.generate(critical, evidence, context)
        except Exception:
            logger.warning(
                "L3 generation failed for %s", critical, exc_info=True
            )
            return None

        # Update state
        self._intervened_dimensions.add(critical)
        if phase == "conclusion":
            self.state.conclusion_intervention_count += 1
        else:
            self.state.intervention_count += 1

        tag = f"[Investigation Advisor — {critical}]"
        logger.info(
            "L3 %s intervention #%d (%s) at query %d",
            phase,
            self.state.intervention_count
            + self.state.conclusion_intervention_count,
            critical,
            self.state.query_count,
        )
        return f"{tag}\n{intervention}"
