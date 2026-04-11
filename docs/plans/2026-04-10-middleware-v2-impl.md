# Middleware v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the metacognitive middleware to align with the inductively-derived D×R failure taxonomy, with dual-timing detection (process + conclusion), dimension-level dedup, and cross-framework reasoning extraction.

**Architecture:** Replace old D1-D5 deficiencies with B1/B2/B3/B5 (behavioral) + M1/M2/M3/M4 (metacognitive). Split L2 into ProcessDetector and ConclusionDetector with separate prompts. Add reasoning_log to state for think_tool/assistant.content extraction. Keep L1 (intent_classifier) and L3 (intervention_generator) structure, update references.

**Tech Stack:** Python 3.12, LangChain (langchain_core.messages), no new dependencies.

**Design doc:** `Deep_Research/docs/plans/2026-04-10-middleware-v2-design.md`

---

### Task 1: Update `config.py`

**Files:**
- Modify: `Deep_Research/middleware/config.py`

**Step 1: Rewrite config with new parameters**

```python
"""Middleware configuration. All behavior controlled via environment variables."""

import os
from dataclasses import dataclass, field


@dataclass
class MiddlewareConfig:
    # Master switch: ENABLE_MIDDLEWARE=0 (default, silent) or 1 (active)
    enabled: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_MIDDLEWARE", "0") == "1"
    )

    # Per-deficiency toggles: MIDDLEWARE_DEFICIENCIES=B1,B2,B3,B5,M1,M2,M3,M4
    active_deficiencies: set[str] = field(
        default_factory=lambda: set(
            os.environ.get(
                "MIDDLEWARE_DEFICIENCIES", "B1,B2,B3,B5,M1,M2,M3,M4"
            ).split(",")
        )
    )

    # Process detection parameters
    check_interval: int = int(os.environ.get("MW_CHECK_INTERVAL", "5"))
    min_queries_before_check: int = int(os.environ.get("MW_MIN_QUERIES", "10"))
    max_interventions: int = int(os.environ.get("MW_MAX_INTERVENTIONS", "5"))

    # Conclusion detection parameters
    check_before_conclusion: bool = True
    max_conclusion_interventions: int = int(
        os.environ.get("MW_MAX_CONCLUSION", "1")
    )

    # Dedup: same dimension intervened at most N times
    max_per_dimension: int = int(os.environ.get("MW_MAX_PER_DIM", "1"))

    @property
    def process_deficiencies(self) -> set[str]:
        return self.active_deficiencies & {"B1", "B2", "B3", "B5", "M1", "M2"}

    @property
    def conclusion_deficiencies(self) -> set[str]:
        return self.active_deficiencies & {"M1", "M2", "M3", "M4"}
```

**Step 2: Verify import still works**

Run: `cd /home/nn/SOTA-agents/Deep_Research && uv run python -c "from middleware.config import MiddlewareConfig; c = MiddlewareConfig(); print(c.process_deficiencies, c.conclusion_deficiencies)"`

Expected: `{'B1', 'B2', 'B3', 'B5', 'M1', 'M2'} {'M1', 'M2', 'M3', 'M4'}`

---

### Task 2: Update `state.py`

**Files:**
- Modify: `Deep_Research/middleware/state.py`

**Step 1: Add reasoning_log and extract_reasoning, update to_prompt_context**

```python
"""Investigation state tracking. Pure data class, no external dependencies."""

import json
import re
from dataclasses import dataclass, field

PHASE_MAP = {
    "latency_ranking": "triage",
    "throughput_compare": "triage",
    "error_rate_scan": "triage",
    "error_log_overview": "triage",
    "metric_scan": "triage",
    "service_trace_scan": "trace",
    "trace_follow": "trace",
    "call_tree_build": "trace",
    "service_error_log": "log",
    "service_log_browse": "log",
    "keyword_search": "log",
    "error_timeline": "log",
    "container_resource": "metric",
    "jvm_state": "metric",
    "network_layer": "metric",
    "k8s_state": "metric",
    "db_state": "metric",
    "baseline_collect": "baseline",
    "baseline_contrast": "baseline",
}

ALL_PHASES = {"triage", "trace", "log", "metric", "baseline"}


@dataclass
class InvestigationState:
    intent_sequence: list[dict] = field(default_factory=list)
    raw_sqls: list[str] = field(default_factory=list)
    reasoning_log: list[str] = field(default_factory=list)
    query_count: int = 0
    intervention_count: int = 0
    conclusion_intervention_count: int = 0
    conclusion_checked: bool = False

    @property
    def phases_visited(self) -> set[str]:
        return {
            PHASE_MAP.get(i["intent"], "unknown")
            for i in self.intent_sequence
        } - {"unknown"}

    @property
    def modalities_used(self) -> set[str]:
        return {
            i["data_type"]
            for i in self.intent_sequence
            if i.get("data_type") and i["data_type"] != "unknown"
        }

    @property
    def services_investigated(self) -> set[str]:
        svcs: set[str] = set()
        for i in self.intent_sequence:
            svcs.update(i.get("services", []))
        return svcs

    @property
    def has_queried_normal(self) -> bool:
        for i in self.intent_sequence:
            if i["intent"] in ("baseline_collect", "baseline_contrast"):
                return True
        for sql in self.raw_sqls:
            if re.search(r"(?<!ab)normal_", sql.lower()):
                return True
        return False

    @property
    def recent_intents(self) -> list[str]:
        return [i["intent"] for i in self.intent_sequence[-10:]]

    def add_intent(self, intent_result: dict) -> None:
        self.intent_sequence.append(intent_result)

    def extract_reasoning(
        self, tool_calls: list[dict], assistant_content: str = ""
    ) -> None:
        """Extract reasoning from current round. Call after every tool_node."""
        for tc in tool_calls:
            if tc.get("name") == "think_tool":
                args = tc.get("args", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                reflection = args.get("reflection", "")
                if reflection:
                    self.reasoning_log.append(reflection[:500])

        if assistant_content and len(assistant_content) > 50:
            self.reasoning_log.append(assistant_content[:500])

    def to_prompt_context(self) -> dict:
        return {
            "query_count": str(self.query_count),
            "intent_sequence": " -> ".join(
                i["intent"] for i in self.intent_sequence
            ),
            "recent_intents": " -> ".join(self.recent_intents),
            "phases_visited": ", ".join(sorted(self.phases_visited)) or "none",
            "phases_missing": ", ".join(
                sorted(ALL_PHASES - self.phases_visited)
            ) or "none",
            "modalities_used": ", ".join(sorted(self.modalities_used)) or "none",
            "services_list": ", ".join(sorted(self.services_investigated)) or "none",
            "has_queried_normal": "Yes" if self.has_queried_normal else "No",
            "recent_reasoning": "\n---\n".join(
                self.reasoning_log[-5:]
            ) or "none",
        }
```

**Step 2: Quick verify**

Run: `cd /home/nn/SOTA-agents/Deep_Research && uv run python -c "
from middleware.state import InvestigationState
s = InvestigationState()
s.extract_reasoning([{'name': 'think_tool', 'args': {'reflection': 'Testing reflection'}}], '')
s.extract_reasoning([], 'This is a long enough assistant content to be captured by the filter')
print(len(s.reasoning_log), s.reasoning_log)
ctx = s.to_prompt_context()
print('recent_reasoning:', ctx['recent_reasoning'][:80])
"`

Expected: 2 entries in reasoning_log, recent_reasoning shows both.

---

### Task 3: Rewrite `deficiency_detector.py`

**Files:**
- Modify: `Deep_Research/middleware/deficiency_detector.py`

**Step 1: Full rewrite with ProcessDetector + ConclusionDetector**

```python
"""Layer 2: Deficiency detection via LLM reasoning.

v2: Dual-detector design aligned with D×R failure taxonomy.
  Process detection (B1/B2/B3/B5/M1/M2) — during investigation
  Conclusion detection (M1/M2/M3/M4) — before compress_research
"""

import json
import re

from langchain_core.messages import HumanMessage

# ── Dimension definitions ─────────────────────────────────────────────────────

PROCESS_DEFICIENCIES: dict[str, dict] = {
    "B1": {
        "name": "Investigation Stagnation",
        "description": (
            "The agent's recent queries repeat the same intent type on "
            "the same services without gaining new information."
        ),
        "check_question": (
            "Are recent queries repetitive? Is the agent executing the "
            "same type of query on the same services without gaining "
            "new information?"
        ),
    },
    "B2": {
        "name": "Modal Incompleteness",
        "description": (
            "The agent only uses 1-2 data modalities (e.g., only "
            "traces/logs, no metrics). Critical evidence dimensions "
            "remain completely unexplored."
        ),
        "check_question": (
            "Has the agent only used 1-2 data modalities? After "
            "{query_count} queries, is there a critical evidence "
            "dimension completely unexplored?"
        ),
    },
    "B3": {
        "name": "Missing Baseline",
        "description": (
            "The agent hasn't compared normal-period vs abnormal-period "
            "data. It may be treating pre-existing anomalies as fault "
            "signals."
        ),
        "check_question": (
            "Has the agent compared normal-period vs abnormal-period "
            "data? If not, it may be treating pre-existing anomalies "
            "as fault signals."
        ),
    },
    "B5": {
        "name": "Upstream Tunnel Vision",
        "description": (
            "The agent's investigation radiates only downstream or "
            "sideways from initial findings. It has not checked "
            "services upstream of the anomalous services."
        ),
        "check_question": (
            "Look at the services investigated: {services_list}. Is "
            "the agent only expanding downstream/sideways from its "
            "initial findings? Has it checked services that CALL the "
            "anomalous services (upstream)?"
        ),
    },
    "M1": {
        "name": "Missing Causal Direction",
        "description": (
            "The agent found anomalies in multiple services but hasn't "
            "reasoned about causal relationships — just collecting "
            "anomalies without determining which caused which."
        ),
        "check_question": (
            "If the agent found anomalies in >3 services, has it "
            "reasoned about causal relationships between them? Or is "
            "it just collecting anomalies without determining which "
            "caused which?"
        ),
    },
    "M2": {
        "name": "Shared Component Hypothesis",
        "description": (
            "The agent gravitates toward a shared infrastructure "
            "component (message queue, database, gateway) as root "
            "cause. When multiple services share a component showing "
            "anomalies, the component may be a victim of upstream "
            "service failures, not the source."
        ),
        "check_question": (
            "Is the agent gravitating toward a shared infrastructure "
            "component as root cause? When multiple services share a "
            "component and it shows anomalies, the component may be a "
            "victim, not the source."
        ),
    },
}

CONCLUSION_DEFICIENCIES: dict[str, dict] = {
    "M1": {
        "name": "Missing Causal Direction",
        "description": (
            "The agent found anomalies in multiple services. When "
            "selecting root cause, is it reasoning by causal chain "
            "position or by anomaly magnitude?"
        ),
        "check_question": (
            "Has the agent explicitly reasoned about which service's "
            "failure CAUSED other services to fail? Or is it selecting "
            "root cause by which service has the most/loudest anomalies?"
        ),
    },
    "M2": {
        "name": "Shared Component Hypothesis",
        "description": (
            "If the candidate root cause is a shared infrastructure "
            "component (database, message queue, load balancer), has "
            "the agent verified this component has its OWN independent "
            "fault, rather than merely reflecting problems from "
            "application services?"
        ),
        "check_question": (
            "If the candidate root cause is a shared infrastructure "
            "component, has the agent verified it has an independent "
            "fault vs merely reflecting upstream service problems?"
        ),
    },
    "M3": {
        "name": "Absence vs Health",
        "description": (
            "When the agent eliminated a service from consideration, "
            "was it because the service showed POSITIVE healthy "
            "indicators, or merely because the agent found NO "
            "anomalous data? 'No anomalous data' and 'confirmed "
            "healthy' are different — the service could have been "
            "down/unreachable during the incident."
        ),
        "check_question": (
            "When the agent eliminated a service, was it based on "
            "positive healthy indicators or merely absence of "
            "anomalous data? These are different."
        ),
    },
    "M4": {
        "name": "Over-recursion",
        "description": (
            "The agent identified an application service with clear "
            "anomalies, but then continued tracing deeper to "
            "infrastructure (database, network), overriding its "
            "earlier finding. If the service-level anomaly sufficiently "
            "explains downstream impact, the root cause is that "
            "service, not a speculative deeper layer."
        ),
        "check_question": (
            "Did the agent identify an application service with clear "
            "anomalies but then override it with a deeper "
            "infrastructure component? If the service-level anomaly "
            "explains the downstream impact, the root cause is that "
            "service."
        ),
    },
}

# Merged dict for L3 intervention generator lookups
ALL_DEFICIENCIES = {**PROCESS_DEFICIENCIES, **CONCLUSION_DEFICIENCIES}

# ── Prompts ───────────────────────────────────────────────────────────────────

PROCESS_DETECT_PROMPT = """\
You are a metacognitive monitor for an RCA agent investigating a \
microservice incident. Based on its investigation trajectory, determine \
if it exhibits any behavioral issues.

## Agent's Investigation State
- Total SQL queries: {query_count}
- Recent 10 intents: {recent_intents}
- Phases covered: {phases_visited} (missing: {phases_missing})
- Data modalities used: {modalities_used}
- Services investigated: {services_list}
- Has compared normal vs abnormal data: {has_queried_normal}
- Recent reasoning: {recent_reasoning}

## Check Each Pattern
{deficiency_descriptions}

## Output
Respond with JSON only:
{{"observations": [{{"id": "...", "detected": true/false, "evidence": "brief reason"}}], "most_critical": "single most critical ID" or null}}
"""

CONCLUSION_DETECT_PROMPT = """\
You are a metacognitive monitor for an RCA agent that is about to \
finalize its root cause conclusion. Check its reasoning for issues.

## Agent's Investigation Summary
- Total queries: {query_count}
- Services investigated: {services_list}
- Phases covered: {phases_visited}
- Has baseline comparison: {has_queried_normal}
- Recent reasoning: {recent_reasoning}

## Check Each Pattern
{deficiency_descriptions}

## Output
Respond with JSON only:
{{"observations": [{{"id": "...", "detected": true/false, "evidence": "brief reason"}}], "most_critical": "single most critical ID" or null}}
"""


# ── Detectors ─────────────────────────────────────────────────────────────────


class _BaseDetector:
    """Shared detection logic for both process and conclusion detectors."""

    def __init__(self, model, active_deficiencies: set[str]):
        self.model = model
        self.active_deficiencies = active_deficiencies

    def _build_descriptions(
        self, deficiency_defs: dict, state_context: dict
    ) -> str:
        parts = []
        for did in sorted(self.active_deficiencies):
            ddef = deficiency_defs.get(did)
            if not ddef:
                continue
            try:
                check_q = ddef["check_question"].format(**state_context)
            except KeyError:
                check_q = ddef["check_question"]
            parts.append(
                f"**{did} ({ddef['name']})**: {ddef['description']}\n"
                f"Check: {check_q}"
            )
        return "\n\n".join(parts)

    def _call_llm(self, prompt: str) -> dict:
        response = self.model.invoke([HumanMessage(content=prompt)])
        return self._parse_response(response.content)

    def _parse_response(self, text: str) -> dict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "observations" in data:
                data["observations"] = [
                    d for d in data["observations"]
                    if d.get("id") in self.active_deficiencies
                ]
                if data.get("most_critical") not in self.active_deficiencies:
                    detected = [
                        d["id"] for d in data["observations"]
                        if d.get("detected")
                    ]
                    data["most_critical"] = detected[0] if detected else None
                return data
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return {"observations": [], "most_critical": None}


class ProcessDetector(_BaseDetector):
    """Detect behavioral deficiencies during investigation."""

    def detect(self, state_context: dict) -> dict:
        descs = self._build_descriptions(PROCESS_DEFICIENCIES, state_context)
        if not descs:
            return {"observations": [], "most_critical": None}
        prompt = PROCESS_DETECT_PROMPT.format(
            deficiency_descriptions=descs, **state_context
        )
        return self._call_llm(prompt)


class ConclusionDetector(_BaseDetector):
    """Detect reasoning defects before conclusion."""

    def detect(self, state_context: dict) -> dict:
        descs = self._build_descriptions(
            CONCLUSION_DEFICIENCIES, state_context
        )
        if not descs:
            return {"observations": [], "most_critical": None}
        prompt = CONCLUSION_DETECT_PROMPT.format(
            deficiency_descriptions=descs, **state_context
        )
        return self._call_llm(prompt)
```

**Step 2: Verify import**

Run: `cd /home/nn/SOTA-agents/Deep_Research && uv run python -c "from middleware.deficiency_detector import ProcessDetector, ConclusionDetector, ALL_DEFICIENCIES; print(sorted(ALL_DEFICIENCIES.keys()))"`

Expected: `['B1', 'B2', 'B3', 'B5', 'M1', 'M2', 'M3', 'M4']`

---

### Task 4: Update `intervention_generator.py`

**Files:**
- Modify: `Deep_Research/middleware/intervention_generator.py`

**Step 1: Update import to use ALL_DEFICIENCIES**

```python
"""Layer 3: Targeted intervention generation based on detected deficiency."""

from langchain_core.messages import HumanMessage

from .deficiency_detector import ALL_DEFICIENCIES

INTERVENTION_PROMPT = """\
You are generating a targeted metacognitive prompt for an RCA agent \
that is exhibiting a specific cognitive deficiency during its investigation.

## Detected Deficiency
{deficiency_id} — {deficiency_name}: {deficiency_description}
Evidence: {evidence}

## Current Investigation Context
- Recent intents: {recent_intents}
- Services investigated: {services_list}
- Missing evidence dimensions: {phases_missing}
- Recent reasoning: {recent_reasoning}

## Rules
1. Do NOT tell the agent what the answer is or which service is the root cause
2. Do NOT name specific tables or SQL queries to run
3. DO make the agent aware of its blind spot
4. DO pose a pointed question that leads it to self-correct
5. Keep it to 2-4 sentences, like a senior SRE reviewing a junior's investigation
6. Be concrete — reference what the agent has actually done (from the context)

## Output
Respond with ONLY the intervention message (2-4 sentences). No JSON, no markdown.
"""


class InterventionGenerator:
    """Generate context-specific intervention for a detected deficiency."""

    def __init__(self, model):
        self.model = model

    def generate(
        self, deficiency_id: str, evidence: str, state_context: dict
    ) -> str:
        ddef = ALL_DEFICIENCIES[deficiency_id]

        prompt = INTERVENTION_PROMPT.format(
            deficiency_id=deficiency_id,
            deficiency_name=ddef["name"],
            deficiency_description=ddef["description"],
            evidence=evidence,
            **state_context,
        )

        response = self.model.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
```

---

### Task 5: Rewrite `pipeline.py`

**Files:**
- Modify: `Deep_Research/middleware/pipeline.py`

**Step 1: Full rewrite with dual-detector and dimension-level dedup**

```python
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

        if not self._should_check():
            return None

        return self._run_process_cycle()

    def check_before_conclusion(self) -> str | None:
        """Pre-conclusion check. Returns intervention or None."""
        if not self.enabled:
            return None
        if not self.config.check_before_conclusion:
            return None
        if self.state.conclusion_checked:
            return None

        self.state.conclusion_checked = True
        return self._run_conclusion_cycle()

    # ── Internal ────────────────────────────────────────────────────────

    def _should_check(self) -> bool:
        if self.state.intervention_count >= self.config.max_interventions:
            return False
        if self.state.query_count < self.config.min_queries_before_check:
            return False
        if self.state.query_count % self.config.check_interval != 0:
            return False
        return True

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
```

**Step 2: Update `__init__.py`** (no change needed — already exports MiddlewarePipeline and MiddlewareConfig)

**Step 3: Verify full import chain**

Run: `cd /home/nn/SOTA-agents/Deep_Research && uv run python -c "from middleware import MiddlewarePipeline, MiddlewareConfig; print('OK')"`

Expected: `OK`

---

### Task 6: Update `agent_runner.py` integration

**Files:**
- Modify: `Deep_Research/agent_runner.py` (only the `tool_node_mw` function)

The current `tool_node_mw` calls `pipeline.process_tool_calls(tool_calls, round_num)`. We need to pass `assistant_content` for reasoning extraction.

Find the `tool_node_mw` function (around line 146-159) and update:

```python
    def tool_node_mw(state: ResearcherState):
        messages = state["researcher_messages"]
        tool_calls = messages[-1].tool_calls
        outputs = []
        for tc in tool_calls:
            tool = RCA_TOOLS_BY_NAME[tc["name"]]
            result = tool.invoke(tc["args"])
            outputs.append(ToolMessage(content=str(result), name=tc["name"], tool_call_id=tc["id"]))

        round_num = sum(1 for m in messages if hasattr(m, "tool_calls") and m.tool_calls)

        # Extract assistant content for reasoning log
        # Walk back to find the most recent assistant message with content
        assistant_content = ""
        for m in reversed(messages):
            if hasattr(m, "content") and hasattr(m, "tool_calls"):
                # AIMessage with content
                assistant_content = m.content or ""
                break

        intervention = pipeline.process_tool_calls(
            tool_calls, round_num, assistant_content
        )
        if intervention:
            outputs.append(HumanMessage(content=intervention))

        return {"researcher_messages": outputs}
```

The `should_continue_mw` function needs no change — `check_before_conclusion()` API is the same.

---

### Task 7: Smoke test

**Step 1: Run import verification**

Run: `cd /home/nn/SOTA-agents/Deep_Research && uv run python -c "
from middleware import MiddlewarePipeline, MiddlewareConfig
from middleware.deficiency_detector import ProcessDetector, ConclusionDetector, ALL_DEFICIENCIES, PROCESS_DEFICIENCIES, CONCLUSION_DEFICIENCIES
from middleware.state import InvestigationState
from middleware.config import MiddlewareConfig as MC

# Verify dimensions
assert set(PROCESS_DEFICIENCIES.keys()) == {'B1','B2','B3','B5','M1','M2'}
assert set(CONCLUSION_DEFICIENCIES.keys()) == {'M1','M2','M3','M4'}
assert set(ALL_DEFICIENCIES.keys()) == {'B1','B2','B3','B5','M1','M2','M3','M4'}

# Verify config defaults
c = MC()
assert c.max_interventions == 5
assert c.max_per_dimension == 1
assert c.process_deficiencies == {'B1','B2','B3','B5','M1','M2'}
assert c.conclusion_deficiencies == {'M1','M2','M3','M4'}

# Verify state
s = InvestigationState()
s.extract_reasoning([{'name':'think_tool','args':{'reflection':'test'}}], '')
assert len(s.reasoning_log) == 1
ctx = s.to_prompt_context()
assert 'recent_reasoning' in ctx

# Verify pipeline disabled mode
p = MiddlewarePipeline()
assert not p.enabled
assert p.process_tool_calls([], 1) is None
assert p.check_before_conclusion() is None

print('All checks passed')
"`

Expected: `All checks passed`

---
