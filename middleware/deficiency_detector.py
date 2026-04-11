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
