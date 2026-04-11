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
