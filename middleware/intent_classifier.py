"""Layer 1: Real-time intent classification via Qwen-as-judge.

Reuses the same prompt design as RCAgentEval/utu/eval/analysis/intent_prompt.py
but runs independently — no imports from RCAgentEval at runtime.

Key method: classify_batch() — classifies multiple SQLs in a single LLM call.
"""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ── Prompt constants (mirrored from intent_prompt.py, do NOT modify originals) ──

VALID_INTENTS = [
    "latency_ranking", "throughput_compare", "error_rate_scan",
    "service_trace_scan", "trace_follow", "call_tree_build",
    "error_log_overview", "service_error_log", "service_log_browse",
    "keyword_search", "error_timeline", "metric_scan",
    "container_resource", "jvm_state", "network_layer",
    "k8s_state", "db_state", "baseline_collect", "baseline_contrast",
]

SYSTEM_PROMPT = """\
Classify each SQL query into one of the intents below. Pick the closest match.

## Intents

Traces:
- latency_ranking: Global latency overview across all services (GROUP BY service, AVG/MAX duration)
- throughput_compare: Global request volume comparison (GROUP BY service, COUNT)
- error_rate_scan: Global error distribution (GROUP BY service, status_code / error count)
- service_trace_scan: Examine a specific service's traces — spans, duration, status, endpoints
  (WHERE service = X, or LIKE '%service_name%' on traces)
- trace_follow: Follow one request by trace_id (WHERE trace_id = X)
- call_tree_build: Build call graph (JOIN on parent_span_id)

Logs:
- error_log_overview: Global log scan across services (GROUP BY service, level)
- service_error_log: Specific service's error logs (WHERE service = X AND level = ERROR/WARN)
- service_log_browse: Browse a service's logs without level filter (WHERE service = X)
- keyword_search: Search by keyword (LIKE '%timeout%', '%OOM%', '%chaos%', etc.)
- error_timeline: Establish error timeline — first/last occurrence, time range
  (MIN/MAX time, ORDER BY time with error focus, EPOCH)

Metrics:
- metric_scan: Explore or browse metrics — available metric names, dimensions, values
  (SELECT DISTINCT metric/pod/workload, or browse metrics without specific domain filter)
- container_resource: CPU / memory metrics (container.cpu, container.memory, memory.working_set)
- jvm_state: JVM metrics (jvm, gc, hikari, thread, heap)
- network_layer: Network metrics (hubble, http_request, tcp, drop, p95)
- k8s_state: Kubernetes state (k8s.pod.phase, restart, deployment)
- db_state: Database metrics (db.client, mysql, connections)

Baseline:
- baseline_collect: Query only normal_* tables (establishing baseline)
- baseline_contrast: Compare normal vs abnormal (JOIN/UNION normal + abnormal tables)

## Rules

1. One intent per SQL. If a round has N SQL queries, return N entries.
2. Use ONLY the intent names listed above.
3. Keep reasoning under 15 words.
4. Return only the JSON array.
"""

FEW_SHOT_USER = """\
Round 4 (2 SQL):
```sql
-- SQL 1
SELECT service_name, COUNT(*), AVG(duration), SUM(CASE WHEN attr_status_code='ERROR' THEN 1 ELSE 0 END) as errors
FROM abnormal_traces GROUP BY service_name ORDER BY errors DESC
-- SQL 2
SELECT service_name, level, COUNT(*) FROM abnormal_logs GROUP BY service_name, level
```

Round 6 (2 SQL):
```sql
-- SQL 1
SELECT time, message FROM abnormal_logs
WHERE service_name = 'ts-order-service' AND level IN ('ERROR','SEVERE')
-- SQL 2
SELECT DISTINCT service_name FROM abnormal_traces
```

Round 8 (1 SQL):
```sql
-- SQL 1
SELECT * FROM abnormal_traces WHERE trace_id = 'abc123' ORDER BY time
```
"""

FEW_SHOT_ASSISTANT = """\
```json
[
  {"round": 4, "sql_index": 1, "intent": "error_rate_scan", "reasoning": "global GROUP BY service with error count"},
  {"round": 4, "sql_index": 2, "intent": "error_log_overview", "reasoning": "global log scan by service and level"},
  {"round": 6, "sql_index": 1, "intent": "service_error_log", "reasoning": "specific service + ERROR level filter"},
  {"round": 6, "sql_index": 2, "intent": "latency_ranking", "reasoning": "listing all services from traces"},
  {"round": 8, "sql_index": 1, "intent": "trace_follow", "reasoning": "WHERE trace_id = specific ID"}
]
```\
"""

_VALID_INTENTS_SET = set(VALID_INTENTS)

# ── Data type classification (self-contained, same logic as llm_intent_classifier.py) ──


def classify_data_type(sql: str) -> str:
    """Classify SQL query's data modality by table name in FROM/JOIN clause."""
    from_tables = re.findall(r"(?:FROM|JOIN)\s+(\w+)", sql, re.IGNORECASE)
    types_found: set[str] = set()
    for table in from_tables:
        tl = table.lower()
        if "log" in tl:
            types_found.add("logs")
        elif "trace" in tl:
            types_found.add("traces")
        elif "metric" in tl:
            types_found.add("metrics")
    if len(types_found) == 1:
        return types_found.pop()
    if len(types_found) > 1:
        for table in from_tables:
            tl = table.lower()
            if "log" in tl:
                return "logs"
            if "trace" in tl:
                return "traces"
            if "metric" in tl:
                return "metrics"
    return "unknown"


def extract_services_from_sql(sql: str) -> list[str]:
    """Extract ts-*-service names mentioned in SQL."""
    return list(set(re.findall(r"ts-[\w-]+-service", sql, re.IGNORECASE)))


# ── Classifier ─────────────────────────────────────────────────────────────────


class IntentClassifier:
    """Batch intent classification — classifies multiple SQLs in one LLM call."""

    def __init__(self, model):
        self.model = model

    def classify_batch(self, sqls: list[tuple[str, int]]) -> list[dict]:
        """Classify multiple SQL queries in a single LLM call.

        Args:
            sqls: list of (sql_text, round_num) tuples.

        Returns:
            list of {"intent": str, "data_type": str, "services": list[str], "round": int}
            Same length as input. Falls back to "unknown" on parse failure.
        """
        if not sqls:
            return []

        # Build multi-round user message
        # Group by round_num, assign sql_index within each round
        rounds: dict[int, list[str]] = {}
        for sql, rnd in sqls:
            rounds.setdefault(rnd, []).append(sql)

        parts = []
        for rnd in sorted(rounds):
            sql_list = rounds[rnd]
            n = len(sql_list)
            sql_block = "\n".join(f"-- SQL {i+1}\n{sql}" for i, sql in enumerate(sql_list))
            parts.append(f"Round {rnd} ({n} SQL):\n```sql\n{sql_block}\n```")

        user_text = "\n\n".join(parts)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=FEW_SHOT_USER),
            AIMessage(content=FEW_SHOT_ASSISTANT),
            HumanMessage(content=user_text),
        ]

        response = self.model.invoke(messages)
        parsed = self._parse_batch_response(response.content)

        # Build result list aligned with input order
        # Create lookup: (round, sql_index) -> intent
        intent_lookup: dict[tuple[int, int], str] = {}
        for entry in parsed:
            key = (entry.get("round", 0), entry.get("sql_index", 0))
            intent_lookup[key] = entry.get("intent", "unknown")

        results = []
        # Reconstruct (round, sql_index) in input order
        round_counters: dict[int, int] = {}
        for sql, rnd in sqls:
            round_counters.setdefault(rnd, 0)
            round_counters[rnd] += 1
            sql_index = round_counters[rnd]

            intent = intent_lookup.get((rnd, sql_index), "unknown")
            if intent not in _VALID_INTENTS_SET:
                intent = "unknown"

            results.append({
                "intent": intent,
                "data_type": classify_data_type(sql),
                "services": extract_services_from_sql(sql),
                "round": rnd,
            })

        return results

    @staticmethod
    def _parse_batch_response(text: str) -> list[dict]:
        """Parse LLM response into list of intent entries."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: try to find JSON array in the text
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return []
