# Middleware v2 Design — D×R Aligned Metacognitive Pipeline

> Based on inductively-derived D×R failure taxonomy from thinkdepthai × Qwen 3.5 Plus analysis (21 failures, 151 judged).
> Previous middleware experiment: 50% fix rate (5/10), core gaps identified.

---

## 1. Design Principles

1. **Behavioral correction only** — middleware observes agent behavior patterns and provides metacognitive prompts. It NEVER hints at the answer, eliminates specific candidates, or injects domain-specific knowledge (e.g., "rabbitmq is not an injection target").
2. **D×R aligned** — deficiency definitions map to the inductively-derived D×R framework, not ad-hoc categories.
3. **Cross-framework** — works with any agent that uses the three parquet query tools (`list_tables_in_directory`, `get_schema`, `query_parquet_files`). Does not depend on framework-specific tool names or trajectory formats.
4. **Dual-timing** — process checks during investigation + conclusion check before final output.

---

## 2. Architecture

```
agent tool_call 后:
  L1: intent 分类 (batch, 每次 SQL 都分类, 累积到 state)
  reasoning 提取 (每轮从 think_tool reflection / assistant.content 提取)
      │
      ▼
  门槛检查: query_count ≥ 10 且 query_count % 5 == 0 且 intervention_count < 5
      │ 满足
      ▼
  检测一 (过程检测): LLM 检测 B1/B2/B3/B5/M1/M2
      │ 检出 most_critical (去重: 同维度最多干预 1 次)
      ▼
  L3: 生成干预 → 注入 HumanMessage

compress_research 前 (should_continue 路由拦截):
  检测二 (结论前检测): LLM 检测 M1/M2/M3/M4 → 最多干预 1 次
      │ 检出
      ▼
  L3: 生成干预 → 注入 HumanMessage → 打回 llm_call
```

### Layer Summary

| Layer | What | How | LLM calls |
|-------|------|-----|-----------|
| L1 | Intent classification | Batch classify pending SQLs | 1 per check cycle |
| L2-process | Behavioral + early metacognitive detection | Single LLM prompt, 6 dimensions | 1 per check cycle |
| L2-conclusion | Reasoning defect detection | Single LLM prompt, 4 dimensions | 1 (once) |
| L3 | Intervention generation | LLM generates targeted prompt | 1 per intervention |

---

## 3. Parameters

| Parameter | Value | Env var | Description |
|-----------|-------|---------|-------------|
| `enabled` | false | `ENABLE_MIDDLEWARE` | Master switch |
| `check_interval` | 5 | `MW_CHECK_INTERVAL` | Process check every N SQL queries |
| `min_queries_before_check` | 10 | `MW_MIN_QUERIES` | No checks before this |
| `max_interventions` | 5 | `MW_MAX_INTERVENTIONS` | Process intervention cap |
| `max_conclusion_interventions` | 1 | `MW_MAX_CONCLUSION` | Conclusion intervention cap |
| `max_per_dimension` | 1 | `MW_MAX_PER_DIM` | Same dimension intervention cap |
| `active_deficiencies` | B1,B2,B3,B5,M1,M2,M3,M4 | `MIDDLEWARE_DEFICIENCIES` | Toggle individual dimensions |
| `check_before_conclusion` | true | — | Enable conclusion check |

---

## 4. Deficiency Definitions

### 4.1 Process Detection (检测一) — 6 Dimensions

| ID | Name | D×R Mapping | Definition |
|----|------|-------------|------------|
| **B1** | Investigation Stagnation | behavioral | Agent's recent queries repeat the same intent type on the same services without gaining new information. |
| **B2** | Modal Incompleteness | D1 (infra-layer signal) | Agent only uses 1-2 data modalities (e.g., only traces/logs, no metrics). Critical evidence dimensions remain unexplored. |
| **B3** | Missing Baseline | D4 (noise defense) | Agent hasn't compared normal-period vs abnormal-period data. May be treating pre-existing anomalies as fault signals. |
| **B5** | Upstream Tunnel Vision | R5 (upstream unexplored) | Agent's investigation radiates only downstream/sideways from initial findings. Has not checked services upstream of anomalous services. |
| **M1** | Missing Causal Direction | R1+R2 (second-order + magnitude) | Agent found anomalies in >3 services but hasn't reasoned about causal relationships — just collecting anomalies without determining which caused which. |
| **M2** | Shared Component Hypothesis | R3 (infra anchoring) | Agent gravitates toward a shared infrastructure component as root cause. When multiple services share a component showing anomalies, the component may be a victim of upstream service failures, not the source. |

### 4.2 Conclusion Detection (检测二) — 4 Dimensions

| ID | Name | D×R Mapping | Definition |
|----|------|-------------|------------|
| **M1** | Missing Causal Direction | R1+R2 | When selecting root cause among multiple anomalous services, is the agent reasoning by causal chain position or by anomaly magnitude? |
| **M2** | Shared Component Hypothesis | R3 | If candidate root cause is a shared infrastructure component, has the agent verified it has an independent fault vs merely reflecting upstream service problems? |
| **M3** | Absence ≠ Health | R4 (missing data) | When eliminating a service, is the agent relying on positive healthy indicators or merely on absence of anomalous data? "No anomalous data" ≠ "confirmed healthy" — a service could have been down/unreachable. |
| **M4** | Over-recursion | R7 (over-tracing) | Did the agent identify an application service with clear anomalies but then continue tracing deeper to infrastructure (database, network), overriding its earlier finding? If the service-level anomaly sufficiently explains downstream impact, the root cause is that service. |

### 4.3 Mapping to Old Middleware

| New | Old | Status |
|-----|-----|--------|
| B1 | D1 (Stagnation) | Preserved — most effective intervention (core of 5/5 fixes) |
| B2 | D3+D4 (Tunnel Vision + Signal Layer) | Merged — both about missing evidence modalities |
| B3 | D5 (Noise Conflation) | Rephrased — "compare baseline" instead of "noise is misleading you" |
| B5 | — | New — addresses R5 (24% of failures) |
| M1 | D2 (Causal Confusion) | Strengthened — now with reasoning context, split process+conclusion |
| M2 | — | New — addresses R3 (19-39%), rephrased as behavioral check |
| M3 | — | New — addresses R4 (19-29%) |
| M4 | — | New — addresses R7 (10%, ClaudeCode-originated, cross-framework) |

---

## 5. Detection Prompts

### 5.1 Process Detection Prompt

```
You are a metacognitive monitor for an RCA agent investigating a
microservice incident. Based on its investigation trajectory, determine
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

B1 (Investigation Stagnation): Are recent queries repetitive? Is the
agent executing the same type of query on the same services without
gaining new information?

B2 (Modal Incompleteness): Has the agent only used 1-2 data modalities
(e.g., only traces, no metrics)? After {query_count} queries, is there
a critical evidence dimension completely unexplored?

B3 (Missing Baseline): Has the agent compared normal-period vs
abnormal-period data? If not, it may be treating pre-existing
anomalies as fault signals.

B5 (Upstream Tunnel Vision): Look at the services investigated. Is the
agent only expanding downstream/sideways from initial findings? Has it
checked services that CALL the anomalous services (upstream)?

M1 (Missing Causal Direction): If the agent found anomalies in >3
services, has it reasoned about causal relationships between them? Or
is it just collecting anomalies without determining which caused which?

M2 (Shared Component Hypothesis): Is the agent gravitating toward a
shared infrastructure component (message queue, database, gateway) as
root cause? When multiple services share a component showing anomalies,
the component may be a victim of upstream service failures, not the
source.

## Output
JSON only:
{
  "observations": [
    {"id": "B1/B2/B3/B5/M1/M2", "detected": true/false,
     "evidence": "brief reason"}
  ],
  "most_critical": "single most critical ID" or null
}
```

### 5.2 Conclusion Detection Prompt

```
You are a metacognitive monitor for an RCA agent that is about to
finalize its root cause conclusion. Check its reasoning for issues.

## Agent's Investigation Summary
- Total queries: {query_count}
- Services investigated: {services_list}
- Phases covered: {phases_visited}
- Has baseline comparison: {has_queried_normal}
- Recent reasoning: {recent_reasoning}

## Check Each Pattern

M1 (Missing Causal Direction): The agent found anomalies in multiple
services. Has it explicitly reasoned about which service's failure
CAUSED other services to fail? Or is it selecting root cause by which
service has the most/loudest anomalies?

M2 (Shared Component Hypothesis): If the candidate root cause is a
shared infrastructure component (database, message queue, load
balancer), has the agent verified this component has its OWN
independent fault, rather than merely reflecting problems from
application services?

M3 (Absence ≠ Health): When the agent eliminated a service from
consideration, was it because the service showed POSITIVE healthy
indicators, or merely because the agent found NO anomalous data?
"No anomalous data" and "confirmed healthy" are different — the
service could have been down/unreachable during the incident.

M4 (Over-recursion): Did the agent identify an application service
with clear anomalies, but then continue tracing deeper to
infrastructure (database, network), overriding its earlier finding?
If a service-level anomaly sufficiently explains downstream impact,
the root cause is that service, not a speculative deeper layer.

## Output
JSON only:
{
  "observations": [
    {"id": "M1/M2/M3/M4", "detected": true/false,
     "evidence": "brief reason"}
  ],
  "most_critical": "single most critical ID" or null
}
```

---

## 6. State Tracking

### New fields in `InvestigationState`

| Field | Type | Source | Used by |
|-------|------|--------|---------|
| `reasoning_log` | `list[str]` | Extracted every round from `think_tool.reflection` + `assistant.content` | Both prompts (`recent_reasoning` = last 3-5 entries joined) |
| `intervened_dimensions` | `set[str]` | Updated after each intervention | Dedup logic |

### Reasoning extraction (cross-framework)

```python
def extract_reasoning(self, tool_calls: list[dict], assistant_content: str):
    """Extract reasoning from current round. Call after every tool_node."""
    # Source 1: think_tool reflection (thinkdepthai, aiq, etc.)
    for tc in tool_calls:
        if tc["name"] == "think_tool":
            args = json.loads(tc["args"]) if isinstance(tc["args"], str) else tc["args"]
            reflection = args.get("reflection", "")
            if reflection:
                self.reasoning_log.append(reflection[:500])

    # Source 2: assistant content (ClaudeCode, deerflow, etc.)
    if assistant_content and len(assistant_content) > 50:
        self.reasoning_log.append(assistant_content[:500])
```

### Prompt context builder

```python
def to_prompt_context(self) -> dict:
    base = {
        "query_count": str(self.query_count),
        "intent_sequence": " → ".join(i["intent"] for i in self.intent_sequence),
        "recent_intents": " → ".join(self.recent_intents),
        "phases_visited": ", ".join(sorted(self.phases_visited)) or "none",
        "phases_missing": ", ".join(sorted(ALL_PHASES - self.phases_visited)) or "none",
        "modalities_used": ", ".join(sorted(self.modalities_used)) or "none",
        "services_list": ", ".join(sorted(self.services_investigated)) or "none",
        "has_queried_normal": "Yes" if self.has_queried_normal else "No",
        "recent_reasoning": "\n---\n".join(self.reasoning_log[-5:]) or "none",
    }
    return base
```

---

## 7. Dedup Logic

```python
def _select_critical(self, detection: dict) -> str | None:
    """Select most critical deficiency, respecting per-dimension cap."""
    candidates = [
        d["id"] for d in detection.get("observations", [])
        if d.get("detected") and d["id"] not in self._intervened_dimensions
    ]
    if not candidates:
        return None

    # Prefer LLM's most_critical if it passes dedup
    most_critical = detection.get("most_critical")
    if most_critical and most_critical in candidates:
        return most_critical

    # Otherwise pick first available
    return candidates[0]
```

After intervention:
```python
self._intervened_dimensions.add(critical)
self.state.intervention_count += 1
```

---

## 8. Intervention Generation

L3 (InterventionGenerator) is unchanged in structure. The prompt template
references the new dimension definitions from `DEFICIENCY_DEFINITIONS`.

Key constraint in L3 prompt (preserved):
```
Rules:
1. Do NOT tell the agent what the answer is or which service is the root cause
2. Do NOT name specific tables or SQL queries to run
3. DO make the agent aware of its blind spot
4. DO pose a pointed question that leads it to self-correct
5. Keep it to 2-4 sentences
6. Be concrete — reference what the agent has actually done
```

---

## 9. File Changes

| File | Change |
|------|--------|
| `config.py` | New params: `max_per_dimension`, `max_conclusion_interventions`. Deficiency codes → B1/B2/B3/B5/M1/M2/M3/M4. Default `max_interventions=5`. |
| `deficiency_detector.py` | Split into `ProcessDetector` + `ConclusionDetector` with separate prompts and dimension sets. New `PROCESS_DEFICIENCIES` and `CONCLUSION_DEFICIENCIES` dicts. |
| `pipeline.py` | Dedup changed to dimension-level (`_intervened_dimensions` set). `check_before_conclusion()` uses ConclusionDetector. `_run_batch_cycle()` uses ProcessDetector. |
| `state.py` | New fields: `reasoning_log: list[str]`, method `extract_reasoning()`. `to_prompt_context()` adds `recent_reasoning`. |
| `intervention_generator.py` | Update `DEFICIENCY_DEFINITIONS` import to new dict. No logic change. |
| `intent_classifier.py` | No change. |

---

## 10. Empirical Basis

### Why these dimensions

| Dimension | Source evidence | Expected impact |
|-----------|----------------|-----------------|
| B1 Stagnation | 5/5 FIXED cases used D1 intervention to break loops | High — proven effective |
| B2 Modal | D1 (infra-layer signal) affects 71% of failures | Medium — behavioral nudge |
| B3 Baseline | D4 (noise) affects 19-39%; baseline comparison prevents anchoring on pre-existing anomalies | Medium |
| B5 Upstream | R5 accounts for 24% of failures (GT RC never queried) | High — directly addresses root cause of 5 cases |
| M1 Causal | R1 (38%) + R2 (19%); old D2 was the #1 missing detection in MW experiment | High — addresses 3/5 MW failures |
| M2 Shared | R3 (19-39%); rabbitmq anchoring is the #1 ClaudeCode failure mode | High for ClaudeCode, medium for thinkdepthai |
| M3 Absence | R4 (19-29%); PodChaos cases where GT RC dismissed as "healthy" | Medium — targets specific failure mode |
| M4 Over-recursion | R7 (10%); ClaudeCode-specific but generalizable | Low-medium — preventive |

### What the old middleware missed

| Old MW failure | Root cause | New coverage |
|----------------|-----------|--------------|
| idx=807 (WRONG) | Agent found GT RC but selected downstream ts-ui-dashboard | M1 conclusion check |
| idx=2988 (WRONG) | Agent found GT RC but selected mysql | M4 conclusion check |
| idx=755 (WRONG) | Agent selected by signal magnitude (GC 66.8s) | M1 conclusion check |
| idx=2682 (WRONG) | 3× D5 repeated, no dedup | Dimension-level dedup (max 1 per dim) |
| idx=1798 (WRONG) | Data challenge too hard (D5 counter-intuitive) | Not addressable by middleware |

---

## 11. Validation Plan

1. **Smoke test**: Run MW v2 on the same 10 cases from MW v1 experiment
2. **Compare**: Did the 5 FIXED cases stay fixed? Did any of 807/2988/755 flip to FIXED?
3. **Regression check**: Did any previously correct cases break?
4. **Scale**: If smoke passes, run on all 21 failure cases
5. **Cross-framework**: Test on ClaudeCode with same dimensions
