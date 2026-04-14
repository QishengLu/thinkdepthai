# CLAUDE.md — Metacognitive Middleware (v3)

元认知中间件：在 RCA agent 推理过程中注入行为纠偏提示，不泄漏答案方向。

---

## 设计原则

1. **只做元认知，不泄漏信息**——中间件观察 agent 的行为模式（查了什么数据类型、覆盖了哪些阶段、是否重复查询），提供方法论层面的提醒。绝不提示具体该查哪个服务、哪张表、或答案方向。
2. **D×R 失败分类对齐**——缺陷维度（B1/B2/B3/B5/M1-M4）来自对 thinkdepthai × Qwen 3.5 Plus 失败 case 的归纳分析。
3. **双时机检测**——中期过程检查（agent 还在调查时）+ 结论前检查（agent 准备收敛时）。
4. **跨框架兼容**——只依赖 `query_parquet_files` 调用计数和 `think_tool` 反思内容，不依赖特定 agent 框架。

---

## 三层架构

```
agent tool_call 后:
  L1: Intent 分类 (batch, 累积到 state)
  Reasoning 提取 (think_tool reflection / assistant.content)
      │
      ▼
  检查点判断: query_count 到达 [37, 44] 且未检查过该点 且 intervention_count < 3
      │ 满足
      ▼
  L2-process: LLM 检测 B1/B2/B3/B5/M1/M2 → 选 most_critical（同维度去重）
      │ 检出
      ▼
  L3: 生成干预 → 注入 HumanMessage

compress_research 前:
  L2-conclusion: LLM 检测 M1/M2/M3/M4 → 全局最多 1 次
      │ 检出
      ▼
  L3: 生成干预 → 注入 HumanMessage → 打回 llm_call
```

| Layer | 功能 | LLM 调用 |
|-------|------|---------|
| L1 | SQL 意图批量分类 | 1 次/检查周期 |
| L2-process | 行为缺陷检测（6 维） | 1 次/检查周期 |
| L2-conclusion | 推理缺陷检测（4 维） | 1 次（全局） |
| L3 | 干预文本生成 | 1 次/干预 |

---

## 检查点设计（v3 核心改进）

### 从固定间隔到数据驱动的检查点

v2 使用 `query_count % 5 == 0` 固定间隔，导致：
- 过早检查（query 10/15）：此时正确/错误 case 都在正常调查，误伤率高
- 过多检查：典型 case 触发 4-5 次检查 × 3 次 LLM 调用 = 12-15 次额外 LLM，每次 ~120s

v3 基于 thinkdepthai-qwen3.5-plus 500 条全量数据的统计分析，选择精确率跳变点：

#### 数据依据

| query count | 正确 case 已结束 | 错误 case 还在跑 | 精确率（wrong/all_running） |
|-------------|-----------------|-----------------|--------------------------|
| 15 | 6.8% | 98.1% | 21.9% |
| 25 | 27.6% | 88.6% | 24.5% |
| 30 | 43.3% | 75.2% | 26.1% |
| **37** | **59.5%** | **56.2%** | **30.6% (Δ+1.8%)** |
| 40 | 72.9% | 45.7% | 31.0% |
| **44** | **— ** | **36.2%** | **36.5% (Δ+5.5%)** |
| 50 | — | 20.0% | 36.2% |

**选择 query 37 和 44 的原因**：这两个点是精确率曲线斜率最大的拐点——大批正确 case 刚集中收敛离场，剩余 pool 中错误 case 占比突然升高。在这两个点检查，能最大化"命中错误 case"的概率，同时最小化"干扰正确 case"的副作用。

#### 检查触发逻辑

```python
check_points = [37, 44]  # 可通过 MW_CHECK_POINTS 环境变量覆盖

def _should_check(self) -> bool:
    if self.state.intervention_count >= self.config.max_interventions:
        return False
    for cp in self.config.check_points:
        if self.state.query_count >= cp and cp not in self._checked_points:
            self._checked_points.add(cp)
            return True
    return False
```

#### 三道防线分工

| 防线 | 触发时机 | 覆盖目标 |
|------|---------|---------|
| 中期检查 1 | query 37 | 长期跑偏的错误 case（query 30-50，占 75%） |
| 中期检查 2 | query 44 | 顽固维度锁定（query 40+，精确率 36%+） |
| 结论前检查 | compress 前 | 过早收敛型（query <37 的 26 个错误 case） |

---

## 参数配置

| 参数 | 默认值 | 环境变量 | 说明 |
|------|--------|---------|------|
| `enabled` | false | `ENABLE_MIDDLEWARE` | 总开关 |
| `check_points` | [37, 44] | `MW_CHECK_POINTS` | 中期检查点（逗号分隔） |
| `max_interventions` | 3 | `MW_MAX_INTERVENTIONS` | 干预上限（含中期+结论） |
| `max_conclusion_interventions` | 1 | `MW_MAX_CONCLUSION` | 结论前干预上限（全局 1 次） |
| `max_per_dimension` | 1 | `MW_MAX_PER_DIM` | 同维度干预上限 |
| `active_deficiencies` | B1-M4 | `MIDDLEWARE_DEFICIENCIES` | 启用的缺陷维度 |

---

## 缺陷维度定义

### 过程检测（L2-process）— 6 维

| ID | 名称 | 检测内容 |
|----|------|---------|
| **B1** | Investigation Stagnation | 最近查询在同一服务上重复相同意图，无新信息 |
| **B2** | Modal Incompleteness | 只用了 1-2 种数据模态（如只看 traces，没看 metrics） |
| **B3** | Missing Baseline | 未对比正常期 vs 异常期数据，可能把常态当异常 |
| **B5** | Upstream Tunnel Vision | 只向下游/平级扩展，未检查上游调用者 |
| **M1** | Missing Causal Direction | 发现多服务异常但未推理因果关系，只是收集异常 |
| **M2** | Shared Component Hypothesis | 锁定共享组件（DB/MQ）为根因，但它可能只是受害者 |

### 结论前检测（L2-conclusion）— 4 维

| ID | 名称 | 检测内容 |
|----|------|---------|
| **M1** | Missing Causal Direction | 选根因靠异常幅度而非因果链位置 |
| **M2** | Shared Component Hypothesis | 候选根因是共享组件，未验证其独立故障 |
| **M3** | Absence ≠ Health | 排除服务靠"没有异常数据"而非"确认健康" |
| **M4** | Over-recursion | 已识别应用层根因但继续深挖基础设施层，覆盖了正确答案 |

---

## 干预生成约束

L3 生成干预文本时的硬规则：

1. **不能**告诉 agent 答案或哪个服务是根因
2. **不能**指定具体的表名或 SQL 查询
3. **应当**让 agent 意识到自己的盲点
4. **应当**提出引导性问题让 agent 自我纠正
5. 限制在 2-4 句话
6. 要具体——引用 agent 实际做过的事

---

## 实测分析：idx=99 case study

### 背景
- GT: `ts-consign-price-service`（JVMMemoryStress）
- 原版（无 MW）：47 rounds / 272s → **答错**（选了 ts-consign-service）
- MW v2：131 rounds / 2760s → **答对**

### 干预时间线（v2 配置，check_interval=5）

| # | Step | 类型 | 干预内容 | 效果 |
|---|------|------|---------|------|
| 1 | 30 | B3 | 交叉验证 error logs | 查了更多 error，仍盯着 consign-service |
| 2 | 40 | B2 | 只看了 traces/logs，没看 metrics | 查了 metrics，但仍查旧服务 |
| 3 | 50 | B1 | 连续 5 次查 consign-service | 换了查询角度，没换目标服务 |
| 4 | 60 | M1 | 别统计数量，分析因果关系 | **转折点**：agent 做深度反思 |
| 5 | 94 | B5 | （最后一次干预） | agent 已在深挖 consign-price-service |

**关键转折**：第 4 次干预（M1）后 agent 在 step 74 首次查到 `ts-consign-price-service`，发现 `memory.major_page_fault` 证据，最终答对。

### 耗时分解

| 类别 | 时间 | 占比 |
|------|------|------|
| MW LLM 调用（5 次检查 × 3 层） | ~800s | 29% |
| MW 未干预的评估调用 | ~400s | 14% |
| Agent 正常 LLM 调用 | ~1200s | 44% |
| Tool 执行 | ~360s | 13% |

### v3 优化预期

使用 v3 检查点 [37, 44] + max_interventions=3：
- 检查次数：从 ~10 次降到 2 次中期 + 1 次结论 = **3 次**
- MW LLM 调用：从 ~15 次降到 ~9 次（3 次 × 3 层）
- 预计耗时：从 2760s 降到 ~800-1200s（仍高于原版 272s，但合理范围内）
- 干预效果：第 1 次中期检查在 query 37 才触发，此时 agent 已经积累了足够的调查数据，干预更有针对性

---

## 正确/错误 case 统计（thinkdepthai-qwen3.5-plus, 500 条）

| 指标 | 正确 (395) | 错误 (105) |
|------|-----------|-----------|
| 平均 rounds | 43.1 | 50.8 |
| 平均 queries | 33.5 | 39.8 |
| P25-P75 queries | — | 31-48 |
| 最少 queries | 11 | 15 |

错误 case 的两种失败模式：
- **过早收敛**（query 15-30, 26 个）：查太少就下结论 → `check_before_conclusion` 拦截
- **维度锁定**（query 30-50+, 79 个）：查很多但方向错 → 中期检查 37/44 纠偏

---

## v3 实跑结果（thinkdepthai-qwen3.5-plus-mw-v3, 105 case, 2026-04-14 分析）

**整体翻正率 49/105 = 46.7%**，零回归。完整 case 对比报告：[`analysis/4-middleware/MW-vs-w_o-MW.md`](../../analysis/4-middleware/MW-vs-w_o-MW.md)。

### 干预触发覆盖

| 干预类型 | 触发 case 数 | 触发率 |
|---|---|---|
| Process advisor (B1/B2/B3/B5/M1/M2) | 61 / 105 | 58% |
| Conclusion check (固定 9 项) | **105 / 105** | 100% |

### 干预触发分布与设计预期的偏差

| 设计预期 | 实测 | 偏差原因 |
|---|---|---|
| query 37/44 双中期检查触发 ~80% case | 仅 58% (61/105) | **44/105 (42%) case 在 qpf < 37 提前收敛**，process 检查永远没机会触发 |
| Conclusion check 兜底 ~26 个过早收敛 case | 实际 105/105 都触发 | 兜底机制工作，但效果几乎不可观测——见下方 Bug 1 |

### wrong→correct 的 49 个 case 的承重干预归因

| 救回方式 | 数量 | 备注 |
|---|---|---|
| Process + Conclusion 双触发 | 28 | process 干预承重 (B3 最有效) |
| 仅 Conclusion 触发 (process silent) | 21 | qpf<37 提前收敛 |
| ↳ conclusion=back_to_tools | 9 | agent 真听话又查了 1-3 轮 tool |
| ↳ conclusion=rewrite | 12 | agent 没听话只重写——疑似 L1 分类器扰动采样 |

### Process advisor 中各维度的实际承重次数

| 维度 | 触发次数 (process 阶段) | wrong→correct 承重 | wrong→wrong 触发但失效 |
|---|---|---|---|
| B3 缺基线 | 14 | **14 (highest impact)** | 0 |
| B1 调查停滞 | 14 | 14 | 多次 misdirected |
| B5 上游盲区 | 6 | 6 | 0 |
| M1 缺因果方向 | 6 | 6 | 多次 misdirected |
| M2 共享组件锚定 | 6 | 6 | 0 |
| B2 模态不全 | 2 | 2 | 0 |

---

## ⚠️ v3 已知 Bug（必须 v4 修复）

### Bug 1 — `should_continue_mw` 在 conditional edge 里 mutate state

**位置**：[`agent_runner.py:174-181`](../agent_runner.py#L174-L181)

```python
def should_continue_mw(state: ResearcherState) -> Literal[...]:
    if state["researcher_messages"][-1].tool_calls:
        return "tool_node"
    intervention = pipeline.check_before_conclusion()
    if intervention:
        state["researcher_messages"].append(HumanMessage(content=intervention))  # ← BUG
        return "llm_call"
    return "compress_research"
```

**问题**：LangGraph conditional edge 函数的契约是**只能返回路由字符串、不能 mutate state**。这里 `state["researcher_messages"].append(...)` 是在原地改 dict，**不会经过 `add_messages` reducer**，结果是：

1. 注入的 HumanMessage **能被下一个 llm_call 节点读到**（state 是同一个内存对象）→ agent 确实会响应它
2. 但 LangGraph 的 stream event **不会 emit 这条 HumanMessage**（它不是节点 return value）
3. 因此 trajectory 序列化时**永远看不到这条 conclusion 干预内容**
4. 同样的 reducer 也保证了 v3 重启不会读到 conclusion 干预——只在内存 lifecycle 内有效

**实证**：
- DB 全文搜 `Pre-Conclusion` 在 105 个 MW case 里命中 **0 次**
- baseline 50 sample 中**没有任何**连续 assistant 形态
- MW 105 case 中**全部 105/105**有 `(assistant 无 tool_calls) → (assistant)` 的连续形态——这是 conclusion check 触发后 agent 收到不可见 HumanMessage 的间接指纹

**影响**：
- 每个 case 的 conclusion 干预内容在 trajectory 里完全不可审计
- 只能通过"下一条 assistant 是否带 tool_calls"间接判断 agent 是否听话（22 个 back_to_tools / 83 个 rewrite）
- 用户希望分析 "中间件具体提了什么" 时，无法回答

**修复方案**：把 conclusion 检查从 conditional edge 重构成一个真正的图节点 `pre_conclusion_check`：

```python
# 新增节点 - return value 走正常 reducer
def pre_conclusion_check_node(state: ResearcherState):
    intervention = pipeline.check_before_conclusion()
    if intervention:
        return {"researcher_messages": [HumanMessage(content=intervention)]}
    return {}

# 把 should_continue_mw 拆成两个：
# (1) llm_call 后判断有没有 tool_calls
# (2) 如果没有，进入 pre_conclusion_check_node，节点判断要不要 conclude
def should_continue_mw(state):
    if state["researcher_messages"][-1].tool_calls:
        return "tool_node"
    return "pre_conclusion_check"

def after_pre_conclusion(state):
    # 新增的节点跑完后判断要不要回 llm_call
    last = state["researcher_messages"][-1]
    if isinstance(last, HumanMessage) and "Investigation Advisor" in last.content:
        return "llm_call"
    return "compress_research"

builder.add_node("pre_conclusion_check", pre_conclusion_check_node)
builder.add_conditional_edges("llm_call", should_continue_mw, {...})
builder.add_conditional_edges("pre_conclusion_check", after_pre_conclusion, {...})
```

修完后 trajectory 会自然包含 conclusion 干预的完整文本，所有间接的 loopback 启发式都可以删除。

### Bug 2 — `_run_conclusion_cycle` 是死代码

**位置**：[`pipeline.py:193-212`](pipeline.py#L193-L212)

`check_before_conclusion()` 直接返回硬编码的 `_CONCLUSION_PROMPT`（[pipeline.py:108-141](pipeline.py#L108-L141)），**根本不调用** `_run_conclusion_cycle()` / `ConclusionDetector` / L3 generator。

后果：
- v3 的 conclusion 检查**没有动态检测**，每次都注入完全相同的 9 条目固定文本
- `ConclusionDetector` 类里的 M1/M2/M3/M4 维度从来不会被 LLM 评估
- L3 `InterventionGenerator` 在 conclusion 阶段从来不被调用

**修复方案**：要么把 `_run_conclusion_cycle()` 真正接进 `check_before_conclusion()`，要么删掉 `ConclusionDetector` 和 `_run_conclusion_cycle()` 让代码意图清晰。

### Bug 3 — v3 检查点 [37, 44] 对 42% 的 case 太晚

**实测**：105 case 中有 **44 个 (42%) qpf < 37**，process 检查永远没机会触发。这正是 [middleware/CLAUDE.md 上文](#过早收敛)预期由 conclusion check 兜底的"过早收敛"群体——但因为 Bug 1，conclusion 干预不可观测，无法验证它实际有没有用。

**已有数据反推**：
- 21 个 wrong→correct 的"silent process" case 中，9 个 conclusion=back_to_tools（agent 真去查了证据），12 个 rewrite（agent 没动作）
- back_to_tools 那 9 个**强烈暗示 conclusion check 起到了挽救作用**——但需要修 Bug 1 才能直接观测

**修复选项**（三选一或组合）：

| 方案 | 描述 | 代价 |
|---|---|---|
| (a) 提前增加 query 20 检查点 | `check_points = [20, 37, 44]`，但 query 20 时正确 case 还有 ~95% 在跑，误伤率会很高 | 低实现成本，高 LLM 调用成本 |
| (b) 检查点改成相对终点而非绝对 query 数 | 监控 agent 的 think_tool reflection，发现 "I have sufficient evidence" 类语句时立即触发 | 中等实现成本，需要做关键词匹配 |
| (c) 修 Bug 1 让 conclusion check 内容可见 + 把 conclusion check 升级成动态 | 让 conclusion check 真的跑 `_run_conclusion_cycle()` 而不是固定文本 | 中等成本，从根本上解决"过早收敛"分支 |

**推荐 (a)+(c) 组合**：先修 Bug 1 让 conclusion 可观测，再加 query 20 检查点弥补长尾。

---

## v3 已知失败模式（来自 105 case 分析）

详细 case 见[`MW-vs-w_o-MW.md`](../../analysis/4-middleware/MW-vs-w_o-MW.md) 的 batch 分析。这里只列总结。

### 干预条目无法对抗的反模式

#### M3-recurring："消失即根因" (Empty-Service Hypothesis)
- 表现：agent 把 "missing from abnormal_traces" / "metric is NaN" 当作根因证据
- 出现在：case 341, 1934 等
- 当前 MW 缺口：B/M 检测维度里没有专门项；M3 (Absence ≠ Health) 只在 conclusion detector 出现，且 conclusion detector 是死代码 (Bug 2)
- 建议新维度 **B6: Empty-Service Hypothesis**：当 agent 把 "X is missing/NaN/empty" 作为根因证据时给出反向提醒

#### R5 太强 / R6 锚定回退
- 表现：MW 干预正确指出方向，agent 形式上响应（多查几条），但思考方向不变
- 出现在：case 579, 4375 等
- 当前 MW 缺口：现有 prompt 偏"提醒思考"，没有"强制重新评估早期假设"的强干预
- 建议：在多次 B1 触发同一服务后升级为强提示——明确要求 agent 列出"如果 X 被排除会怎样"

#### B3 引发 baseline 重复 → B1 又开火（自循环）
- 表现：B3 让 agent 大量做 baseline_collect，B1 检测到这些为"重复"再次开火
- 出现在：case 1495
- 修复：让 B1 的检测器排除"baseline 类意图"，或者在 B3 触发后给 baseline 重复 5 次的宽限期

### Conclusion rewrite 翻正之谜（12 个 case）

12 个 wrong→correct 的 case 没有任何 process advisor，conclusion check 是 `rewrite` 模式（agent 没多查任何 tool），但答案就是从错变对了。可能解释：

1. **L1 分类器的 background LLM 调用扰动 qwen3.5 的 token 采样**（最可能）
2. conclusion prompt 中的 R6 锚定 / R7 幸存者偏差提示让 agent 重新看了同一份证据
3. 随机性

**统计意义上不应算作 MW 真实增益**——D 段建议把这 12 个 case 从"MW 救回率"里剔除，得到更保守的 49-12 = 37/105 = 35% 真实救回率。

---

## v4 改进路线图

按优先级排序（修 bug 优先于加功能）：

| 优先级 | 项目 | 类型 | 预期收益 |
|---|---|---|---|
| **P0** | 修 Bug 1（conclusion check 改成节点） | bugfix | 让 conclusion 干预可审计，所有间接启发式可删 |
| **P0** | 决定 Bug 2 处理（接通或删除 `ConclusionDetector`） | bugfix | 代码意图清晰，要么用上动态检测要么减少死代码 |
| **P1** | 加 query 20 早期检查点 | feature | 覆盖 42% qpf<37 case |
| **P1** | 新增 B6: Empty-Service Hypothesis 维度 | feature | 解决 M3-recurring 反模式 |
| **P2** | B1 检测器排除 baseline 类意图，避免与 B3 自循环 | bugfix | 解决 case 1495 类自循环 |
| **P2** | 升级 R5/R6 高反复案例的强干预（"列出反例假设" 类） | feature | 突破 R6 锚定回退的厚墙 |
| **P3** | 区分 L1 扰动效应 vs MW 真实救回（实验设计层面） | analysis | 真实 MW 增益统计可信 |

---

## 关键文件

| 文件 | 作用 |
|------|------|
| `middleware/config.py` | 配置（检查点、上限、维度开关） |
| `middleware/pipeline.py` | 三层管线入口，检查点触发逻辑 |
| `middleware/deficiency_detector.py` | L2 双检测器（ProcessDetector + ConclusionDetector） |
| `middleware/intent_classifier.py` | L1 SQL 意图分类 |
| `middleware/intervention_generator.py` | L3 干预文本生成 |
| `middleware/state.py` | 调查状态追踪（意图序列、推理日志、阶段覆盖） |

---

## 版本变更

| 版本 | 日期 | 变更 |
|------|------|------|
| v1 | 2026-04-09 | 初版：D1-D5 维度，固定间隔 check_interval=5 |
| v2 | 2026-04-10 | D×R 对齐重构：B1/B2/B3/B5/M1-M4，双时机检测，维度去重 |
| v3 | 2026-04-11 | 检查点优化：固定间隔 → 数据驱动检查点 [37,44]，max_interventions 5→3 |
| v3-doc | 2026-04-14 | 105 case 实跑分析；记录 3 个 v3 已知 bug；发布 v4 路线图 |
