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
