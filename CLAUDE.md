# CLAUDE.md — Deep Research / RCA Agent

## 项目概述

基于 LangGraph 构建的多智能体深度研究与根因分析（RCA）系统。支持两种运行模式：
1. **单 Researcher Agent**：适合 RCA 任务（读取本地 parquet 数据 + 网络搜索）
2. **Supervisor + Researcher 多智能体**：适合开放式深度研究，Supervisor 并行调度多个 Researcher

---

## 入口文件

| 文件 | 作用 |
|------|------|
| `thinkdepthai_rca.py` | **顶层入口**。读取 `data/problem.json`，生成 `task.json`，通过子进程调用 `uv run python run_rca.py` |
| `run_rca.py` | 实际运行器。读取 `task.json`，调用 `researcher_agent.stream()`，结果写入 `experiments/output.json` |
| `thinkdepthai_deepresearch.ipynb` | 交互式 Jupyter 入口，供手动调试使用 |

**任务描述从哪里传入：**
`data/problem.json` → `thinkdepthai_rca.py` 解析 → `task.json` → `run_rca.py` 读取 → 包装为 `HumanMessage` → 作为初始 `researcher_messages` 注入 `researcher_agent`

---

## Agent 核心图结构

### 1. Researcher Agent（`src/research_agent.py`）

```
START
  └─► llm_call          # 调用 model_with_tools，注入 research_agent_prompt 作为 SystemMessage
        ├─► tool_node   # 执行所有 tool_calls（tavily_search / think_tool / rca_tools）
        │     └─► llm_call   # 循环直到无 tool_calls
        └─► compress_research  # 压缩研究结果为 compressed_research 字段
              └─► END
```

路由逻辑（`should_continue`）：last message 有 `tool_calls` → `tool_node`，否则 → `compress_research`

### 2. Supervisor 多智能体（`src/multi_agent_supervisor.py`）

```
START
  └─► supervisor        # Supervisor LLM 决策：调用 ConductResearch / think_tool / refine_draft_report / ResearchComplete
        └─► supervisor_tools
              ├─► [并行] researcher_agent.ainvoke(×N)   # 每个 ConductResearch 调用启动一个独立 Researcher
              ├─► think_tool（同步）
              ├─► refine_draft_report（同步）
              └─► supervisor（继续循环）或 END（ResearchComplete / 超出迭代限制 / 无 tool_calls）
```

关键常量（`multi_agent_supervisor.py`）：
- `max_researcher_iterations = 15`（Supervisor 的工具调用总预算）
- `max_concurrent_researchers = 3`（单次并行 Researcher 上限）

---

## Prompt 定义与注入位置

所有 prompt 字符串定义在 **`src/prompts.py`**。

| Prompt 变量名 | 用途 | 注入位置 |
|--------------|------|---------|
| `research_agent_prompt` | Researcher 的系统提示 | `research_agent.py::llm_call()` → `SystemMessage(content=research_agent_prompt)` |
| `lead_researcher_with_multiple_steps_diffusion_double_check_prompt` | Supervisor 的系统提示（含 Diffusion Algorithm 指令） | `multi_agent_supervisor.py::supervisor()` → `.format(date=..., max_concurrent_research_units=..., max_researcher_iterations=...)` |
| `compress_research_system_prompt` | 压缩研究结果的系统提示 | `research_agent.py::compress_research()` → `.format(date=...)` |
| `compress_research_human_message` | 压缩节点的 Human 消息 | `research_agent.py::compress_research()` |
| `summarize_webpage_prompt` | 网页内容摘要 | `utils.py::summarize_webpage_content()` |
| `report_generation_with_draft_insight_prompt` | 迭代精炼草稿报告 | `utils.py::refine_draft_report()` |
| `final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt` | 最终报告生成（供 notebook 使用） | `thinkdepthai_deepresearch.ipynb` |

> **注意**：`research_agent_prompt` 含 `{date}` 占位符，但当前 `llm_call()` 直接使用字符串而未调用 `.format()`，日期不会自动填充。

---

## Tools 加载方式

### Researcher Agent 工具（`src/research_agent.py`）

```python
tools = [tavily_search, think_tool, list_tables_in_directory, get_schema, query_parquet_files]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
```

| Tool | 来源 | 功能 |
|------|------|------|
| `tavily_search` | `src/utils.py` | Tavily 网络搜索 + 网页摘要 |
| `think_tool` | `src/utils.py` | 反思占位工具（返回记录的思考内容） |
| `list_tables_in_directory` | `src/rca_tools.py` | 列出目录中所有 parquet 文件及元数据 |
| `get_schema` | `src/rca_tools.py` | 获取单个 parquet 文件的 schema |
| `query_parquet_files` | `src/rca_tools.py` | 用 DuckDB SQL 查询 parquet 数据，默认 token 限制 5000 |

### Supervisor 工具（`src/multi_agent_supervisor.py`）

```python
supervisor_tools = [ConductResearch, ResearchComplete, think_tool, refine_draft_report]
```

| Tool | 来源 | 功能 |
|------|------|------|
| `ConductResearch` | `src/state_multi_agent_supervisor.py` | 派发研究子任务给 Researcher Agent |
| `ResearchComplete` | `src/state_multi_agent_supervisor.py` | 标记研究完成，触发 END |
| `think_tool` | `src/utils.py` | 反思规划 |
| `refine_draft_report` | `src/utils.py` | 基于最新 findings 迭代精炼草稿报告 |

---

## 环境管理

**工具：`uv`**（见 `pyproject.toml` + `uv.lock`）

```bash
# 安装依赖
uv sync

# 直接运行脚本（uv 自动激活虚拟环境）
uv run python run_rca.py
```

Python 要求：`>=3.11`

主要依赖：`langgraph`, `langchain`, `langchain-openai`, `langchain-anthropic`, `tavily-python`, `duckdb`, `python-dotenv`

---

## 环境变量

在项目根目录创建 `.env` 文件（参考现有 `.env`）：

```
OPENAI_API_KEY=sk-...       # 必填，驱动所有 LLM 调用（当前模型：openai:gpt-5）
TAVILY_API_KEY=tvly-...     # 必填，驱动 tavily_search 工具（留空则搜索静默失败）
```

> **安全提示**：`.env` 中不应提交真实密钥到版本库。当前 `.env` 含明文密钥，请检查 `.gitignore`。

---

## 运行完整 Agent 的命令

### 方式一：标准入口（推荐）

```bash
# 1. 编辑任务描述（problem.json 内的三引号字符串）
vim data/problem.json

# 2. 运行
uv run python thinkdepthai_rca.py
```

输出保存至：`experiments/output.json`

### 方式二：直接调用 runner（跳过 problem.json 解析）

```bash
# 先确保 task.json 存在且包含 task_description 字段
uv run python run_rca.py
```

### 方式三：Jupyter Notebook

```bash
uv run jupyter notebook thinkdepthai_deepresearch.ipynb
```

---

## RCA 任务数据流

### 第一阶段：任务描述的传递

```
data/problem.json
  │  内容："""任务描述文本"""（三引号包裹）
  │
  ▼
thinkdepthai_rca.py
  │  re.search(r'"""(.*?)"""') 提取纯文本
  │  写入 task.json → {"task_description": "..."}
  │  subprocess.run(["uv", "run", "python", "run_rca.py"])
  │
  ▼
run_rca.py
  │  json.load(task.json) → task_description
  │  包装：HumanMessage(content=task_description)
  │  构造初始 state：
  │    {"researcher_messages": [HumanMessage(task_description)]}
  │
  ▼
researcher_agent.stream(initial_state, config={"recursion_limit": 100})
```

### 第二阶段：Researcher Agent 图内部循环

```
ResearcherState {
  researcher_messages: [HumanMessage(task_description)]  ← 初始
  compressed_research: ""
  raw_notes: []
}

┌─────────────────────────────────────────────────────┐
│  NODE: llm_call                                     │
│                                                     │
│  输入：                                              │
│    [SystemMessage(research_agent_prompt)]            │
│    + state["researcher_messages"]                   │
│                                                     │
│  调用：model_with_tools.invoke(messages)             │
│  模型：openai:gpt-5（已 bind 5 个 tools）            │
│                                                     │
│  输出：AIMessage（含 tool_calls 或不含）              │
│  → append 到 researcher_messages                    │
└─────────────┬───────────────────────────────────────┘
              │
              ▼ should_continue()
    last_message.tool_calls 是否为空？
              │
      ┌───────┴───────┐
     有              无
      │               │
      ▼               ▼
  tool_node    compress_research
```

**tool_node 内部**（RCA 场景下的典型调用链）：

```
AIMessage.tool_calls = [
  {name: "list_tables_in_directory", args: {"directory": "data"}},
  ...
]

逐个执行：
  list_tables_in_directory("data")
    └─ 扫描 data/*.parquet，返回文件名/行数/列数 JSON

  get_schema("data/abnormal_metrics.parquet")
    └─ DuckDB DESCRIBE → 列名+类型 JSON

  query_parquet_files(
      parquet_files=["data/abnormal_metrics.parquet"],
      query="SELECT ...",
      limit=10
  )
    └─ DuckDB 内存 DB → CREATE VIEW → 执行 SQL
    └─ token 超限时返回建议而非数据

  tavily_search(query="...")
    └─ TavilyClient.search → 去重 → 摘要模型处理网页 → 格式化字符串

  think_tool(reflection="...")
    └─ 直接返回 "Reflection recorded: ..."（无副作用）

每个结果包装为 ToolMessage(content=..., name=..., tool_call_id=...)
→ append 到 researcher_messages → 回到 llm_call
```

循环收敛：当模型判断已有足够信息，输出无 `tool_calls` 的 AIMessage → 进入 `compress_research`

### 第三阶段：研究结果压缩

```
NODE: compress_research

输入：state["researcher_messages"]（完整对话历史）

构造摘要请求：
  [SystemMessage(compress_research_system_prompt.format(date=...))]
  + state["researcher_messages"]
  + [HumanMessage(compress_research_human_message)]

compress_model.invoke(messages)
  → 过滤掉 think_tool 的内部反思
  → 保留 tavily_search 结果 + DuckDB 查询结果
  → 输出结构化报告（含内联引用）

raw_notes = 所有 tool/ai 消息的 content 拼接

输出 state 更新：
  compressed_research: str  ← 压缩后的完整研究报告
  raw_notes: [str]          ← 原始详细记录
```

### 第四阶段：结果写出

```
run_rca.py 收集 stream 事件

researcher_messages 中每条消息序列化为：
  {
    "type": "human"/"ai"/"tool",
    "content": "...",
    "tool_calls": [...],    ← AIMessage 有 tool_calls 时
    "tool_call_id": "...",  ← ToolMessage 有
    "name": "..."           ← ToolMessage 有
  }

追加压缩结果：
  {"type": "compressed_research", "content": compressed_research}

写入 experiments/output.json
```

### 状态字段全程变化示意

```
时间点          researcher_messages 内容
─────────────────────────────────────────────────────────────────────
初始           [HumanMessage("分析...异常的根因")]
llm_call #1    [..., AIMessage(tool_calls=[list_tables, get_schema])]
tool_node #1   [..., ToolMessage(文件列表), ToolMessage(schema)]
llm_call #2    [..., AIMessage(tool_calls=[query_parquet_files, think_tool])]
tool_node #2   [..., ToolMessage(SQL结果), ToolMessage("Reflection recorded:...")]
llm_call #3    [..., AIMessage(tool_calls=[query_parquet_files])]
tool_node #3   [..., ToolMessage(SQL结果)]
llm_call #4    [..., AIMessage(无 tool_calls)]  ← 模型判断信息充足
compress       compressed_research = "## 根因分析报告..."
```

### 关键约束

| 约束 | 配置位置 | 值 |
|------|---------|---|
| prompt 中建议的搜索次数上限 | `src/prompts.py::research_agent_prompt` | 复杂查询 ≤5 次 |
| LangGraph 递归限制 | `run_rca.py` stream config | 100 |
| DuckDB 结果 token 限制 | `src/rca_tools.py::TOKEN_LIMIT` | 5000 tokens |
| Tavily 单次结果数 | `src/utils.py::tavily_search` 默认参数 | max_results=3 |

---

## 关键文件索引

```
Deep_Research/
├── thinkdepthai_rca.py          # 顶层入口
├── run_rca.py                   # 实际运行器
├── data/
│   ├── problem.json             # 任务描述输入
│   └── *.parquet                # RCA 所需的观测数据（metrics/traces/logs）
├── experiments/
│   └── output.json              # 运行结果输出
├── src/
│   ├── research_agent.py        # Researcher Agent 图定义 + 工具绑定
│   ├── multi_agent_supervisor.py # Supervisor 图定义
│   ├── prompts.py               # 所有 prompt 字符串
│   ├── utils.py                 # tavily_search, think_tool, refine_draft_report
│   ├── rca_tools.py             # DuckDB parquet 工具
│   ├── state_research.py        # ResearcherState / ResearcherOutputState
│   └── state_multi_agent_supervisor.py  # SupervisorState / ConductResearch / ResearchComplete
├── pyproject.toml               # 依赖声明（uv 管理）
└── .env                         # API 密钥（不提交到 git）
```
