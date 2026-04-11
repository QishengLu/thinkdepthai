#!/usr/bin/env python
"""
agent_runner.py — ThinkDepthAI RCA 测评接口

stdin:  JSON { question, system_prompt, user_prompt,
               compress_system_prompt, compress_user_prompt, data_dir }
stdout: JSON { output (CausalGraph JSON), trajectory (OpenAI 格式) }
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/nn/SOTA-agents/RolloutRunner")
from src.usage_tracker import UsageTracker

_tracker = UsageTracker()
_tracker.install_openai_hooks()

# 根据模型选择 hook：Claude 走 Anthropic SDK，其余走 OpenAI SDK
_RCA_MODEL = os.environ.get("RCA_MODEL", "claude-sonnet-4-6")
if _RCA_MODEL.startswith("claude"):
    _tracker.install_anthropic_hooks()

# 清理 RolloutRunner 路径和 src 模块缓存，避免与本项目的 src 包冲突
sys.path.remove("/home/nn/SOTA-agents/RolloutRunner")
for _mod in list(sys.modules):
    if _mod == "src" or _mod.startswith("src."):
        del sys.modules[_mod]


from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from model_factory import create_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)
from langgraph.graph import END, START, StateGraph
from typing_extensions import Literal

from deep_research.prompts import rca_think_prompt
from deep_research.state_research import ResearcherOutputState, ResearcherState
from deep_research.utils import think_tool
from deep_research.rca_tools import get_schema, list_tables_in_directory, query_parquet_files
from middleware import MiddlewareConfig, MiddlewarePipeline

# ── 工具集（去掉 tavily_search，与 RCA_ANALYSIS_SP 描述一致）─────────────────
RCA_TOOLS = [think_tool, list_tables_in_directory, get_schema, query_parquet_files]
RCA_TOOLS_BY_NAME = {t.name: t for t in RCA_TOOLS}


# ── 节点工厂（闭包注入 prompt/tools，保持原拓扑）─────────────────────────────

RCA_MODEL = os.environ.get("RCA_MODEL", "claude-sonnet-4-6")


def _make_model(max_tokens: int = 32768):
    """Create LLM via model_factory. Model name from RCA_MODEL env var."""
    return create_model(RCA_MODEL, max_tokens=max_tokens)


def make_llm_call(combined_system_prompt: str):
    model = _make_model()
    model_with_tools = model.bind_tools(RCA_TOOLS)

    def llm_call(state: ResearcherState):
        return {
            "researcher_messages": [
                model_with_tools.invoke(
                    [SystemMessage(content=combined_system_prompt)]
                    + state["researcher_messages"]
                )
            ]
        }

    return llm_call


def tool_node(state: ResearcherState):
    tool_calls = state["researcher_messages"][-1].tool_calls
    outputs = []
    for tc in tool_calls:
        tool = RCA_TOOLS_BY_NAME[tc["name"]]
        result = tool.invoke(tc["args"])
        outputs.append(ToolMessage(content=str(result), name=tc["name"], tool_call_id=tc["id"]))
    return {"researcher_messages": outputs}


def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    return "tool_node" if state["researcher_messages"][-1].tool_calls else "compress_research"


def make_compress_research(compress_sp: str, compress_up: str):
    compress_model = _make_model(max_tokens=32000)

    def compress_research(state: ResearcherState) -> dict:
        messages = (
            [SystemMessage(content=compress_sp)]
            + state.get("researcher_messages", [])
            + [HumanMessage(content=compress_up)]
        )
        response = compress_model.invoke(messages)
        raw_notes = [
            str(m.content)
            for m in filter_messages(
                state["researcher_messages"], include_types=["tool", "ai"]
            )
        ]
        return {
            "compressed_research": str(response.content),
            "raw_notes": ["\n".join(raw_notes)],
        }

    return compress_research


def build_agent(combined_sp: str, compress_sp: str, compress_up: str):
    """与原始 research_agent.py 完全相同的图拓扑，仅 prompt/tools 不同。"""
    builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
    builder.add_node("llm_call", make_llm_call(combined_sp))
    builder.add_node("tool_node", tool_node)
    builder.add_node("compress_research", make_compress_research(compress_sp, compress_up))
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {"tool_node": "tool_node", "compress_research": "compress_research"},
    )
    builder.add_edge("tool_node", "llm_call")
    builder.add_edge("compress_research", END)
    return builder.compile()


def build_agent_with_middleware(
    combined_sp: str, compress_sp: str, compress_up: str,
    pipeline: MiddlewarePipeline,
):
    """Same graph topology as build_agent, but with middleware hooks."""

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
        assistant_content = ""
        for m in reversed(messages):
            if hasattr(m, "content") and hasattr(m, "tool_calls"):
                assistant_content = m.content or ""
                break

        intervention = pipeline.process_tool_calls(
            tool_calls, round_num, assistant_content
        )
        if intervention:
            outputs.append(HumanMessage(content=intervention))

        return {"researcher_messages": outputs}

    def should_continue_mw(state: ResearcherState) -> Literal["tool_node", "compress_research", "llm_call"]:
        if state["researcher_messages"][-1].tool_calls:
            return "tool_node"
        intervention = pipeline.check_before_conclusion()
        if intervention:
            state["researcher_messages"].append(HumanMessage(content=intervention))
            return "llm_call"
        return "compress_research"

    builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
    builder.add_node("llm_call", make_llm_call(combined_sp))
    builder.add_node("tool_node", tool_node_mw)
    builder.add_node("compress_research", make_compress_research(compress_sp, compress_up))
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call",
        should_continue_mw,
        {"tool_node": "tool_node", "compress_research": "compress_research", "llm_call": "llm_call"},
    )
    builder.add_edge("tool_node", "llm_call")
    builder.add_edge("compress_research", END)
    return builder.compile()


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def strip_markdown_json(text: str) -> str:
    """剥离 LLM 返回的 ```json ... ``` 代码块，提取纯 JSON。"""
    import re
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


# ── LangChain → OpenAI 格式转换 ──────────────────────────────────────────────

def to_openai_message(msg) -> dict | None:
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": str(msg.content)}

    if isinstance(msg, AIMessage):
        tool_calls = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"], ensure_ascii=False),
                },
            }
            for tc in (msg.tool_calls or [])
        ]
        # Anthropic SDK returns content as list of blocks; extract text parts
        content = msg.content
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) and b.get("type") == "text" else ""
                for b in content
            ).strip()
        entry: dict = {"role": "assistant", "content": str(content) if content else ""}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        return entry

    if isinstance(msg, ToolMessage):
        return {"role": "tool", "content": str(msg.content), "tool_call_id": msg.tool_call_id}

    return None


def convert_trajectory(messages: list) -> list[dict]:
    return [m for msg in messages if (m := to_openai_message(msg)) is not None]


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable real-time step logging to stderr")
    parser.add_argument("--log-file", default=None,
                        help="Write verbose logs to file (implies --verbose)")
    args, _ = parser.parse_known_args()

    logger = logging.getLogger("agent_runner")
    if args.log_file or args.verbose:
        handlers = []
        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        if args.verbose:
            h = logging.StreamHandler(sys.stderr)
            h.setFormatter(fmt)
            handlers.append(h)
        if args.log_file:
            os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
            h = logging.FileHandler(args.log_file, mode="w")
            h.setFormatter(fmt)
            handlers.append(h)
        logging.basicConfig(level=logging.INFO, handlers=handlers)
    else:
        logging.basicConfig(level=logging.WARNING)

    payload = json.loads(sys.stdin.read())

    system_prompt = payload["system_prompt"]
    user_prompt = payload["user_prompt"]
    compress_sp = payload["compress_system_prompt"]
    compress_up = payload["compress_user_prompt"]
    data_dir = payload.get("data_dir", "")

    # data_dir 追加到 user_prompt，让 agent 知道从哪里开始探索数据
    if data_dir:
        user_prompt = f"{user_prompt}\n\n## Data Location\n\nThe telemetry data for this incident is located at: `{data_dir}`\n\nStart by calling `list_tables_in_directory(directory=\"{data_dir}\")` to discover available parquet files."

    # RCA 领域指令在前（主体），反思方法论在后（补充）
    combined_sp = system_prompt + "\n\n---\n\n" + rca_think_prompt

    mw_config = MiddlewareConfig()
    if mw_config.enabled:
        mw_model = _make_model(max_tokens=512)
        pipeline = MiddlewarePipeline(mw_model, mw_config)
        agent = build_agent_with_middleware(combined_sp, compress_sp, compress_up, pipeline)
    else:
        agent = build_agent(combined_sp, compress_sp, compress_up)

    initial_state = {"researcher_messages": [HumanMessage(content=user_prompt)]}

    all_messages: list = []
    compressed_research = ""
    step_num = 0

    for event in agent.stream(initial_state, config={"recursion_limit": 300}):
        for key, value in event.items():
            if not isinstance(value, dict):
                continue
            if "researcher_messages" in value:
                new_msgs = value["researcher_messages"]
                all_messages.extend(new_msgs)
                step_num += 1
                for m in new_msgs:
                    if isinstance(m, AIMessage):
                        tc_names = [tc["name"] for tc in (m.tool_calls or [])]
                        content_preview = str(m.content)[:120] if m.content else ""
                        if tc_names:
                            logger.info(f"[step {step_num}] {key}: AI → tool_calls={tc_names}")
                        else:
                            logger.info(f"[step {step_num}] {key}: AI (no tools) → {content_preview}...")
                    elif isinstance(m, ToolMessage):
                        logger.info(f"[step {step_num}] {key}: Tool({m.name}) → {str(m.content)[:100]}...")
                    elif isinstance(m, HumanMessage):
                        logger.info(f"[step {step_num}] {key}: MW intervention → {str(m.content)[:120]}...")
            if "compressed_research" in value:
                compressed_research = value["compressed_research"]
                logger.info(f"[step {step_num}] compress_research done, len={len(compressed_research)}")

    # Usage 采集：Claude 走 UsageTracker（Anthropic SDK hook），其余从 LangChain response metadata 采集
    if RCA_MODEL.startswith("claude"):
        usage = _tracker.get_usage()
    else:
        # 非 Claude 模型：从 AIMessage.usage_metadata 累加（LangChain 自动解析 OpenAI SDK response）
        _total_in, _total_out, _llm_calls = 0, 0, 0
        for msg in all_messages:
            if isinstance(msg, AIMessage) and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                um = msg.usage_metadata
                _total_in += um.get("input_tokens", 0)
                _total_out += um.get("output_tokens", 0)
                _llm_calls += 1
        usage = {
            "total_tokens": _total_in + _total_out,
            "prompt_tokens": _total_in,
            "completion_tokens": _total_out,
            "reasoning_tokens": 0,
            "llm_call_count": _llm_calls,
        }

    result = {
        "output": strip_markdown_json(compressed_research),
        "trajectory": convert_trajectory(all_messages),
        "usage": usage,
    }
    # 单行输出，runner._parse_last_json 从末行解析
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
