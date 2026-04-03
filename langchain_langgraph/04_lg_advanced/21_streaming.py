import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
主题：Streaming —— 实时追踪图的每一步执行

学习目标：
  1. 理解为什么需要流式输出（用户体验、调试）
  2. 掌握 stream(mode="updates")：每步更新的字典
  3. 掌握 stream(mode="values")：每步后的完整 State
  4. 掌握异步流式 astream_events：token 级别追踪
  5. 理解不同流式模式的适用场景

核心概念：
  stream_mode="updates" → 只返回本步节点修改的字段
  stream_mode="values"  → 返回整个 State（包含所有字段）
  astream_events        → 异步事件流，可追踪每个 token

  适用场景：
  "updates" → 调试，了解每个节点做了什么
  "values"  → 监控，了解全局状态变化
  astream_events → 前端实时显示 AI 输出（流式打字效果）

前置知识：已完成 13_hello_langgraph.py
"""

import os
import asyncio
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# 初始化 LLM
llm = ChatAnthropic(model="ppio/pa/claude-sonnet-4-6")

print("=" * 60)
print("LangGraph 21：Streaming —— 实时追踪图执行")
print("=" * 60)


# ============================================================
# 构建一个 3 节点的分析图（供 Part 1 和 Part 2 使用）
# ============================================================

class AnalysisState(TypedDict):
    """分析图的 State"""
    question: str       # 用户问题
    category: str       # 问题分类
    answer: str         # 详细回答
    formatted: str      # 格式化输出


def classify_node(state: AnalysisState) -> dict:
    """分类节点：判断问题属于哪个类别"""
    response = llm.invoke(
        f"把以下问题归类为：技术/科学/文化/其他，只输出类别名称：{state['question']}"
    )
    category = response.content.strip()
    print(f"    [classify] 分类结果：{category}")
    return {"category": category}


def answer_node(state: AnalysisState) -> dict:
    """回答节点：根据分类提供详细回答"""
    response = llm.invoke(
        f"这是一个关于{state['category']}的问题，请简洁回答（50字内）：{state['question']}"
    )
    print(f"    [answer] 回答完成（{len(response.content)} 字）")
    return {"answer": response.content}


def format_node(state: AnalysisState) -> dict:
    """格式化节点：包装最终输出"""
    formatted = (
        f"【{state['category']}】\n"
        f"问题：{state['question']}\n"
        f"回答：{state['answer']}"
    )
    print(f"    [format] 格式化完成")
    return {"formatted": formatted}


# 构建分析图
analysis_graph = StateGraph(AnalysisState)
analysis_graph.add_node("classify", classify_node)
analysis_graph.add_node("answer", answer_node)
analysis_graph.add_node("format", format_node)
analysis_graph.add_edge(START, "classify")
analysis_graph.add_edge("classify", "answer")
analysis_graph.add_edge("answer", "format")
analysis_graph.add_edge("format", END)
analysis_app = analysis_graph.compile()

analysis_input = {
    "question": "Python 为什么适合数据科学？",
    "category": "",
    "answer": "",
    "formatted": ""
}


# ============================================================
# Part 1：stream mode="updates"
# ============================================================
print("\n--- Part 1：stream(mode='updates') ---")
print("说明：每步只返回本节点修改的字段（增量更新）")
print("适用场景：调试，了解每个节点具体修改了哪些字段")

print(f"\n[问题] {analysis_input['question']}")
print("[逐步执行，每步输出 updates]")

step_count = 0
for chunk in analysis_app.stream(analysis_input, stream_mode="updates"):
    step_count += 1
    # chunk 格式：{node_name: {field: value, ...}}
    for node_name, updates in chunk.items():
        print(f"\n  Step {step_count} - 节点「{node_name}」的更新：")
        for field, value in updates.items():
            value_preview = str(value)[:60] if value else "(空)"
            print(f"    {field}: {value_preview}")

print(f"\n共执行 {step_count} 步")


# ============================================================
# Part 2：stream mode="values"
# ============================================================
print("\n--- Part 2：stream(mode='values') ---")
print("说明：每步后返回完整 State（所有字段的当前值）")
print("适用场景：监控，了解全局状态如何演变")

print(f"\n[问题] {analysis_input['question']}")
print("[逐步执行，每步输出完整 State]")

step_count = 0
for full_state in analysis_app.stream(analysis_input, stream_mode="values"):
    step_count += 1
    # full_state 是完整的 State 字典
    print(f"\n  Step {step_count} - 完整 State 快照：")
    for field, value in full_state.items():
        if value:  # 只显示非空字段
            value_preview = str(value)[:60]
            print(f"    {field}: {value_preview}")

print(f"\n共 {step_count} 个 State 快照（初始状态 + 每个节点执行后）")


# ============================================================
# Part 3：astream_events（token 级别流式追踪）
# ============================================================
print("\n--- Part 3：astream_events（token 级别追踪）---")
print("说明：异步事件流，可以逐个 token 接收 LLM 输出")
print("适用场景：前端实时显示 AI 打字效果")

# 为 Part 3 创建一个简单的聊天图
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chat_node(state: ChatState) -> dict:
    """简单聊天节点"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


chat_graph = StateGraph(ChatState)
chat_graph.add_node("chat", chat_node)
chat_graph.add_edge(START, "chat")
chat_graph.add_edge("chat", END)
chat_app = chat_graph.compile()


async def stream_tokens(app, input_state: dict):
    """异步函数：逐 token 流式输出 LLM 响应"""
    print("[token 流] ", end="", flush=True)
    token_count = 0
    async for event in app.astream_events(input_state, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)
                token_count += 1
    print()  # 换行
    return token_count


chat_input = {"messages": [HumanMessage(content="简述 Python 的主要优点，不超过 60 字")]}
print(f"\n[问题] {chat_input['messages'][0].content}")
print("[astream_events 流式输出]")
token_count = asyncio.run(stream_tokens(chat_app, chat_input))
print(f"[共接收 {token_count} 个 token 事件]")


def main():
    print("\n--- 三种流式模式对比 ---")
    print()
    print(f"{'模式':<20} {'返回内容':<25} {'适用场景'}")
    print("-" * 70)
    print(f"{'stream(updates)':<20} {'节点修改的字段增量':<25} {'调试：了解每步做了什么'}")
    print(f"{'stream(values)':<20} {'每步后的完整 State':<25} {'监控：了解状态如何变化'}")
    print(f"{'astream_events':<20} {'异步事件（含每个token）':<25} {'前端：实时打字效果'}")
    print()
    print("注意：astream_events 需要在 async 函数中使用 async for")
    print("      在同步代码中用 asyncio.run() 调用异步函数")


if __name__ == "__main__":
    main()
