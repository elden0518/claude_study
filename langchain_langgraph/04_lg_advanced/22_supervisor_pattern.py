import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
主题：Supervisor Pattern —— 集中式多 Agent 协调

学习目标：
  1. 理解 Supervisor 模式的核心思想：一个协调者调度多个专职 Agent
  2. 掌握把 Agent 包装成工具供 Supervisor 调用
  3. 理解 Supervisor 的决策逻辑（调用哪个 Agent or 结束）
  4. 学会追踪多 Agent 的调度过程
  5. 理解 Supervisor 模式的优缺点

核心概念：
  Supervisor（协调者）= 一个特殊的 LLM 节点，它不直接完成任务，
  而是决定"下一步应该调用哪个专职 Agent"

  专职 Agent = 只做一件事的 Agent（如：只搜索 / 只写作）

  Supervisor 的工具 = 专职 Agent 的名字列表
  Supervisor 调用工具 = 把控制权转给对应的专职 Agent

  优点：逻辑清晰，易于控制和调试
  缺点：Supervisor 是瓶颈，所有决策都经过它

前置知识：已完成 20_tool_node.py（工具调用）
"""

import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# 初始化 LLM
llm = ChatAnthropic(model="ppio/pa/claude-sonnet-4-6")

print("=" * 60)
print("LangGraph 22：Supervisor Pattern —— 集中式多 Agent 协调")
print("=" * 60)


# ============================================================
# Part 1：定义团队成员 Agent 和共享 State
# ============================================================
print("\n--- Part 1：定义多 Agent 团队的 State 和成员 ---")
print("说明：Supervisor 模式有一个协调者和多个专职 Agent")

# 团队成员名称
MEMBERS = ["researcher", "writer"]
FINISH = "FINISH"

class TeamState(TypedDict):
    """多 Agent 团队的共享 State"""
    messages: Annotated[list, add_messages]  # 消息历史（所有 Agent 共享）
    next: str       # Supervisor 决定下一步调用哪个 Agent
    topic: str      # 研究主题
    research: str   # 研究员收集的事实
    article: str    # 写作者生成的文章


def researcher_node(state: TeamState) -> dict:
    """专职研究员：收集关于主题的关键事实"""
    print(f"  [研究员] 开始研究主题：{state['topic']}")
    response = llm.invoke(
        f"请收集关于「{state['topic']}」的3个关键事实，每条一句话，用数字编号。"
    )
    print(f"  [研究员] 研究完成（{len(response.content)} 字）")
    return {
        "research": response.content,
        "messages": [AIMessage(content=f"[研究员] {response.content}", name="researcher")]
    }


def writer_node(state: TeamState) -> dict:
    """专职写作者：根据研究结果撰写介绍文章"""
    print(f"  [写作者] 开始基于研究结果写文章...")
    response = llm.invoke(
        f"根据以下事实，写一段不超过 150 字的介绍文章：\n{state['research']}"
    )
    print(f"  [写作者] 写作完成（{len(response.content)} 字）")
    return {
        "article": response.content,
        "messages": [AIMessage(
            content=f"[写作者] 文章已完成（{len(response.content)} 字）",
            name="writer"
        )]
    }


print(f"团队成员：{MEMBERS}")
print("Supervisor：负责调度，自身不直接完成任务")


# ============================================================
# Part 2：定义 Supervisor 节点和路由逻辑
# ============================================================
print("\n--- Part 2：Supervisor 节点和调度逻辑 ---")
print("说明：Supervisor 检查当前状态，决定下一步调用哪个 Agent")


def supervisor_node(state: TeamState) -> dict:
    """协调者节点：基于规则决定下一步的执行方向

    生产环境中可以用 LLM 做决策（更灵活）
    这里用规则做决策（更清晰易懂）
    """
    has_research = bool(state.get("research"))
    has_article = bool(state.get("article"))

    if not has_research:
        next_agent = "researcher"
        reason = "研究结果为空，需要先收集资料"
    elif not has_article:
        next_agent = "writer"
        reason = "已有研究结果，需要撰写文章"
    else:
        next_agent = FINISH
        reason = "研究和写作均已完成"

    print(f"  [Supervisor] 决策：→ {next_agent}（{reason}）")
    return {"next": next_agent}


def route_supervisor(state: TeamState) -> str:
    """路由函数：根据 state['next'] 决定下一个节点"""
    return state["next"]


# ============================================================
# Part 3：构建 Supervisor 图并运行
# ============================================================
print("\n--- Part 3：构建并运行 Supervisor 图 ---")

# 构建图
graph = StateGraph(TeamState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)

# 图的入口：从 Supervisor 开始
graph.add_edge(START, "supervisor")

# Supervisor 的条件路由：根据 next 字段决定去哪里
graph.add_conditional_edges("supervisor", route_supervisor, {
    "researcher": "researcher",
    "writer": "writer",
    FINISH: END,                 # FINISH → 终止图
})

# 每个 Agent 执行完后回到 Supervisor（重新决策）
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")

app = graph.compile()

print("Supervisor 图结构：")
print("  START → supervisor → [researcher | writer | END]")
print("          ↑─────────────────────────────────┘")
print("  （researcher 和 writer 执行完后都回到 supervisor）")

# 运行团队
topic = "人工智能在医疗领域的应用"
initial_state: TeamState = {
    "messages": [HumanMessage(content=f"请研究并撰写关于「{topic}」的文章")],
    "next": "",
    "topic": topic,
    "research": "",
    "article": ""
}

print(f"\n[运行团队] 主题：{topic}")
print("[执行追踪]")

final_state = app.invoke(initial_state)

print("\n[执行结果]")
print(f"调度记录（消息数）：{len(final_state['messages'])} 条")
print(f"\n研究结果预览：\n{final_state['research'][:200]}...")
print(f"\n最终文章：\n{final_state['article']}")

print("\n[调用顺序追踪]")
for i, msg in enumerate(final_state["messages"], 1):
    sender = getattr(msg, "name", type(msg).__name__)
    content_preview = str(msg.content)[:60]
    print(f"  {i}. [{sender}] {content_preview}")


def main():
    print("\n--- Supervisor 模式总结 ---")
    print()
    print("图结构：")
    print("  START → Supervisor（决策中心）")
    print("          ↓ 调度 ↓")
    print("  Agent A / Agent B / ... / END")
    print("          ↑ 回报 ↑")
    print()
    print("优点：")
    print("  1. 控制流清晰，所有决策集中在 Supervisor")
    print("  2. 易于调试：只需检查 Supervisor 的决策逻辑")
    print("  3. 专职 Agent 简单，各司其职")
    print()
    print("缺点：")
    print("  1. Supervisor 是单点瓶颈")
    print("  2. 每次都要经过 Supervisor 增加延迟")
    print("  3. 复杂任务可能导致 Supervisor 逻辑膨胀")
    print()
    print("适用场景：")
    print("  - 有明确流程的多步骤任务")
    print("  - 需要集中控制和审计的工作流")
    print("  - Agent 数量较少（2-5 个）的团队")


if __name__ == "__main__":
    main()
