"""
项目：研究团队 —— 多 Agent 协作完成深度研究报告

项目目标：
  输入一个研究主题，输出一份完整的研究报告（事实+分析+正文）

Agent 角色：
  - 搜索员（Researcher）：收集关于主题的关键事实和数据
  - 分析师（Analyst）   ：分析研究结果，提炼核心洞察
  - 写作者（Writer）    ：撰写完整报告（标题+正文+结论）
  - 协调员（Supervisor）：决定调用顺序，判断何时完成

工作流（Supervisor 模式）：
  用户输入主题
  → Supervisor 决定调用 Researcher
  → Researcher 完成后返回 Supervisor
  → Supervisor 决定调用 Analyst
  → Analyst 完成后返回 Supervisor
  → Supervisor 决定调用 Writer
  → Writer 完成后返回 Supervisor
  → Supervisor 判断完成，结束

学习重点：
  - Supervisor 模式的完整实现
  - 多 Agent 状态传递
  - 真实工作流的设计思路

前置知识：已完成 22_supervisor_pattern.py
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=512)

# ── 团队成员名称 ──
MEMBERS = ["researcher", "analyst", "writer"]
FINISH = "FINISH"

# ── 共享状态 ──────────────────────────────────────────────────────────────────
class ResearchTeamState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str           # 研究主题
    research: str        # 搜索员的研究结果
    analysis: str        # 分析师的分析结果
    report: str          # 写作者的最终报告
    next: str            # 下一个要调用的 Agent


# ── Agent 节点 ────────────────────────────────────────────────────────────────

def researcher_node(state: ResearchTeamState) -> dict:
    """搜索员：收集关于主题的关键事实"""
    print(f"\n  [Researcher] 正在研究：{state['topic']}")
    response = llm.invoke(
        f"你是一位专业研究员。请收集关于「{state['topic']}」的5个关键事实，"
        f"每条用一句话表述，格式：1. xxx 2. xxx..."
    )
    print(f"  [Researcher] 完成，收集了 {len(response.content.split(chr(10)))} 条信息")
    return {
        "research": response.content,
        "messages": [AIMessage(content=f"[研究员完成]\n{response.content}", name="researcher")],
    }


def analyst_node(state: ResearchTeamState) -> dict:
    """分析师：提炼洞察"""
    print(f"\n  [Analyst] 正在分析研究结果...")
    response = llm.invoke(
        f"你是一位数据分析师。基于以下研究结果，提炼3个核心洞察（每条包含：洞察+原因+影响）：\n\n"
        f"{state['research']}"
    )
    print(f"  [Analyst] 完成分析")
    return {
        "analysis": response.content,
        "messages": [AIMessage(content=f"[分析师完成]\n{response.content}", name="analyst")],
    }


def writer_node(state: ResearchTeamState) -> dict:
    """写作者：撰写最终报告"""
    print(f"\n  [Writer] 正在撰写报告...")
    response = llm.invoke(
        f"你是一位专业写作者。根据以下研究和分析，撰写一份结构化报告：\n\n"
        f"主题：{state['topic']}\n\n"
        f"研究结果：\n{state['research']}\n\n"
        f"核心洞察：\n{state['analysis']}\n\n"
        f"报告格式：标题 + 概述（2句）+ 主要发现（3点）+ 结论（1句）"
    )
    print(f"  [Writer] 报告撰写完成")
    return {
        "report": response.content,
        "messages": [AIMessage(content=f"[写作者完成]\n{response.content[:100]}...", name="writer")],
    }


def supervisor_node(state: ResearchTeamState) -> dict:
    """协调员：决定下一步调用哪个 Agent"""
    if not state.get("research"):
        next_agent = "researcher"
    elif not state.get("analysis"):
        next_agent = "analyst"
    elif not state.get("report"):
        next_agent = "writer"
    else:
        next_agent = FINISH

    print(f"\n  [Supervisor] → {next_agent}")
    return {"next": next_agent}


def route_supervisor(state: ResearchTeamState) -> str:
    return state["next"]


# ── 构建图 ─────────────────────────────────────────────────────────────────────
def build_research_team():
    graph = StateGraph(ResearchTeamState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_supervisor, {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        FINISH: END,
    })
    # 每个 Agent 完成后回到 Supervisor
    for member in MEMBERS:
        graph.add_edge(member, "supervisor")

    return graph.compile()


def main():
    print("=" * 60)
    print("研究团队 Multi-Agent 系统")
    print("=" * 60)

    app = build_research_team()

    topic = "人工智能在医疗诊断中的应用"
    print(f"\n研究主题：{topic}\n")

    result = app.invoke({
        "messages": [HumanMessage(content=f"请研究：{topic}")],
        "topic": topic,
        "research": "",
        "analysis": "",
        "report": "",
        "next": "",
    })

    print("\n" + "=" * 60)
    print("最终研究报告")
    print("=" * 60)
    print(result["report"])

    print(f"\n[执行统计] 消息总数: {len(result['messages'])}")


if __name__ == "__main__":
    main()
