import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
主题：Network Pattern —— 去中心化多 Agent 协作

学习目标：
  1. 理解 Network 模式 vs Supervisor 模式的区别
  2. 掌握 Command(goto=...) 实现 Agent 间的直接 Handoff
  3. 理解去中心化协作：每个 Agent 自己决定把控制权给谁
  4. 学会用 Command(update=...) 同时更新 State 和路由
  5. 理解两种模式的适用场景

核心概念：
  Network 模式 = 每个 Agent 自主决定下一步（无中心调度者）

  Command(goto="agent_b", update={"key": "value"})
  = 同时做两件事：更新 State + 路由到 agent_b

  Supervisor 模式：中心化，易控制，适合明确流程
  Network 模式  ：去中心化，更灵活，适合动态协作

  Handoff = 一个 Agent 把控制权移交给另一个 Agent

前置知识：已完成 22_supervisor_pattern.py
"""

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command

load_dotenv()

# 初始化 LLM
llm = ChatAnthropic(model="ppio/pa/claude-sonnet-4-6")

print("=" * 60)
print("LangGraph 23：Network Pattern —— 去中心化多 Agent 协作")
print("=" * 60)


# ============================================================
# Part 1：两 Agent 直接 Handoff（Writer → Reviewer）
# ============================================================
print("\n--- Part 1：两 Agent 直接 Handoff ---")
print("说明：Writer 完成草稿后，自主把控制权移交给 Reviewer")
print("      无需 Supervisor 协调，Agent 之间直接传递控制权")


class NetworkState(TypedDict):
    """Network 模式的共享 State"""
    messages: Annotated[list, add_messages]  # 消息历史
    task: str       # 写作任务
    draft: str      # 草稿内容
    review: str     # 审核意见


def agent_writer(state: NetworkState) -> Command:
    """写作 Agent：完成草稿后把控制权移交给 Reviewer

    返回 Command 而不是普通 dict：
    - goto: 指定下一个节点
    - update: 同时更新 State
    """
    print(f"  [Writer] 开始写作任务：{state['task']}")
    response = llm.invoke(f"写一段关于「{state['task']}」的简短介绍（80字内）")
    print(f"  [Writer] 草稿完成，移交给 Reviewer")

    # Command：更新 State 并路由到 reviewer
    return Command(
        goto="reviewer",
        update={
            "draft": response.content,
            "messages": [AIMessage(content=response.content, name="writer")]
        }
    )


def agent_reviewer(state: NetworkState) -> Command:
    """审核 Agent：评审草稿后自主决定结束流程

    Network 模式：Reviewer 自己决定结束，不需要 Supervisor
    """
    print(f"  [Reviewer] 开始审核草稿...")
    response = llm.invoke(
        f"请简短评价以下文章（好/中/差 + 一句理由，不超过 30 字）：\n{state['draft']}"
    )
    print(f"  [Reviewer] 审核完成，结束流程")

    # Command：更新 State 并路由到 END
    return Command(
        goto=END,
        update={
            "review": response.content,
            "messages": [AIMessage(content=response.content, name="reviewer")]
        }
    )


# 构建两 Agent 网络图
network_graph = StateGraph(NetworkState)
network_graph.add_node("writer", agent_writer)
network_graph.add_node("reviewer", agent_reviewer)
network_graph.add_edge(START, "writer")  # 从 writer 开始
# 注意：不需要 add_edge("writer", "reviewer")
# Command(goto="reviewer") 自动处理路由
network_app = network_graph.compile()

print("Network 图结构：")
print("  START → writer（Command goto=reviewer）→ reviewer（Command goto=END）→ END")
print("  关键：路由由 Command 对象控制，不是 add_edge")

task = "可持续能源的未来发展"
network_input: NetworkState = {
    "messages": [HumanMessage(content=f"请写作并审核关于「{task}」的文章")],
    "task": task,
    "draft": "",
    "review": ""
}

print(f"\n[两 Agent Handoff 测试] 任务：{task}")
network_result = network_app.invoke(network_input)
print(f"\n草稿：{network_result['draft'][:100]}...")
print(f"审核：{network_result['review']}")


# ============================================================
# Part 2：三 Agent 网络（含条件路由）
# ============================================================
print("\n--- Part 2：三 Agent 网络（Reviewer 可退回修改）---")
print("说明：Reviewer 可以批准（→ END）或退回（→ Writer 重写）")


class ConditionalNetworkState(TypedDict):
    """含条件路由的 Network State"""
    messages: Annotated[list, add_messages]
    task: str
    draft: str
    review: str
    revision_count: int     # 修改次数（防止无限循环）
    approved: bool          # 是否通过审核


def conditional_writer(state: ConditionalNetworkState) -> Command:
    """写作 Agent：每次（包括修改后）都写新版本"""
    revision = state.get("revision_count", 0)
    if revision > 0:
        print(f"  [Writer] 第 {revision + 1} 次写作（修改版本）")
        prompt = (
            f"根据以下审核意见修改文章：\n"
            f"原文：{state['draft']}\n"
            f"审核意见：{state['review']}\n"
            f"请改进后重新输出（80字内）"
        )
    else:
        print(f"  [Writer] 第 1 次写作（初稿）")
        prompt = f"写一段关于「{state['task']}」的简短介绍（80字内）"

    response = llm.invoke(prompt)
    print(f"  [Writer] 完成，移交 Reviewer")

    return Command(
        goto="conditional_reviewer",
        update={
            "draft": response.content,
            "revision_count": revision + 1,
            "messages": [AIMessage(
                content=f"[Writer v{revision + 1}] {response.content[:50]}...",
                name="writer"
            )]
        }
    )


def conditional_reviewer(state: ConditionalNetworkState) -> Command:
    """审核 Agent：决定批准或退回修改（最多修改 2 次）"""
    revision_count = state.get("revision_count", 1)
    print(f"  [Reviewer] 审核第 {revision_count} 版...")

    # 防止无限循环：超过 2 次修改直接批准
    if revision_count >= 2:
        print(f"  [Reviewer] 达到最大修改次数，强制批准")
        return Command(
            goto=END,
            update={
                "approved": True,
                "review": "达到最大修改次数，文章已接受",
                "messages": [AIMessage(content="[Reviewer] 文章已通过（强制）", name="reviewer")]
            }
        )

    # 让 LLM 评估文章质量
    eval_response = llm.invoke(
        f"评估以下文章质量，只回答'通过'或'需要修改'：\n{state['draft']}"
    )
    decision = eval_response.content.strip()
    approved = "通过" in decision

    if approved:
        print(f"  [Reviewer] 批准，结束流程")
        return Command(
            goto=END,
            update={
                "approved": True,
                "review": f"文章通过审核：{decision}",
                "messages": [AIMessage(content=f"[Reviewer] 批准：{decision}", name="reviewer")]
            }
        )
    else:
        print(f"  [Reviewer] 退回修改，重新写作")
        return Command(
            goto="conditional_writer",          # 退回给 Writer 重写
            update={
                "approved": False,
                "review": f"需要改进：{decision}",
                "messages": [AIMessage(
                    content=f"[Reviewer] 退回修改：{decision}",
                    name="reviewer"
                )]
            }
        )


# 构建条件路由网络图
cond_graph = StateGraph(ConditionalNetworkState)
cond_graph.add_node("conditional_writer", conditional_writer)
cond_graph.add_node("conditional_reviewer", conditional_reviewer)
cond_graph.add_edge(START, "conditional_writer")
cond_app = cond_graph.compile()

print("条件 Network 图结构：")
print("  START → writer → reviewer → [END（批准）| writer（退回修改）]")
print("  writer ← ─────────────────────────────────────────────────┘")

cond_input: ConditionalNetworkState = {
    "messages": [HumanMessage(content="写一篇关于「气候变化」的文章")],
    "task": "气候变化",
    "draft": "",
    "review": "",
    "revision_count": 0,
    "approved": False
}

print(f"\n[条件路由 Network 测试] 任务：{cond_input['task']}")
cond_result = cond_app.invoke(cond_input)
print(f"\n最终版本（第 {cond_result['revision_count']} 版）：")
print(f"  {cond_result['draft'][:120]}...")
print(f"审核结论：{cond_result['review']}")
print(f"是否批准：{'是' if cond_result['approved'] else '否'}")


# ============================================================
# Part 3：Supervisor vs Network 对比表
# ============================================================
def main():
    print("\n--- Part 3：Supervisor vs Network 对比 ---")
    print()
    print("=" * 65)
    print(f"{'维度':<12} {'Supervisor 模式':<25} {'Network 模式'}")
    print("=" * 65)
    print(f"{'协调方式':<12} {'中心化（Supervisor 决策）':<25} {'去中心化（Agent 自决）'}")
    print(f"{'路由机制':<12} {'条件边 + next 字段':<25} {'Command(goto=...)'}")
    print(f"{'决策位置':<12} {'所有决策在 Supervisor':<25} {'每个 Agent 自主决定'}")
    print(f"{'灵活性':<12} {'较低（流程固定）':<25} {'较高（动态路由）'}")
    print(f"{'可控性':<12} {'高（集中审计）':<25} {'较低（分散决策）'}")
    print(f"{'调试难度':<12} {'低（单点分析）':<25} {'较高（多点追踪）'}")
    print(f"{'适用场景':<12} {'明确流程 / 少数 Agent':<25} {'动态协作 / 灵活流程'}")
    print("=" * 65)
    print()
    print("核心语法对比：")
    print()
    print("  Supervisor 模式：")
    print("    graph.add_conditional_edges('supervisor', route_fn, {...})")
    print("    # 所有 Agent 执行完后回到 supervisor")
    print()
    print("  Network 模式：")
    print("    def agent_a(state) -> Command:")
    print("        return Command(goto='agent_b', update={'key': 'val'})")
    print("    # Command 同时更新状态和路由，无需 add_edge")
    print()
    print("选择建议：")
    print("  - 流程明确、需要集中控制 → Supervisor 模式")
    print("  - 协作灵活、Agent 需要自主决策 → Network 模式")
    print("  - 混合使用：Supervisor 管理全局，子团队内部用 Network")


if __name__ == "__main__":
    main()
