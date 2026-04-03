"""
项目：代码审查团队 —— 多 Agent 协作完成代码开发与审查

项目目标：
  输入一个编程任务，经过开发→测试→审查的迭代循环，输出高质量代码

Agent 角色：
  - 开发者（Developer）：根据需求编写 Python 函数
  - 测试员（Tester）   ：编写测试用例，验证代码正确性
  - 审核者（Reviewer） ：评审代码质量，给出通过/修改意见
  - 协调员（Supervisor）：协调工作流，决定是否需要修改

工作流（循环模式）：
  输入任务
  → Developer 编写代码
  → Tester 编写测试
  → Reviewer 评审
  → 如果通过 → 结束
  → 如果不通过 → Developer 修改 → 循环（最多3次）

学习重点：
  - Supervisor 模式 + 循环（综合应用）
  - iteration_count 防止无限循环
  - 多 Agent 工作流的实际业务场景

前置知识：已完成 24_research_team.py
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=768)

FINISH = "FINISH"


class CodeReviewState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str            # 编程任务描述
    code: str            # 当前代码
    tests: str           # 测试用例
    review: str          # 审核意见
    approved: bool       # 是否通过审核
    iteration: int       # 迭代次数
    next: str            # 下一个 Agent


def developer_node(state: CodeReviewState) -> dict:
    """开发者：编写或修改代码"""
    iteration = state.get("iteration", 0)
    print(f"\n  [Developer] 第 {iteration + 1} 次编写代码...")

    if iteration == 0:
        prompt = f"请用Python实现以下功能，只输出代码（包含函数定义和简短注释）：\n{state['task']}"
    else:
        prompt = (
            f"请根据以下审核意见修改代码：\n"
            f"原代码：\n{state['code']}\n\n"
            f"审核意见：{state['review']}\n\n"
            f"请输出修改后的完整代码："
        )

    response = llm.invoke(prompt)
    print(f"  [Developer] 代码编写完成（{len(response.content)} 字符）")
    return {
        "code": response.content,
        "iteration": iteration + 1,
        "messages": [AIMessage(content=f"[开发者-第{iteration+1}次]\n{response.content[:100]}...", name="developer")],
    }


def tester_node(state: CodeReviewState) -> dict:
    """测试员：编写测试用例"""
    print(f"\n  [Tester] 编写测试用例...")
    response = llm.invoke(
        f"为以下Python代码编写3个测试用例（使用 assert 语句，要有边界值测试）：\n\n{state['code']}"
    )
    print(f"  [Tester] 测试用例编写完成")
    return {
        "tests": response.content,
        "messages": [AIMessage(content=f"[测试员]\n{response.content[:100]}...", name="tester")],
    }


def reviewer_node(state: CodeReviewState) -> dict:
    """审核者：评审代码质量"""
    print(f"\n  [Reviewer] 正在审核代码...")
    response = llm.invoke(
        f"请审核以下Python代码的质量。\n"
        f"代码：\n{state['code']}\n\n"
        f"测试：\n{state['tests']}\n\n"
        f"请回答：\n"
        f"1. 代码是否正确？\n"
        f"2. 是否有改进空间？\n"
        f"3. 最终结论：通过 或 需要修改（并说明原因）\n"
        f"最后一行必须是：结论：通过 或 结论：需要修改"
    )

    approved = "通过" in response.content and "需要修改" not in response.content.split("结论：")[-1]
    print(f"  [Reviewer] 审核结论: {'通过' if approved else '需要修改'}")
    return {
        "review": response.content,
        "approved": approved,
        "messages": [AIMessage(content=f"[审核者]\n{response.content[:100]}...", name="reviewer")],
    }


def supervisor_node(state: CodeReviewState) -> dict:
    """协调员：决定工作流走向"""
    iteration = state.get("iteration", 0)
    approved = state.get("approved", False)

    if not state.get("code"):
        next_agent = "developer"
    elif not state.get("tests"):
        next_agent = "tester"
    elif not state.get("review"):
        next_agent = "reviewer"
    elif approved or iteration >= 3:
        next_agent = FINISH
        if iteration >= 3 and not approved:
            print(f"\n  [Supervisor] 达到最大迭代次数（{iteration}），强制结束")
    else:
        # 需要修改 → 重置测试和审核，让开发者重新修改
        next_agent = "developer"

    print(f"\n  [Supervisor] → {next_agent}")
    return {"next": next_agent}


def route_supervisor(state: CodeReviewState) -> str:
    return state["next"]


def build_code_review_team():
    graph = StateGraph(CodeReviewState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("developer", developer_node)
    graph.add_node("tester", tester_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_supervisor, {
        "developer": "developer",
        "tester": "tester",
        "reviewer": "reviewer",
        FINISH: END,
    })
    graph.add_edge("developer", "supervisor")
    graph.add_edge("tester", "supervisor")
    graph.add_edge("reviewer", "supervisor")

    return graph.compile()


def main():
    print("=" * 60)
    print("代码审查团队 Multi-Agent 系统")
    print("=" * 60)

    app = build_code_review_team()

    task = "实现一个函数 flatten(lst)，将嵌套列表展平为一维列表，支持任意深度嵌套"
    print(f"\n编程任务：{task}\n")

    result = app.invoke({
        "messages": [HumanMessage(content=task)],
        "task": task,
        "code": "",
        "tests": "",
        "review": "",
        "approved": False,
        "iteration": 0,
        "next": "",
    })

    print("\n" + "=" * 60)
    print("最终代码")
    print("=" * 60)
    print(result["code"])

    print("\n" + "=" * 60)
    print("测试用例")
    print("=" * 60)
    print(result["tests"])

    print(f"\n[执行统计]")
    print(f"  迭代次数: {result['iteration']}")
    print(f"  最终状态: {'通过' if result['approved'] else '未通过（超出最大迭代）'}")


if __name__ == "__main__":
    main()
