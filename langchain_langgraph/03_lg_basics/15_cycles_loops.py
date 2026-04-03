"""
主题：Cycles & Loops —— 让图可以循环，实现迭代优化

学习目标：
  1. 理解为什么 LangGraph 支持循环（LangChain Chain 不支持）
  2. 掌握循环图的退出条件设计
  3. 实现"写作→评分→改写"的迭代优化循环
  4. 掌握用 iteration_count 防止无限循环
  5. 理解循环在 Agent 中的作用（ReAct 的核心）

核心概念：
  循环 = 条件边的目标节点 = 当前节点的祖先节点
  退出条件：score >= threshold 或 iteration >= max_iter

  无限循环风险：必须设置最大循环次数！
  生产环境：max_iterations=10 是常见设置

  循环的价值：让 AI 自我改进，而不是一次生成定终身

前置知识：已完成 14_conditional_edges.py
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import re
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=512)


# =============================================================================
# Part 1：写作迭代优化循环（写作→评分→改写）
# =============================================================================

class WritingState(TypedDict):
    topic: str
    draft: str
    score: int      # 1-10 评分
    feedback: str   # 评分反馈
    iteration: int  # 循环计数
    history: list   # 保存每轮草稿和分数，用于最终对比


def write_node(state: WritingState) -> dict:
    """写作节点：首轮写初稿，后续根据反馈改写"""
    if state["iteration"] == 0:
        prompt = f"写一段关于「{state['topic']}」的简短介绍（100字内）"
        print(f"  [write_node] 第1轮：生成初稿")
    else:
        prompt = (
            f"根据反馈改进文章：\n"
            f"原文：{state['draft']}\n"
            f"反馈：{state['feedback']}\n"
            f"请改进（100字内）"
        )
        print(f"  [write_node] 第{state['iteration'] + 1}轮：根据反馈改写")

    response = llm.invoke(prompt)
    new_draft = response.content

    # 将本轮草稿追加到历史记录（注意：不能直接 append，需返回新列表）
    new_history = state["history"] + [{"iteration": state["iteration"], "draft": new_draft}]

    return {
        "draft": new_draft,
        "iteration": state["iteration"] + 1,
        "history": new_history,
    }


def score_node(state: WritingState) -> dict:
    """评分节点：对草稿打分并给出改进建议"""
    print(f"  [score_node] 对第{state['iteration']}轮草稿评分...")

    response = llm.invoke(
        f"请给以下文章打分（1-10分）并给出一条改进建议。\n"
        f"格式：\n分数：X\n建议：...\n\n文章：{state['draft']}"
    )

    # 从回复中解析分数
    score_match = re.search(r'分数[：:]\s*(\d+)', response.content)
    score = int(score_match.group(1)) if score_match else 7
    score = max(1, min(10, score))  # 限制在 1-10 范围内

    print(f"  [score_node] 得分: {score}/10")

    # 将分数更新到历史记录的最后一条
    new_history = state["history"][:-1] + [
        {**state["history"][-1], "score": score}
    ] if state["history"] else state["history"]

    return {"score": score, "feedback": response.content, "history": new_history}


def should_continue(state: WritingState) -> str:
    """循环控制：分数达标或达到最大迭代次数则退出"""
    if state["score"] >= 8:
        print(f"  [should_continue] 分数 {state['score']} >= 8，达标，退出循环")
        return END
    if state["iteration"] >= 3:
        print(f"  [should_continue] 已迭代 {state['iteration']} 轮，达到上限，退出循环")
        return END
    print(f"  [should_continue] 分数 {state['score']} < 8，继续改写...")
    return "write"


def demo_writing_loop():
    """演示写作迭代优化循环"""
    graph = StateGraph(WritingState)

    graph.add_node("write", write_node)
    graph.add_node("score", score_node)

    # 连接：START → write → score → 条件判断
    graph.add_edge(START, "write")
    graph.add_edge("write", "score")
    # 条件边：从 score 节点出发，由 should_continue 决定继续还是结束
    graph.add_conditional_edges("score", should_continue)

    app = graph.compile()

    result = app.invoke({
        "topic": "Python 的优雅之道",
        "draft": "",
        "score": 0,
        "feedback": "",
        "iteration": 0,
        "history": [],
    })

    print(f"\n[写作循环] 共执行 {result['iteration']} 轮迭代")
    print(f"  最终得分: {result['score']}/10")
    print(f"  最终草稿: {result['draft'][:100]}...")

    return result


# =============================================================================
# Part 2：计数器循环（验证循环机制本身）
# =============================================================================

class CounterState(TypedDict):
    count: int
    max_count: int
    log: list


def increment_node(state: CounterState) -> dict:
    """递增计数器"""
    new_count = state["count"] + 1
    new_log = state["log"] + [f"count={new_count}"]
    print(f"  [increment_node] count: {state['count']} → {new_count}")
    return {"count": new_count, "log": new_log}


def check_done(state: CounterState) -> str:
    """检查是否达到目标次数"""
    if state["count"] >= state["max_count"]:
        return END
    return "increment"


def demo_counter_loop():
    """演示纯粹的计数器循环，验证循环机制"""
    graph = StateGraph(CounterState)
    graph.add_node("increment", increment_node)

    graph.add_edge(START, "increment")
    graph.add_conditional_edges("increment", check_done)

    app = graph.compile()

    result = app.invoke({"count": 0, "max_count": 5, "log": []})

    print(f"\n[计数器循环] 执行完毕")
    print(f"  最终 count: {result['count']}")
    print(f"  执行记录: {' → '.join(result['log'])}")


# =============================================================================
# Part 3：打印迭代过程对比，展示 AI 改进效果
# =============================================================================

def demo_improvement_comparison(result: dict):
    """打印各轮草稿和分数，展示改进过程"""
    print(f"\n[迭代改进对比]")
    print(f"  主题: {result['topic']}")
    print(f"  {'轮次':<6} {'分数':<6} {'草稿（前60字）'}")
    print(f"  {'-'*6} {'-'*6} {'-'*30}")

    for item in result["history"]:
        iteration = item.get("iteration", "?")
        score = item.get("score", "-")
        draft_preview = item.get("draft", "")[:60].replace("\n", " ")
        print(f"  第{iteration + 1}轮    {score:<6} {draft_preview}...")

    print(f"\n  最终得分: {result['score']}/10")
    print(f"  结论: {'改写有效提升质量' if len(result['history']) > 1 else '首轮即达标'}")


def main():
    print("=" * 60)
    print("Part 1：写作迭代优化循环")
    print("=" * 60)
    writing_result = demo_writing_loop()

    print("\n" + "=" * 60)
    print("Part 2：计数器循环（验证循环机制）")
    print("=" * 60)
    demo_counter_loop()

    print("\n" + "=" * 60)
    print("Part 3：迭代改进过程对比")
    print("=" * 60)
    demo_improvement_comparison(writing_result)


if __name__ == "__main__":
    main()
