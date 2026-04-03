import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
主题：Parallel Nodes —— 让多个节点同时执行，提高效率

学习目标：
  1. 理解 LangGraph 的并行执行机制
  2. 掌握静态并行：一个节点连接多个下游节点（扇出）
  3. 掌握 Send API：动态创建并行任务
  4. 学会用 reducer（operator.add）收集并行结果
  5. 理解扇入（fan-in）：合并多个并行节点的输出

核心概念：
  静态并行：add_edge("A", "B"); add_edge("A", "C") → B 和 C 同时执行
  动态并行：Send API → 根据列表动态创建 N 个并行任务

  reducer 用于合并：
  items: Annotated[List[str], operator.add]  → 所有并行结果追加到列表

  Send(node_name, state_dict) = 向指定节点发送一条消息（创建并行任务）

前置知识：已完成 14_conditional_edges.py
"""

import os
import operator
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

load_dotenv()

# 初始化 LLM
llm = ChatAnthropic(model="ppio/pa/claude-sonnet-4-6")

print("=" * 60)
print("LangGraph 19：Parallel Nodes —— 并行节点执行")
print("=" * 60)


# ============================================================
# Part 1：静态并行（扇出 + 扇入）
# ============================================================
print("\n--- Part 1：静态并行（Fan-Out + Fan-In）---")
print("说明：一个节点的输出同时触发多个下游节点并行执行")

class StaticParallelState(TypedDict):
    """静态并行的 State"""
    content: str                                    # 输入内容
    results: Annotated[List[str], operator.add]    # 并行结果（reducer 自动合并）
    summary: str                                    # 最终汇总


def start_node(state: StaticParallelState) -> dict:
    """起始节点：不做实际处理，只是触发后续并行节点"""
    print(f"  [start] 准备并行分析：{state['content'][:20]}...")
    return {}


def analysis_a(state: StaticParallelState) -> dict:
    """分析 A：从优点角度分析"""
    response = llm.invoke(f"用一句话说明「{state['content']}」的主要优点")
    result = f"[优点分析] {response.content}"
    print(f"  [analysis_a] 完成")
    return {"results": [result]}


def analysis_b(state: StaticParallelState) -> dict:
    """分析 B：从挑战角度分析"""
    response = llm.invoke(f"用一句话说明「{state['content']}」面临的主要挑战")
    result = f"[挑战分析] {response.content}"
    print(f"  [analysis_b] 完成")
    return {"results": [result]}


def summarize_node(state: StaticParallelState) -> dict:
    """汇总节点：合并两路并行分析的结果"""
    combined = "\n".join(state["results"])
    summary = f"综合分析完成，共 {len(state['results'])} 个维度：\n{combined}"
    print(f"  [summarize] 收到 {len(state['results'])} 条并行结果，已汇总")
    return {"summary": summary}


# 构建静态并行图
static_graph = StateGraph(StaticParallelState)
static_graph.add_node("start", start_node)
static_graph.add_node("analysis_a", analysis_a)
static_graph.add_node("analysis_b", analysis_b)
static_graph.add_node("summarize", summarize_node)

# 关键：同一个节点连接两个下游节点 → 静态并行
static_graph.add_edge(START, "start")
static_graph.add_edge("start", "analysis_a")   # 扇出：两条边指向不同节点
static_graph.add_edge("start", "analysis_b")   # 扇出：analysis_a 和 analysis_b 同时执行
# 扇入：两个节点都完成后才执行 summarize
static_graph.add_edge("analysis_a", "summarize")
static_graph.add_edge("analysis_b", "summarize")
static_graph.add_edge("summarize", END)
static_app = static_graph.compile()

print("图结构：START → start → [analysis_a, analysis_b]（并行）→ summarize → END")
print("reducer：results 字段使用 operator.add，自动合并两路结果")

static_input = {"content": "远程办公", "results": [], "summary": ""}
print(f"\n[静态并行测试] 主题：{static_input['content']}")
static_result = static_app.invoke(static_input)
print(f"\n汇总结果：\n{static_result['summary']}")


# ============================================================
# Part 2：动态并行（Send API）
# ============================================================
print("\n--- Part 2：动态并行（Send API）---")
print("说明：根据输入数据动态创建 N 个并行任务，N 可变")

class ParallelState(TypedDict):
    """动态并行的 State"""
    topics: List[str]                               # 要处理的主题列表
    results: Annotated[List[str], operator.add]    # 并行结果（reducer 合并）


def fan_out(state: ParallelState):
    """扇出节点：为每个 topic 创建一个独立的并行任务

    返回 Send 对象列表，LangGraph 会并行执行所有任务
    Send(node_name, state_dict) = 向指定节点发送一条消息
    """
    print(f"  [fan_out] 为 {len(state['topics'])} 个主题创建并行任务")
    return [
        Send("process_topic", {"topic": t, "results": []})
        for t in state["topics"]
    ]


def process_topic(state: dict) -> dict:
    """处理单个主题的节点：每个并行任务独立执行这个函数"""
    topic = state["topic"]
    response = llm.invoke(f"用一句话描述「{topic}」是什么")
    result = f"{topic}: {response.content}"
    print(f"  [process_topic] 完成：{topic[:10]}...")
    return {"results": [result]}


# 构建动态并行图
dynamic_graph = StateGraph(ParallelState)
dynamic_graph.add_node("process_topic", process_topic)

# 关键：使用 add_conditional_edges + fan_out 函数实现 Send-based 扇出
dynamic_graph.add_conditional_edges(START, fan_out)   # fan_out 返回 Send 列表
dynamic_graph.add_edge("process_topic", END)
dynamic_app = dynamic_graph.compile()

print("图结构：START → fan_out（条件边）→ [process_topic × N]（动态并行）→ END")
print("关键：fan_out 返回 Send 列表，N 个任务同时运行")

topics = ["机器学习", "区块链", "量子计算"]
dynamic_input = {"topics": topics, "results": []}
print(f"\n[动态并行测试] 主题列表：{topics}")
dynamic_result = dynamic_app.invoke(dynamic_input)
print(f"\n所有并行任务结果（共 {len(dynamic_result['results'])} 条）：")
for r in dynamic_result["results"]:
    print(f"  - {r[:70]}")


# ============================================================
# Part 3：并行执行的性能对比说明
# ============================================================
print("\n--- Part 3：性能对比——顺序 vs 并行 ---")

def main():
    n = len(topics)
    single_call_time = 2  # 假设每个 LLM 调用约 2 秒

    print("\n[性能对比示意]")
    print(f"  任务数量：{n} 个")
    print(f"  单次 LLM 调用时间（估计）：~{single_call_time} 秒")
    print()
    print(f"  顺序执行：{n} × {single_call_time}s = ~{n * single_call_time} 秒")
    print(f"  并行执行：max({single_call_time}s × {n}) ≈ ~{single_call_time} 秒（所有任务同时运行）")
    print(f"  理论加速比：{n}x")
    print()
    print("注意事项：")
    print("  - 并行度受 LLM API 的 rate limit 限制")
    print("  - 网络延迟和 API 响应时间影响实际加速效果")
    print("  - 任务数越多，并行优势越明显")
    print()
    print("语法总结：")
    print("  # 静态并行（固定数量）")
    print("  graph.add_edge('A', 'B')  # 同时触发 B 和 C")
    print("  graph.add_edge('A', 'C')")
    print()
    print("  # 动态并行（数量可变）")
    print("  def fan_out(state): return [Send('node', {...}) for item in state['items']]")
    print("  graph.add_conditional_edges(START, fan_out)")
    print()
    print("  # reducer 用于合并并行结果")
    print("  results: Annotated[List[str], operator.add]")


if __name__ == "__main__":
    main()
