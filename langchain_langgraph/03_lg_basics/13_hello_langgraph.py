"""
主题：Hello LangGraph —— 用图结构编排 AI 工作流

学习目标：
  1. 理解 LangGraph 是什么：用有向图描述 AI 工作流的框架
  2. 掌握核心三要素：State（状态）、Node（节点）、Edge（边）
  3. 理解 TypedDict 如何定义图的共享状态
  4. 掌握 StateGraph 的创建、编译和调用
  5. 理解 START / END 特殊节点的作用

核心概念：
  LangGraph vs LangChain Chain：
  Chain  = 线性流水线（A→B→C，固定顺序）
  Graph  = 有向图（可循环、可分支、可并行，更灵活）

  State  = 图中所有节点共享的数据字典（TypedDict）
  Node   = 处理 State 并返回更新的函数
  Edge   = 节点之间的连接关系

  工作流：
  1. 定义 State 类型
  2. 写节点函数（接收 State，返回更新字典）
  3. 建图：add_node → add_edge → compile
  4. 调用：graph.invoke(初始State)

前置知识：已完成 01_lc_basics/ 全部 demo
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
llm = ChatAnthropic(model=MODEL, max_tokens=256)


# =============================================================================
# Part 1：最简单的图（单节点）
# =============================================================================

# State 定义：图中所有节点共享的状态
class SimpleState(TypedDict):
    input: str       # 用户输入
    output: str      # 最终输出
    step_count: int  # 执行步骤计数


def process_node(state: SimpleState) -> dict:
    """节点函数：接收当前 State，返回要更新的字段"""
    print(f"  [process_node] 收到输入: {state['input']}")

    response = llm.invoke(f"用一句话回答：{state['input']}")

    return {
        "output": response.content,
        "step_count": state["step_count"] + 1,
    }


def demo_simple_graph():
    """构建并运行最简单的单节点图"""
    # 1. 创建图，指定 State 类型
    graph = StateGraph(SimpleState)

    # 2. 添加节点（名称，函数）
    graph.add_node("process", process_node)

    # 3. 添加边（START → 节点 → END）
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    # 4. 编译（生成可执行图）
    app = graph.compile()

    # 5. 调用
    result = app.invoke({
        "input": "什么是人工智能？",
        "output": "",
        "step_count": 0,
    })

    print(f"\n[简单图] 最终State:")
    print(f"  input     : {result['input']}")
    print(f"  output    : {result['output']}")
    print(f"  step_count: {result['step_count']}")


# =============================================================================
# Part 2：多节点顺序图
# =============================================================================

class PipelineState(TypedDict):
    user_query: str
    refined_query: str   # 改写后的查询
    final_answer: str


def refine_node(state: PipelineState) -> dict:
    """节点1：改写用户问题，使其更清晰"""
    print(f"  [refine_node] 改写问题...")
    response = llm.invoke(
        f"把这个问题改写得更清晰（只输出改写后的问题）：{state['user_query']}"
    )
    return {"refined_query": response.content}


def answer_node(state: PipelineState) -> dict:
    """节点2：回答改写后的问题"""
    print(f"  [answer_node] 回答问题...")
    response = llm.invoke(f"请简洁地回答：{state['refined_query']}")
    return {"final_answer": response.content}


def demo_multi_node_graph():
    """多节点顺序图：问题改写 → 回答"""
    graph = StateGraph(PipelineState)

    graph.add_node("refine", refine_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "refine")
    graph.add_edge("refine", "answer")
    graph.add_edge("answer", END)

    app = graph.compile()

    result = app.invoke({
        "user_query": "py咋学",
        "refined_query": "",
        "final_answer": "",
    })

    print(f"\n[多节点图]")
    print(f"  原始问题 : {result['user_query']}")
    print(f"  改写问题 : {result['refined_query']}")
    print(f"  最终回答 : {result['final_answer'][:80]}...")


# =============================================================================
# Part 3：消息图（内置 add_messages reducer）
# =============================================================================

class ChatState(TypedDict):
    # Annotated[list, add_messages] 表示：
    # messages 字段使用 add_messages reducer（自动追加，不是覆盖）
    messages: Annotated[list, add_messages]


def chat_node(state: ChatState) -> dict:
    """聊天节点：调用 LLM，追加回复消息"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # add_messages 会把这条消息追加到列表


def demo_message_graph():
    """消息图：最接近真实聊天机器人的模式"""
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    app = graph.compile()

    result = app.invoke({
        "messages": [HumanMessage(content="你好，用一句话自我介绍")]
    })

    print(f"\n[消息图]")
    for msg in result["messages"]:
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content[:80]}")


def main():
    print("=" * 60)
    print("Part 1：最简单的图（单节点）")
    print("=" * 60)
    demo_simple_graph()

    print("\n" + "=" * 60)
    print("Part 2：多节点顺序图")
    print("=" * 60)
    demo_multi_node_graph()

    print("\n" + "=" * 60)
    print("Part 3：消息图（add_messages reducer）")
    print("=" * 60)
    demo_message_graph()


if __name__ == "__main__":
    main()
