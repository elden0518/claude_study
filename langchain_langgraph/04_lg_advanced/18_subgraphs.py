import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
主题：Subgraphs —— 把复杂图模块化，图中嵌套图

学习目标：
  1. 理解子图（Subgraph）的概念和使用场景
  2. 掌握把子图编译后作为父图节点的方法
  3. 理解父图 State 与子图 State 的关系
  4. 学会用子图封装可复用的工作流模块
  5. 理解子图的独立编译和测试

核心概念：
  子图 = 独立编译的 StateGraph，可作为父图的一个节点
  父图调用子图 = parent.add_node("sub_module", subgraph_app)

  State 映射：
  - 父图和子图可以共享相同字段（直接访问）
  - 也可以用包装函数做字段映射（转换）

  使用场景：
  - 把复杂工作流分解为可测试的模块
  - 在多个父图中复用同一个子图
  - 团队协作时各自负责独立的子图

前置知识：已完成 17_persistence.py
"""

import os
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# 初始化 LLM
llm = ChatAnthropic(model="ppio/pa/claude-sonnet-4-6")

print("=" * 60)
print("LangGraph 18：Subgraphs —— 图中嵌套图")
print("=" * 60)


# ============================================================
# Part 1：定义并独立编译一个"翻译子图"
# ============================================================
print("\n--- Part 1：独立子图的定义与编译 ---")
print("说明：子图是一个独立编译的 StateGraph，可以单独测试")

class SubState(TypedDict):
    """子图专用 State：仅包含子图需要的字段"""
    text: str        # 输入：待翻译的中文文本
    translated: str  # 输出：翻译后的英文文本


def translate_node(state: SubState) -> dict:
    """翻译节点：把中文翻译成英文"""
    response = llm.invoke(f"把以下中文翻译成英文，只输出翻译结果：{state['text']}")
    return {"translated": response.content}


# 构建子图
sub_graph = StateGraph(SubState)
sub_graph.add_node("translate", translate_node)
sub_graph.add_edge(START, "translate")
sub_graph.add_edge("translate", END)
subgraph_app = sub_graph.compile()  # 编译子图

print("子图结构：START → translate → END")
print("子图 State：{text: str, translated: str}")

# 独立测试子图
test_input = {"text": "人工智能正在改变世界", "translated": ""}
print(f"\n[子图独立测试]")
print(f"  输入：{test_input['text']}")
result = subgraph_app.invoke(test_input)
print(f"  输出：{result['translated']}")


# ============================================================
# Part 2：把子图嵌入父图作为节点
# ============================================================
print("\n--- Part 2：把子图作为父图的一个节点 ---")
print("说明：父图调用子图时，子图对外表现为普通节点")

class ParentState(TypedDict):
    """父图 State：包含完整的工作流字段"""
    text: str           # 原始中文文本
    translated: str     # 子图翻译结果（与 SubState 共享字段名）
    final_output: str   # 格式化后的最终输出


def format_node(state: ParentState) -> dict:
    """格式化节点：把翻译结果包装成最终输出"""
    final = f"[翻译完成]\n原文：{state['text']}\n译文：{state['translated']}"
    return {"final_output": final}


# 构建父图：使用子图作为节点
parent = StateGraph(ParentState)
parent.add_node("translate_module", subgraph_app)   # 子图作为节点
parent.add_node("format", format_node)
parent.add_edge(START, "translate_module")
parent.add_edge("translate_module", "format")
parent.add_edge("format", END)
parent_app = parent.compile()

print("父图结构：START → translate_module（子图）→ format → END")
print("关键：父图和子图共享 'text' 和 'translated' 字段")

parent_input = {"text": "机器学习让计算机从数据中学习", "translated": "", "final_output": ""}
print(f"\n[父图调用测试]")
print(f"  输入：{parent_input['text']}")
parent_result = parent_app.invoke(parent_input)
print(f"  最终输出：\n{parent_result['final_output']}")


# ============================================================
# Part 3：State 共享——父图和子图共享 messages 字段
# ============================================================
print("\n--- Part 3：父子图共享 messages 字段 ---")
print("说明：当父图和子图有相同字段名时，子图可以读写父图的状态")

class SharedSubState(TypedDict):
    """共享 State 的子图：包含 messages 字段"""
    messages: Annotated[List, add_messages]
    summary: str  # 子图生成的摘要


def summarize_node(state: SharedSubState) -> dict:
    """摘要节点：读取 messages 并生成摘要"""
    # 获取最后一条用户消息
    user_msgs = [m.content for m in state["messages"] if hasattr(m, "type") and m.type == "human"]
    last_msg = user_msgs[-1] if user_msgs else "无消息"
    response = llm.invoke(f"用一句话总结以下内容：{last_msg}")
    # 子图把新消息追加到 messages（父图可以看到）
    return {
        "summary": response.content,
        "messages": [AIMessage(content=f"[摘要] {response.content}")]
    }


# 构建共享 State 的子图
shared_sub = StateGraph(SharedSubState)
shared_sub.add_node("summarize", summarize_node)
shared_sub.add_edge(START, "summarize")
shared_sub.add_edge("summarize", END)
shared_sub_app = shared_sub.compile()


class SharedParentState(TypedDict):
    """父图 State：包含与子图共享的 messages 字段"""
    messages: Annotated[List, add_messages]
    summary: str
    topic: str


def init_node(state: SharedParentState) -> dict:
    """初始化节点：把用户 topic 加入 messages"""
    return {"messages": [HumanMessage(content=state["topic"])]}


def finalize_node(state: SharedParentState) -> dict:
    """结束节点：展示父图收到了子图追加的消息"""
    msg_count = len(state["messages"])
    return {"messages": [AIMessage(content=f"[父图] 共收集到 {msg_count} 条消息，摘要已生成")]}


# 构建父图
shared_parent = StateGraph(SharedParentState)
shared_parent.add_node("init", init_node)
shared_parent.add_node("summarize_module", shared_sub_app)  # 共享 State 的子图
shared_parent.add_node("finalize", finalize_node)
shared_parent.add_edge(START, "init")
shared_parent.add_edge("init", "summarize_module")
shared_parent.add_edge("summarize_module", "finalize")
shared_parent.add_edge("finalize", END)
shared_parent_app = shared_parent.compile()

print("父图结构：START → init → summarize_module（子图）→ finalize → END")
print("共享字段：messages（父子图都可以读写）")

shared_input = {
    "messages": [],
    "summary": "",
    "topic": "深度学习在自然语言处理中的应用"
}
print(f"\n[共享 State 测试]")
print(f"  主题：{shared_input['topic']}")
shared_result = shared_parent_app.invoke(shared_input)
print(f"  子图生成的摘要：{shared_result['summary']}")
print(f"  最终消息数：{len(shared_result['messages'])}")
print("  消息记录：")
for msg in shared_result["messages"]:
    tag = "[用户]" if hasattr(msg, "type") and msg.type == "human" else "[AI]"
    print(f"    {tag} {str(msg.content)[:60]}")


def main():
    print("\n--- 总结 ---")
    print("子图的核心价值：")
    print("  1. 模块化：把复杂逻辑封装成独立可测试的单元")
    print("  2. 复用性：同一个子图可以被多个父图使用")
    print("  3. 团队协作：各团队独立开发和维护各自的子图")
    print("\nState 共享规则：")
    print("  - 同名字段：子图直接读写父图的字段")
    print("  - 不同字段名：需要包装函数做字段映射")
    print("\n语法总结：")
    print("  subgraph_app = sub_graph.compile()")
    print("  parent.add_node('module', subgraph_app)  # 子图即节点")


if __name__ == "__main__":
    main()
