import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
主题：ToolNode —— LangGraph 原生的工具调用节点

学习目标：
  1. 理解 ToolNode 的作用：自动处理 LLM 的工具调用请求
  2. 学会手动搭建 ReAct Agent 图（LLM节点 ↔ 工具节点的循环）
  3. 掌握 tools_condition（判断是否需要调用工具的条件函数）
  4. 掌握 create_react_agent（一行创建完整 Agent）
  5. 对比手动搭建 vs create_react_agent 的代码量差异

核心概念：
  ToolNode = 专门处理工具调用的节点
  接收含 tool_calls 的 AIMessage → 执行工具 → 返回 ToolMessage

  手动 ReAct 图结构：
  START → llm_node → tools_condition → [tool_node → llm_node（循环）| END]

  create_react_agent = 把上面的图封装成一行代码

前置知识：已完成 11_tools_agents.py（LangChain ReAct）
"""

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent

load_dotenv()

# 初始化 LLM
llm = ChatAnthropic(model="ppio/pa/claude-sonnet-4-6")

print("=" * 60)
print("LangGraph 20：ToolNode —— 原生工具调用节点")
print("=" * 60)


# ============================================================
# Part 1：定义工具和 ToolNode
# ============================================================
print("\n--- Part 1：定义工具和 ToolNode ---")
print("说明：ToolNode 自动识别 AIMessage 中的 tool_calls 并执行对应工具")


@tool
def multiply(a: int, b: int) -> int:
    """将两个整数相乘"""
    result = a * b
    print(f"    [工具执行] multiply({a}, {b}) = {result}")
    return result


@tool
def add_numbers(a: int, b: int) -> int:
    """将两个整数相加"""
    result = a + b
    print(f"    [工具执行] add_numbers({a}, {b}) = {result}")
    return result


@tool
def get_weather(city: str) -> str:
    """获取城市的模拟天气信息"""
    weather_data = {
        "北京": "晴天，18°C",
        "上海": "多云，22°C",
        "广州": "小雨，26°C",
        "深圳": "阴天，24°C",
    }
    result = weather_data.get(city, f"{city}：天气数据不可用")
    print(f"    [工具执行] get_weather({city}) = {result}")
    return result


# 工具列表
tools = [multiply, add_numbers, get_weather]

# 创建 ToolNode：传入工具列表，ToolNode 自动分发调用
tool_node = ToolNode(tools)

print(f"已定义 {len(tools)} 个工具：{[t.name for t in tools]}")
print("ToolNode：自动接收 AIMessage 的 tool_calls 并执行对应工具")


# ============================================================
# Part 2：手动搭建 ReAct Agent 图
# ============================================================
print("\n--- Part 2：手动搭建 ReAct Agent 图 ---")
print("说明：展示 ReAct 的完整图结构（LLM ↔ 工具 的循环）")


class AgentState(TypedDict):
    """Agent 的 State：使用 add_messages reducer 追加消息"""
    messages: Annotated[list, add_messages]


def llm_node(state: AgentState) -> dict:
    """LLM 节点：绑定工具后调用 LLM，LLM 决定是否使用工具"""
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    print(f"  [LLM 节点] 响应类型：{'工具调用' if response.tool_calls else '普通回复'}")
    return {"messages": [response]}


# 构建手动 ReAct 图
manual_graph = StateGraph(AgentState)
manual_graph.add_node("llm", llm_node)
manual_graph.add_node("tools", tool_node)

manual_graph.add_edge(START, "llm")
# tools_condition：检查最后一条消息是否含 tool_calls
# 有 tool_calls → 路由到 "tools"；无 tool_calls → 路由到 END
manual_graph.add_conditional_edges("llm", tools_condition)
manual_graph.add_edge("tools", "llm")  # 工具执行后返回 LLM（循环）
manual_app = manual_graph.compile()

print("手动 ReAct 图结构：")
print("  START → llm → tools_condition → tools → llm（循环）")
print("                              ↘ END（无工具调用时）")

query1 = "计算 15 乘以 7，然后告诉我北京的天气"
print(f"\n[手动 ReAct 测试]")
print(f"  问题：{query1}")
manual_result = manual_app.invoke({"messages": [HumanMessage(content=query1)]})
final_msg = manual_result["messages"][-1]
print(f"  最终回答：{str(final_msg.content)[:150]}")
print(f"  总消息数：{len(manual_result['messages'])}（包含工具调用记录）")


# ============================================================
# Part 3：create_react_agent（一行创建完整 Agent）
# ============================================================
print("\n--- Part 3：create_react_agent（一行代码）---")
print("说明：create_react_agent 把手动图的所有代码封装成一行")

# 一行代码创建完整的 ReAct Agent（等价于 Part 2 的手动图）
agent = create_react_agent(llm, tools)

print("create_react_agent(llm, tools) 等价于：")
print("  1. 创建 AgentState（含 add_messages reducer）")
print("  2. 创建 llm_node（llm.bind_tools(tools)）")
print("  3. 创建 ToolNode(tools)")
print("  4. 添加 tools_condition 条件边")
print("  5. 编译图")

query2 = "12 加 34 等于多少？"
print(f"\n[create_react_agent 测试]")
print(f"  问题：{query2}")
agent_result = agent.invoke({"messages": [HumanMessage(content=query2)]})

print("  执行轨迹（所有消息）：")
for msg in agent_result["messages"]:
    msg_type = type(msg).__name__
    content_preview = str(msg.content)[:80] if msg.content else "[tool_calls]"
    print(f"    {msg_type}: {content_preview}")

print(f"\n[对比总结]")
print(f"  手动搭建：约 20 行代码（显式定义每个节点和边）")
print(f"  create_react_agent：1 行代码（适合快速原型）")
print(f"  建议：学习时用手动搭建理解原理，生产中可用 create_react_agent")


def main():
    print("\n--- 总结 ---")
    print("ToolNode 的核心流程：")
    print("  1. LLM 返回含 tool_calls 的 AIMessage")
    print("  2. tools_condition 检测到 tool_calls，路由到 ToolNode")
    print("  3. ToolNode 执行工具，返回 ToolMessage")
    print("  4. ToolMessage 加入 messages，LLM 再次调用")
    print("  5. LLM 生成最终回复（无 tool_calls），路由到 END")
    print()
    print("关键 API：")
    print("  ToolNode(tools)          # 自动处理工具调用")
    print("  tools_condition          # 判断是否需要调用工具")
    print("  llm.bind_tools(tools)    # 让 LLM 知道有哪些工具可用")
    print("  create_react_agent(llm, tools)  # 一行创建 ReAct Agent")


if __name__ == "__main__":
    main()
