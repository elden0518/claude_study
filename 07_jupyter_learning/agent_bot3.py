import os
from typing import TypedDict, List, Union, Annotated, Sequence
from langchain_core.messages import HumanMessage,AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="deepseek-v4-pro",api_key=os.getenv("DEEPSEEK_API_KEY"),
                 base_url=os.getenv("DEEPSEEK_API_BASE_URL"))

class AgentState(TypedDict):
  """
  Agent 的状态
  """
  messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b: int) -> int:
  """两个整数相加，返回两个数的和"""
  print(f"[工具执行] add({a}, {b})")
  return a + b

@tool
def multiply(a:int, b: int) -> int:
  """两个整数相乘，返回两个数的积"""
  print(f"[工具执行] multiply({a}, {b})")
  return a * b

@tool
def subtract(a:int, b: int) -> int:
  """两个整数相减，返回两个数的差"""
  print(f"[工具执行] subtract({a}, {b})")
  return a - b

tools = [add, multiply, subtract]

def model_call(state : AgentState) -> AgentState:
    """模型调用节点：绑定工具后调用 LLM"""
    # 关键：必须绑定工具，LLM 才知道有哪些工具可用
    llm_with_tools = llm.bind_tools(tools)
    
    # 构建系统提示
    system_message = SystemMessage(content="""
      你是一个智能助手，你需要根据用户输入的指令，进行相应的处理。
      如果需要使用工具，请调用相应的工具。
      """)
    
    # 组合系统消息和历史消息
    messages = [system_message] + list(state["messages"])
    
    # 调用 LLM
    response = llm_with_tools.invoke(messages)
    print(f"  [LLM 节点] 响应类型：{'工具调用' if response.tool_calls else '普通回复'}")
    
    return {"messages": [response]}

def should_continue(state: AgentState) :
    """判断是否继续对话"""
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"  # 修正拼写错误

graph = StateGraph(AgentState)
graph.add_node("model_call", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("model_call")
graph.add_conditional_edges(
    "model_call",
    should_continue,
    {
        "continue": "tool_node",  # 修正拼写错误
        "end": END
    }
)

graph.add_edge("tool_node", "model_call")
app = graph.compile()

def print_stream(stream):
    for chunk in stream:
        message = chunk["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [HumanMessage(content="请计算 1 + 1，以及计算 33 + 99的和再乘以5的值")]}

print_stream(app.stream(inputs, stream_mode="values"))



