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

document_content = ""

@tool
def update(content: str) -> str:
    """根据提供的内容进行更新文档"""
    global document_content
    document_content = content
    return f"文档内容已更新,内容是：{document_content}"

@tool
def save(file_name: str) -> str:
    """保存文档内容到指定文件

    参数：
        file_name:文件名，以.txt为结尾
    """
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(document_content)
        print(f"文件保存成功：{file_name}")
        return f"文件保存成功：{file_name}"
    except Exception as e:
        return f"保存文件时出错：{e}"

tools = [update, save]
llm.bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = """
      You are Drafter, a helpful writing assistant. Youare going to help the user update and modify docunments
        - If the user wants to update or modify content, usethe 'update' tool with the complete updated content.
        - If the user wants to save and finish, you need tto use the 'save' tool.
        - Make sure to always show the current documentstate after modifications
      The current document content is:{document_content] 
      """
    if not state["messages"]:
        user_input="我已经准备好帮你创建一个新文档，需要我怎么样开始？"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\n 你想怎么处理你的文档呢？")
        user_message = HumanMessage(content=user_input)

    all_messages=[system_prompt] + list(state["messages"]) + [user_message]
    response = llm.invoke(all_messages)

    print( response.content)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call.name
            tool_args = tool_call.args
            print(f"[工具调用] {tool_name}")
            print(f"[工具参数] {tool_args}")
            tool_output = tools[tool_name](**tool_args)
    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) :
    """判断是否继续对话"""
    messages = state["messages"]
    if not messages:
        return "continue"

    for message in reversed(messages):
        if(isinstance(message, ToolMessage) and
            "saved" in message.content.lower() and
            "document" in message.content.lower() ):
            return "end"
        return "continue"






















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



