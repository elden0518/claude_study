import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
  """
  Agent 的状态
  """
  messages: List[HumanMessage | AIMessage]


llm = ChatOpenAI(model="deepseek-v4-pro",api_key=os.getenv("DEEPSEEK_API_KEY"),
                 base_url=os.getenv("DEEPSEEK_API_BASE_URL"))

def process_agent(state: AgentState) -> AgentState:
  """处理 Agent"""
  print(f"  [process_agent] 处理: {state['messages']}")
  response = llm.invoke(state["messages"])
  print(f"  [process_agent] 响应: {response.content}")
  state["messages"].append(AIMessage(content=response.content))
  return state

graph = StateGraph(AgentState)
graph.add_node("process_node", process_agent)
graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)
app = graph.compile()


conversation_history = []

user_input = input("请输入：")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages":conversation_history})
    print(result["messages"])
    conversation_history = result["messages"]
    user_input=input("请输入：")

with open("history.txt",'w', encoding='utf-8') as file:
    file.write("write history~\n")
    for message in conversation_history:
        if isinstance(message,HumanMessage):
            file.write(f"Human: {message.content}\n")
        elif isinstance(message,AIMessage):
            file.write(f"AI: {message.content}\n")
    file.write("\nend history~\n")

print("对话结束")