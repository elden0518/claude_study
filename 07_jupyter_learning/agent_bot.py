import os
from typing import TypedDict, List
from langchain_core.messages import HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
  """
  Agent 的状态
  """
  messages: List[HumanMessage]


llm = ChatOpenAI(model="deepseek-v4-pro",api_key=os.getenv("DEEPSEEK_API_KEY"),
                 base_url=os.getenv("DEEPSEEK_API_BASE_URL"))

def process_agent(state: AgentState) -> AgentState:
  """处理 Agent"""
  print(f"  [process_agent] 处理: {state['messages']}")
  response = llm.invoke(state["messages"])
  print(f"  [process_agent] 响应: {response}")
  return state

graph = StateGraph(AgentState)
graph.add_node("process_node", process_agent)
graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)
app = graph.compile()

user_input = input("请输入：")
while user_input != "exit":
  app.invoke({"messages":[HumanMessage(content=user_input)]})
  user_input=input("请输入：")