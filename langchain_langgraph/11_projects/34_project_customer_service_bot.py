"""
主题：实战项目 —— 智能客服机器人（完整实现）

项目目标：
  构建一个生产级的智能客服系统，能够：
  1. 理解用户问题（意图识别）
  2. 检索相关知识（RAG）
  3. 生成准确回答（带引用）
  4. 必要时转人工（Human-in-the-Loop）
  5. 持续学习改进（反馈收集）

技术栈：
  - LangGraph：工作流编排
  - RAG：知识检索
  - FastAPI：API 服务
  - PostgreSQL：持久化存储

前置知识：已完成前面所有模块
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=512)


# =============================================================================
# Part 1：定义状态和节点
# =============================================================================

class CustomerServiceState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str              # 用户意图
    urgency: str             # 紧急程度
    answer: str              # 生成的答案
    sources: List[str]       # 引用来源
    needs_human: bool        # 是否需要人工
    feedback: str            # 用户反馈


def intent_classifier_node(state: CustomerServiceState) -> dict:
    """
    意图分类节点
    
    识别用户问题的类型：
    - product_query：产品咨询
    - technical_support：技术支持
    - billing：账单问题
    - complaint：投诉
    - other：其他
    """
    print("\n  [Intent Classifier] 分析用户意图...")
    
    classifier_prompt = ChatPromptTemplate.from_template(
        """你是一个客服意图分类专家。请分析用户问题并分类。

用户问题：{question}

可选类别：
- product_query：产品功能、价格、规格等咨询
- technical_support：使用问题、故障排除
- billing：账单、支付、退款相关问题
- complaint：投诉、不满情绪表达
- other：其他类型

同时评估紧急程度：
- low：一般咨询
- medium：需要尽快处理
- high：紧急问题（如服务中断、资金损失）

输出格式（JSON）：
{{
  "intent": "类别",
  "urgency": "紧急程度",
  "confidence": 0.0-1.0
}}

只输出 JSON，不要解释："""
    )
    
    chain = classifier_prompt | llm | StrOutputParser()
    
    # 简化：实际应解析 JSON
    last_message = state["messages"][-1].content if state["messages"] else ""
    result = chain.invoke({"question": last_message})
    
    print(f"  [Intent Classifier] 结果：{result[:100]}")
    
    # 模拟解析（实际应使用 json.loads）
    return {
        "intent": "technical_support",
        "urgency": "medium"
    }


def knowledge_retriever_node(state: CustomerServiceState) -> dict:
    """
    知识检索节点（RAG）
    
    根据意图和问题检索相关知识文档
    """
    print("\n  [Knowledge Retriever] 检索相关知识...")
    
    # 模拟检索（实际应连接向量数据库）
    mock_knowledge_base = {
        "product_query": [
            "我们的产品支持 Python、Java、Node.js 等多种语言",
            "标准版价格为 $99/月，企业版为 $499/月"
        ],
        "technical_support": [
            "常见错误 'Connection Timeout' 的解决方法：检查网络连接和防火墙设置",
            "API 调用限流：默认 100 次/分钟，可申请提升配额"
        ],
        "billing": [
            "我们接受信用卡、PayPal、银行转账等多种支付方式",
            "退款政策：30 天内无理由全额退款"
        ]
    }
    
    intent = state.get("intent", "other")
    sources = mock_knowledge_base.get(intent, ["未找到相关知识"])
    
    print(f"  [Knowledge Retriever] 找到 {len(sources)} 条相关知识")
    
    return {"sources": sources}


def answer_generator_node(state: CustomerServiceState) -> dict:
    """
    答案生成节点
    
    基于检索的知识生成友好、准确的回答
    """
    print("\n  [Answer Generator] 生成回答...")
    
    generator_prompt = ChatPromptTemplate.from_template(
        """你是一位专业的客服代表。请基于以下知识回答用户问题。

用户问题：{question}

相关知识：
{knowledge}

要求：
1. 语气友好、专业
2. 答案简洁明了（不超过 200 字）
3. 如果知识不足，诚实地告知用户
4. 适当提供后续步骤建议

回答："""
    )
    
    last_message = state["messages"][-1].content if state["messages"] else ""
    knowledge = "\n".join(state.get("sources", []))
    
    chain = generator_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "question": last_message,
        "knowledge": knowledge
    })
    
    print(f"  [Answer Generator] 完成")
    
    return {"answer": answer}


def human_escalation_node(state: CustomerServiceState) -> dict:
    """
    人工升级节点
    
    当遇到以下情况时转人工：
    - 高紧急度问题
    - 用户明确要求
    - AI 置信度低
    - 投诉类问题
    """
    print("\n  [Human Escalation] 评估是否需要人工介入...")
    
    needs_human = False
    
    # 判断条件
    if state.get("urgency") == "high":
        needs_human = True
        print("  → 高紧急度，转人工")
    
    if state.get("intent") == "complaint":
        needs_human = True
        print("  → 投诉类问题，转人工")
    
    if not needs_human:
        print("  → AI 可处理，无需转人工")
    
    return {"needs_human": needs_human}


def response_formatter_node(state: CustomerServiceState) -> dict:
    """
    响应格式化节点
    
    将答案格式化为最终响应
    """
    print("\n  [Response Formatter] 格式化响应...")
    
    if state["needs_human"]:
        final_response = (
            f"{state['answer']}\n\n"
            f"⚠️  由于您的问题较为复杂，我们将为您转接人工客服，请稍候..."
        )
    else:
        final_response = state["answer"]
        
        # 添加引用来源
        if state.get("sources"):
            final_response += "\n\n📚 参考来源：\n"
            for i, source in enumerate(state["sources"], 1):
                final_response += f"{i}. {source[:80]}...\n"
    
    # 添加到消息历史
    formatted_message = AIMessage(content=final_response)
    
    return {"messages": [formatted_message]}


def feedback_collector_node(state: CustomerServiceState) -> dict:
    """
    反馈收集节点（Human-in-the-Loop）
    
    询问用户对回答的满意度
    """
    print("\n  [Feedback Collector] 等待用户反馈...")
    
    # 暂停，等待用户反馈
    feedback = interrupt(
        "请评价本次服务：\n"
        "- 输入 'good' 表示满意\n"
        "- 输入 'bad' 表示不满意\n"
        "- 输入其他内容提供具体反馈"
    )
    
    print(f"  [Feedback Collector] 收到反馈：{feedback}")
    
    return {"feedback": str(feedback)}


# =============================================================================
# Part 2：构建图
# =============================================================================

def should_escalate_to_human(state: CustomerServiceState) -> str:
    """条件路由：是否需要人工"""
    if state["needs_human"]:
        return "human_escalation"
    return "generate_answer"


def build_customer_service_graph():
    """构建客服机器人图"""
    graph = StateGraph(CustomerServiceState)
    
    # 添加节点
    graph.add_node("classify_intent", intent_classifier_node)
    graph.add_node("retrieve_knowledge", knowledge_retriever_node)
    graph.add_node("check_escalation", human_escalation_node)
    graph.add_node("generate_answer", answer_generator_node)
    graph.add_node("format_response", response_formatter_node)
    graph.add_node("collect_feedback", feedback_collector_node)
    graph.add_node("human_escalation", lambda s: {"messages": [AIMessage(content="已转接人工客服")]})
    
    # 添加边
    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "check_escalation")
    
    # 条件分支
    graph.add_conditional_edges("check_escalation", should_escalate_to_human, {
        "human_escalation": "human_escalation",
        "generate_answer": "generate_answer"
    })
    
    graph.add_edge("generate_answer", "format_response")
    graph.add_edge("human_escalation", "format_response")
    graph.add_edge("format_response", "collect_feedback")
    graph.add_edge("collect_feedback", END)
    
    # 编译
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    
    return app


# =============================================================================
# Part 3：运行示例
# =============================================================================

def demo_customer_service():
    """演示客服机器人运行流程"""
    print("=" * 60)
    print("智能客服机器人演示")
    print("=" * 60)
    
    app = build_customer_service_graph()
    
    # 测试用例 1：技术咨询
    print("\n【测试 1】技术咨询")
    print("-" * 60)
    
    config = {"configurable": {"thread_id": "session_001"}}
    
    try:
        result = app.invoke({
            "messages": [HumanMessage(content="API 调用出现 Connection Timeout 错误怎么办？")],
            "intent": "",
            "urgency": "",
            "answer": "",
            "sources": [],
            "needs_human": False,
            "feedback": ""
        }, config)
        
        print(f"\n【最终回答】")
        print(result["messages"][-1].content)
    
    except Exception as e:
        print(f"执行中断（等待人工输入）：{e}")
    
    # 测试用例 2：投诉（需转人工）
    print("\n\n【测试 2】投诉（预期转人工）")
    print("-" * 60)
    
    config2 = {"configurable": {"thread_id": "session_002"}}
    
    try:
        result = app.invoke({
            "messages": [HumanMessage(content="你们的服务太差了！我要投诉！")],
            "intent": "",
            "urgency": "",
            "answer": "",
            "sources": [],
            "needs_human": False,
            "feedback": ""
        }, config2)
        
        print(f"\n【最终回答】")
        print(result["messages"][-1].content)
    
    except Exception as e:
        print(f"执行中断（等待人工输入）：{e}")


# =============================================================================
# Part 4：部署代码
# =============================================================================

def demo_deployment_code():
    """展示完整的部署代码结构"""
    print("\n" + "=" * 60)
    print("Part 4: Deployment Code Structure")
    print("=" * 60)
    
    project_structure = """
  📁 项目目录结构：
  ──────────────────────────────────────────────
  
  customer_service_bot/
  ├── api/
  │   ├── __init__.py
  │   ├── main.py              # FastAPI 主应用
  │   ├── routes.py            # API 路由
  │   └── schemas.py           # Pydantic 模型
  │
  ├── graph/
  │   ├── __init__.py
  │   ├── state.py             # State 定义
  │   ├── nodes.py             # 节点函数
  │   └── workflow.py          # 图构建
  │
  ├── retrieval/
  │   ├── __init__.py
  │   ├── vectorstore.py       # 向量数据库
  │   └── embeddings.py        # Embedding 模型
  │
  ├── config/
  │   ├── settings.py          # 配置管理
  │   └── logging_config.py    # 日志配置
  │
  ├── tests/
  │   ├── test_nodes.py        # 节点测试
  │   ├── test_workflow.py     # 工作流测试
  │   └── fixtures.py          # 测试夹具
  │
  ├── data/
  │   ├── knowledge_base/      # 知识库文档
  │   └── test_questions.json  # 测试问题集
  │
  ├── Dockerfile
  ├── docker-compose.yml
  ├── requirements.txt
  ├── .env.example
  └── README.md
  
  
  📄 requirements.txt：
  ──────────────────────────────────────────────
  
  langchain>=1.0.0
  langgraph>=0.3.0
  langchain-anthropic>=0.3.0
  fastapi>=0.109.0
  uvicorn>=0.27.0
  pydantic>=2.5.0
  psycopg2-binary>=2.9.9     # PostgreSQL
  redis>=5.0.1               # 缓存
  pytest>=7.4.0              # 测试
  ragas>=0.1.0               # 评估
  python-dotenv>=1.0.0
    """
    
    print(project_structure)


def main():
    print("=" * 60)
    print("实战项目：智能客服机器人")
    print("=" * 60)
    
    demo_customer_service()
    demo_deployment_code()
    
    print("\n" + "=" * 60)
    print("项目完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
