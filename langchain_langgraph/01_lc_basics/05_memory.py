"""
主题：Memory（记忆）—— 让对话有上下文

学习目标：
  1. 理解 LLM 本身无状态，Memory 是外部维护历史的机制
  2. 掌握手动管理消息历史（最透明的方式）
  3. 掌握 ChatMessageHistory（标准历史存储）
  4. 掌握 RunnableWithMessageHistory（LCEL 链的历史管理包装器）
  5. 对比不同 Memory 策略：完整历史 vs 截断历史

核心概念：
  LLM 的"记忆"本质：把历史消息作为上下文一起发给模型
  每次对话都要携带完整历史 → token 越来越多 → 需要管理策略

  完整历史（Buffer）: 保留所有消息，简单但 token 多
  截断历史（Trim）: 只保留最近 N 轮，节省 token
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=256)


# =============================================================================
# Part 1：手动管理历史（最透明的方式）
# =============================================================================

def demo_manual_history():
    """手动维护消息列表 —— 最直接，最透明"""
    from langchain_core.messages import HumanMessage, AIMessage

    history = []

    def chat(user_input: str) -> str:
        history.append(HumanMessage(content=user_input))
        response = llm.invoke(history)
        history.append(AIMessage(content=response.content))
        return response.content

    print("── 手动历史对话 ──")
    r1 = chat("我叫小明，今年学Python")
    print(f"  User: 我叫小明，今年学Python")
    print(f"  AI: {r1[:60]}...")

    r2 = chat("我的名字是什么？")
    print(f"  User: 我的名字是什么？")
    print(f"  AI: {r2}")

    print(f"  历史消息数: {len(history)}")


# =============================================================================
# Part 2：RunnableWithMessageHistory（LCEL 链的历史管理）
# =============================================================================

def demo_runnable_with_history():
    """用 RunnableWithMessageHistory 包装链，自动管理历史"""

    # 构建带历史插槽的 prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位友好的编程助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()

    # 存储每个 session 的历史
    store: dict = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # 用 RunnableWithMessageHistory 包装
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config_alice = {"configurable": {"session_id": "alice"}}
    config_bob   = {"configurable": {"session_id": "bob"}}

    print("── session: alice ──")
    r = chain_with_history.invoke({"input": "我叫Alice，我在学习数据分析"}, config=config_alice)
    print(f"  AI: {r[:60]}...")
    r = chain_with_history.invoke({"input": "我的名字是什么？"}, config=config_alice)
    print(f"  AI: {r}")

    print("\n── session: bob（独立历史）──")
    r = chain_with_history.invoke({"input": "我的名字是什么？"}, config=config_bob)
    print(f"  AI: {r}")  # bob 没有历史，不知道名字

    print(f"\n  Alice 历史消息数: {len(store['alice'].messages)}")


# =============================================================================
# Part 3：历史管理策略对比
# =============================================================================

def demo_history_strategies():
    """对比完整历史 vs 手动截断的 token 消耗"""
    from langchain_core.messages import HumanMessage, AIMessage

    # 模拟10轮对话后的历史
    long_history = []
    langs = ['Python','Java','Go','Rust','JS','TS','C++','Swift','Kotlin','Ruby']
    for i in range(10):
        long_history.append(HumanMessage(content=f"第{i+1}轮问题：{langs[i]}有什么特点？"))
        long_history.append(AIMessage(content=f"第{i+1}轮回答：这是一种编程语言，有各自特点。"))

    print(f"── 历史策略对比 ──")
    print(f"  10轮完整历史消息数: {len(long_history)}")

    # 策略1：保留最近 N 轮
    keep_last_n = 4
    trimmed = long_history[-keep_last_n * 2:]
    print(f"  保留最近{keep_last_n}轮后消息数: {len(trimmed)}")
    print(f"  → 适合：对话内容无强依赖，只需近期上下文")
    print(f"  → 优点：节省token，降低成本")

    # 策略2：完整保留
    print(f"\n  完整历史消息数: {len(long_history)}")
    print(f"  → 适合：对话内容高度依赖早期信息（如姓名、需求）")
    print(f"  → 代价：token 随对话轮数线性增长")


def main():
    print("=" * 60)
    print("Part 1：手动管理历史")
    print("=" * 60)
    demo_manual_history()

    print("\n" + "=" * 60)
    print("Part 2：RunnableWithMessageHistory")
    print("=" * 60)
    demo_runnable_with_history()

    print("\n" + "=" * 60)
    print("Part 3：历史管理策略对比")
    print("=" * 60)
    demo_history_strategies()


if __name__ == "__main__":
    main()
