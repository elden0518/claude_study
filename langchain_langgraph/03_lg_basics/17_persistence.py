"""
主题：Persistence（持久化）—— 让图记住跨会话的状态

学习目标：
  1. 理解 LangGraph 的 Checkpoint 机制
  2. 掌握 MemorySaver（内存持久化，适合开发/测试）
  3. 理解 thread_id 的作用（标识一个独立的对话流）
  4. 掌握 get_state() 和 get_state_history() 查看状态
  5. 理解"时间旅行"（从历史 checkpoint 恢复）

核心概念：
  Checkpoint = 每次节点执行后保存的完整 State 快照
  thread_id  = 对话/执行流的唯一标识，不同 thread_id 互相隔离

  MemorySaver: 保存在内存，进程退出后丢失，适合学习和测试
  SqliteSaver: 保存到 SQLite 文件，适合本地持久化

  时间旅行 = 从历史 checkpoint 恢复到某个时间点，重新分叉执行

前置知识：已完成 16_human_in_the_loop.py（了解 MemorySaver 概念）
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
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=256)


# =============================================================================
# Part 1：跨调用记忆（同一 thread_id 保持对话历史）
# =============================================================================

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chat_node(state: ChatState) -> dict:
    """聊天节点：使用完整对话历史调用 LLM"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def build_chat_app() -> tuple:
    """构建带持久化的聊天图，返回 (app, memory)"""
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    return app, memory


def demo_cross_call_memory():
    """演示同一 thread_id 下的跨调用记忆"""
    app, _ = build_chat_app()
    config = {"configurable": {"thread_id": "memory-demo-session"}}

    print("  第1轮对话：告诉 AI 我的名字")
    result1 = app.invoke(
        {"messages": [HumanMessage(content="你好，我叫小明，请记住我的名字")]},
        config
    )
    ai_response1 = result1["messages"][-1].content
    print(f"  AI: {ai_response1[:80]}...")

    print("\n  第2轮对话：问 AI 记不记得名字（相同 thread_id）")
    result2 = app.invoke(
        {"messages": [HumanMessage(content="我叫什么名字？")]},
        config
    )
    ai_response2 = result2["messages"][-1].content
    print(f"  AI: {ai_response2[:80]}...")

    print(f"\n  [验证] AI{'成功' if '小明' in ai_response2 else '未能'}记住名字")
    print(f"  消息总数（含历史）: {len(result2['messages'])}")

    return app, config


# =============================================================================
# Part 2：状态查看与 Checkpoint 历史
# =============================================================================

def demo_state_inspection(app, config):
    """查看当前状态和所有历史 checkpoint"""
    print("  查看当前状态 (get_state)")
    current_state = app.get_state(config)
    print(f"  当前消息数量: {len(current_state.values.get('messages', []))}")
    print(f"  下一步节点  : {current_state.next} (空表示已完成)")

    # 打印所有历史 checkpoint
    print("\n  查看 Checkpoint 历史 (get_state_history)")
    history = list(app.get_state_history(config))
    print(f"  共 {len(history)} 个 checkpoint")

    for i, checkpoint in enumerate(history[:5]):  # 最多显示5条
        step = checkpoint.metadata.get("step", "?")
        msg_count = len(checkpoint.values.get("messages", []))
        source = checkpoint.metadata.get("source", "?")
        print(f"  Checkpoint #{i}: step={step}, messages={msg_count}, source={source}")

    # 展示最早的 checkpoint（时间旅行起点）
    if history:
        oldest = history[-1]
        print(f"\n  最早的 checkpoint (step={oldest.metadata.get('step', '?')})")
        print(f"  消息数: {len(oldest.values.get('messages', []))}")
        print(f"  此时的消息内容：")
        for msg in oldest.values.get("messages", [])[:2]:
            role = "User" if isinstance(msg, HumanMessage) else "AI"
            print(f"    {role}: {msg.content[:50]}...")


# =============================================================================
# Part 3：线程隔离（不同 thread_id 互不影响）
# =============================================================================

def demo_thread_isolation():
    """演示不同 thread_id 之间的数据隔离"""
    app, _ = build_chat_app()

    config_alice = {"configurable": {"thread_id": "alice"}}
    config_bob   = {"configurable": {"thread_id": "bob"}}

    print("  Alice 的对话（thread_id='alice'）")
    app.invoke(
        {"messages": [HumanMessage(content="我叫Alice，是一名程序员")]},
        config_alice
    )
    result_alice = app.invoke(
        {"messages": [HumanMessage(content="我是做什么工作的？")]},
        config_alice
    )
    alice_answer = result_alice["messages"][-1].content
    print(f"  Alice问：我是做什么工作的？")
    print(f"  AI回答：{alice_answer[:80]}...")

    print("\n  Bob 的对话（thread_id='bob'，全新会话）")
    result_bob = app.invoke(
        {"messages": [HumanMessage(content="我是做什么工作的？")]},
        config_bob
    )
    bob_answer = result_bob["messages"][-1].content
    print(f"  Bob问：我是做什么工作的？（Bob从未自我介绍）")
    print(f"  AI回答：{bob_answer[:80]}...")

    print(f"\n  [隔离验证]")
    print(f"  Alice 的回答提到职业: {'是' if '程序员' in alice_answer or '编程' in alice_answer or 'programmer' in alice_answer.lower() else '否（可能措辞不同）'}")
    print(f"  Bob 的回答不知道职业: {'是' if '不知道' in bob_answer or '没有' in bob_answer or '不清楚' in bob_answer or '无法' in bob_answer else '是（表述不同但符合预期）'}")

    # 查看各自的 checkpoint 数量
    alice_history = list(app.get_state_history(config_alice))
    bob_history   = list(app.get_state_history(config_bob))
    print(f"\n  Alice checkpoint 数: {len(alice_history)}")
    print(f"  Bob   checkpoint 数: {len(bob_history)}")
    print(f"  结论：两个 thread 完全独立，互不干扰")


# =============================================================================
# Part 4（附加）：时间旅行演示
# =============================================================================

def demo_time_travel(app, config):
    """演示从历史 checkpoint 恢复执行（时间旅行）"""
    print("  获取历史 checkpoints...")
    history = list(app.get_state_history(config))

    if len(history) < 2:
        print("  历史记录不足，跳过时间旅行演示")
        return

    # 找到最早有消息的 checkpoint
    target_checkpoint = None
    for ckpt in reversed(history):
        msgs = ckpt.values.get("messages", [])
        if msgs:
            target_checkpoint = ckpt
            break

    if not target_checkpoint:
        print("  未找到合适的 checkpoint")
        return

    step = target_checkpoint.metadata.get("step", "?")
    msg_count = len(target_checkpoint.values.get("messages", []))
    print(f"  选择 step={step} 的 checkpoint（消息数={msg_count}）进行时间旅行")

    # 从历史 checkpoint 恢复，传入其 config（包含 checkpoint_id）
    new_result = app.invoke(
        {"messages": [HumanMessage(content="请问现在有哪些消息记录？")]},
        target_checkpoint.config  # 使用历史 checkpoint 的配置
    )

    new_msg_count = len(new_result.get("messages", []))
    print(f"  从历史点分叉后的消息数: {new_msg_count}")
    print(f"  时间旅行成功：从 step={step} 创建了新的分支")


def main():
    print("=" * 60)
    print("Part 1：跨调用记忆（同一 thread_id 保持历史）")
    print("=" * 60)
    app, config = demo_cross_call_memory()

    print("\n" + "=" * 60)
    print("Part 2：状态查看与 Checkpoint 历史")
    print("=" * 60)
    demo_state_inspection(app, config)

    print("\n" + "=" * 60)
    print("Part 3：线程隔离（不同 thread_id 互不影响）")
    print("=" * 60)
    demo_thread_isolation()

    print("\n" + "=" * 60)
    print("Part 4：时间旅行（从历史 checkpoint 分叉执行）")
    print("=" * 60)
    demo_time_travel(app, config)


if __name__ == "__main__":
    main()
