"""
主题：Hello LangChain —— 第一次调用，理解它与直接用 SDK 的区别

学习目标：
  1. 理解 LangChain 是什么：统一接口层，不是 AI 模型本身
  2. 掌握 ChatAnthropic 的初始化和基本调用
  3. 对比 LangChain 与直接用 anthropic SDK 的写法差异
  4. 掌握三种调用方式：invoke / stream / batch
  5. 理解 AIMessage 响应对象的结构

核心概念：
  LangChain 不是模型，是框架。它把不同模型（Claude、GPT、Gemini）
  包装成统一接口，让你的代码可以无缝切换模型。

  ChatAnthropic  ←→  anthropic.Anthropic().messages.create()
  chain.invoke() ←→  client.messages.create()

前置知识：已完成 01_basics/01_hello_claude.py
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
from dotenv import load_dotenv
load_dotenv()

# ── LangChain 的 Anthropic 集成包 ──────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

MODEL = "ppio/pa/claude-sonnet-4-6"


# =============================================================================
# Part 1：直接用 anthropic SDK vs 用 LangChain（对比）
# =============================================================================

def demo_direct_sdk():
    """原生 SDK 调用方式（参照系）"""
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": "用一句话介绍Python"}]
    )
    print(f"[原生SDK] {resp.content[0].text}")


def demo_langchain_invoke():
    """LangChain 调用方式 —— invoke（同步，返回 AIMessage）"""
    llm = ChatAnthropic(model=MODEL, max_tokens=128)

    # invoke 接受字符串或消息列表
    response = llm.invoke("用一句话介绍Python")

    # response 是 AIMessage 对象
    print(f"[LangChain invoke] {response.content}")
    print(f"  类型: {type(response).__name__}")
    print(f"  usage: {response.usage_metadata}")


# =============================================================================
# Part 2：传入消息列表（System + Human）
# =============================================================================

def demo_messages():
    """使用消息列表，等价于设置 system prompt"""
    llm = ChatAnthropic(model=MODEL, max_tokens=256)

    messages = [
        SystemMessage(content="你是一位资深 Python 工程师，回答简洁。"),
        HumanMessage(content="什么是列表推导式？给一个例子。"),
    ]

    response = llm.invoke(messages)
    print(f"[消息列表调用]\n{response.content}")


# =============================================================================
# Part 3：流式输出
# =============================================================================

def demo_streaming():
    """stream() 方法 —— 实时输出每个 token"""
    llm = ChatAnthropic(model=MODEL, max_tokens=256)

    print("[流式输出] ", end="", flush=True)
    for chunk in llm.stream("请列举 Python 3 的三个新特性"):
        print(chunk.content, end="", flush=True)
    print()


# =============================================================================
# Part 4：批量调用（batch）
# =============================================================================

def demo_batch():
    """batch() 方法 —— 并发处理多个输入"""
    llm = ChatAnthropic(model=MODEL, max_tokens=64)

    questions = [
        "什么是 Python？一句话。",
        "什么是 JavaScript？一句话。",
        "什么是 Rust？一句话。",
    ]

    # batch 并发调用，返回列表
    responses = llm.batch(questions)
    print("[批量调用]")
    for q, r in zip(questions, responses):
        print(f"  Q: {q}")
        print(f"  A: {r.content}\n")


def main():
    print("=" * 60)
    print("Part 1：原生 SDK vs LangChain 对比")
    print("=" * 60)
    demo_direct_sdk()
    demo_langchain_invoke()

    print("\n" + "=" * 60)
    print("Part 2：消息列表（System + Human）")
    print("=" * 60)
    demo_messages()

    print("\n" + "=" * 60)
    print("Part 3：流式输出")
    print("=" * 60)
    demo_streaming()

    print("\n" + "=" * 60)
    print("Part 4：批量调用")
    print("=" * 60)
    demo_batch()


if __name__ == "__main__":
    main()
