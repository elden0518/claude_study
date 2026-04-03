"""
主题：Prompt Templates —— 把提示词变成可复用的模板

学习目标：
  1. 理解为什么需要 Prompt Template（复用、版本管理、动态注入）
  2. 掌握 PromptTemplate（纯文本）的创建和使用
  3. 掌握 ChatPromptTemplate（对话格式）的创建和使用
  4. 学会 MessagesPlaceholder（插入动态消息列表）
  5. 掌握 few-shot prompt（示例驱动提示）

核心概念：
  Prompt Template = 带变量的字符串模板
  {变量名} 占位符在 invoke 时被替换为实际值

  PromptTemplate       → 单一字符串，适合补全模型
  ChatPromptTemplate   → 消息列表，适合对话模型（Claude、GPT）
  MessagesPlaceholder  → 在模板中插入动态的消息历史
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=256)


# =============================================================================
# Part 1：PromptTemplate（纯文本）
# =============================================================================

def demo_prompt_template():
    """基础字符串模板，用 {变量} 占位"""
    template = PromptTemplate.from_template(
        "请用{language}写一个函数，功能是{task}。只输出代码，不要解释。"
    )

    # format_prompt() → 查看渲染后的完整 prompt
    rendered = template.format(language="Python", task="计算斐波那契数列")
    print(f"[渲染后的 Prompt]\n{rendered}\n")

    # 与 LLM 串联（| 操作符，LCEL 基础）
    from langchain_core.output_parsers import StrOutputParser
    chain = template | llm | StrOutputParser()
    result = chain.invoke({"language": "Python", "task": "反转字符串"})
    print(f"[模型输出]\n{result}")


# =============================================================================
# Part 2：ChatPromptTemplate（对话格式）
# =============================================================================

def demo_chat_prompt_template():
    """对话格式模板，包含 system/human/ai 角色"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位{expertise}专家，用{style}风格回答。"),
        ("human", "{question}"),
    ])

    # format_messages() → 查看渲染后的消息列表
    messages = prompt.format_messages(
        expertise="Python",
        style="简洁",
        question="什么是装饰器？"
    )
    print(f"[渲染后的消息列表]")
    for msg in messages:
        print(f"  {msg.__class__.__name__}: {msg.content[:50]}")

    # 与 LLM 串联
    from langchain_core.output_parsers import StrOutputParser
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "expertise": "数据库",
        "style": "举例子",
        "question": "什么是索引？"
    })
    print(f"\n[模型输出]\n{result}")


# =============================================================================
# Part 3：MessagesPlaceholder（插入动态历史）
# =============================================================================

def demo_messages_placeholder():
    """在模板中插入动态的对话历史，用于多轮对话"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位耐心的编程助手。"),
        MessagesPlaceholder(variable_name="history"),  # 动态插入历史消息
        ("human", "{input}"),
    ])

    # 模拟已有对话历史
    history = [
        HumanMessage(content="我在学Python"),
        AIMessage(content="很好！Python 是很棒的入门语言。"),
    ]

    from langchain_core.output_parsers import StrOutputParser
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "history": history,
        "input": "我应该先学哪个部分？"
    })
    print(f"[含历史的对话]\n{result}")


# =============================================================================
# Part 4：Few-shot Prompt（示例驱动）
# =============================================================================

def demo_few_shot():
    """通过示例告诉模型期望的输入输出格式"""
    examples = [
        {"input": "快乐", "output": "悲伤"},
        {"input": "黑暗", "output": "光明"},
        {"input": "寒冷", "output": "温暖"},
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "请给出输入词的反义词，只输出一个词。"),
        few_shot_prompt,
        ("human", "{input}"),
    ])

    from langchain_core.output_parsers import StrOutputParser
    chain = final_prompt | llm | StrOutputParser()
    result = chain.invoke({"input": "成功"})
    print(f"[Few-shot 反义词] 成功 → {result}")


def main():
    print("=" * 60)
    print("Part 1：PromptTemplate（纯文本）")
    print("=" * 60)
    demo_prompt_template()

    print("\n" + "=" * 60)
    print("Part 2：ChatPromptTemplate（对话格式）")
    print("=" * 60)
    demo_chat_prompt_template()

    print("\n" + "=" * 60)
    print("Part 3：MessagesPlaceholder（动态历史）")
    print("=" * 60)
    demo_messages_placeholder()

    print("\n" + "=" * 60)
    print("Part 4：Few-shot Prompt")
    print("=" * 60)
    demo_few_shot()


if __name__ == "__main__":
    main()
