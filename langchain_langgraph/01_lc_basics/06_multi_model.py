"""
主题：多模型切换 —— LangChain 最核心价值：一套代码多个模型

学习目标：
  1. 理解 LangChain 的"模型无关性"设计理念
  2. 对比 ChatAnthropic 和 ChatOpenAI 的接口一致性
  3. 学会在运行时动态切换模型
  4. 理解不同模型的响应差异
  5. 掌握 configurable_alternatives（可配置模型）

核心概念：
  LangChain 的核心价值之一：把不同厂商的模型包装成统一的 BaseChatModel 接口
  ChatAnthropic、ChatOpenAI、ChatGoogleGenerativeAI 都实现了相同的接口
  → 切换模型 = 换一行初始化代码，其余代码不变

  注意：需要在 .env 中同时配置 ANTHROPIC_API_KEY 和 OPENAI_API_KEY
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

ANTHROPIC_MODEL = "ppio/pa/claude-sonnet-4-6"


# =============================================================================
# Part 1：统一接口演示
# =============================================================================

def demo_unified_interface():
    """用同一个函数调用不同模型 —— 接口完全一致"""

    prompt = ChatPromptTemplate.from_template("用一句话（中文）解释：{concept}")
    parser = StrOutputParser()

    # Claude（Anthropic）
    claude = ChatAnthropic(model=ANTHROPIC_MODEL, max_tokens=128)
    claude_chain = prompt | claude | parser

    concept = "量子纠缠"
    claude_result = claude_chain.invoke({"concept": concept})
    print(f"[Claude] {claude_result}")

    # OpenAI（如果有 key 则运行，否则跳过）
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        gpt = ChatOpenAI(model="gpt-4o-mini", max_tokens=128)
        gpt_chain = prompt | gpt | parser
        gpt_result = gpt_chain.invoke({"concept": concept})
        print(f"[GPT-4o-mini] {gpt_result}")
    else:
        print(f"[GPT-4o-mini] 跳过（未配置 OPENAI_API_KEY）")
        print(f"  → 如果换成 ChatOpenAI，代码完全一样，只改一行初始化")


# =============================================================================
# Part 2：可配置模型（运行时切换）
# =============================================================================

def demo_configurable_model():
    """使用 configurable_alternatives 在运行时选择模型"""

    # 定义一个以 Claude 为默认、可配置替换模型的链
    claude = ChatAnthropic(model=ANTHROPIC_MODEL, max_tokens=128).configurable_alternatives(
        which={"id": "llm"},   # 配置项名称
        default_key="claude",
        # 如果有 OpenAI Key，可以取消注释：
        # openai=ChatOpenAI(model="gpt-4o-mini"),
    )

    chain = (
        ChatPromptTemplate.from_template("用一句话介绍：{topic}")
        | claude
        | StrOutputParser()
    )

    # 使用默认模型（Claude）
    result = chain.invoke(
        {"topic": "深度学习"},
        config={"configurable": {"llm": "claude"}}
    )
    print(f"[可配置链 - Claude] {result}")


# =============================================================================
# Part 3：模型参数对比
# =============================================================================

def demo_model_params():
    """展示不同 temperature 参数的影响"""

    prompt = ChatPromptTemplate.from_template("写一首关于{topic}的四行短诗")
    parser = StrOutputParser()

    # 低温度（更确定性）
    cold = ChatAnthropic(model=ANTHROPIC_MODEL, max_tokens=128, temperature=0.0)
    # 高温度（更有创意）
    hot = ChatAnthropic(model=ANTHROPIC_MODEL, max_tokens=128, temperature=1.0)

    topic = "秋天"
    print(f"[temperature=0.0 - 确定性]\n{(prompt | cold | parser).invoke({'topic': topic})}")
    print(f"\n[temperature=1.0 - 创意]\n{(prompt | hot | parser).invoke({'topic': topic})}")


def main():
    print("=" * 60)
    print("Part 1：统一接口 —— 一套代码调用不同模型")
    print("=" * 60)
    demo_unified_interface()

    print("\n" + "=" * 60)
    print("Part 2：可配置模型（运行时切换）")
    print("=" * 60)
    demo_configurable_model()

    print("\n" + "=" * 60)
    print("Part 3：模型参数对比（temperature）")
    print("=" * 60)
    demo_model_params()


if __name__ == "__main__":
    main()
