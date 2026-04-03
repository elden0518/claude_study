"""
主题：LCEL 链（LangChain Expression Language）—— 用 | 把组件串联

学习目标：
  1. 理解 LCEL 是什么：用 | 操作符把 Runnable 对象串联成流水线
  2. 掌握基础链：prompt | llm | parser
  3. 掌握 RunnableParallel（并行执行多个分支）
  4. 掌握 RunnablePassthrough（透传输入值）
  5. 掌握链的组合（链套链）

核心概念：
  LCEL（LangChain Expression Language）
  每个 Runnable 都有 invoke / stream / batch 方法
  | 操作符等价于 RunnableSequence(a, b, c)

  a | b | c  ≡  RunnableSequence(steps=[a, b, c])
  调用时：输入 → a → b → c → 输出
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=256)


# =============================================================================
# Part 1：基础链 prompt | llm | parser
# =============================================================================

def demo_basic_chain():
    """最经典的三段式链"""
    prompt = ChatPromptTemplate.from_template("用一句话解释：{concept}")
    parser = StrOutputParser()

    # | 把三个 Runnable 串联
    chain = prompt | llm | parser

    result = chain.invoke({"concept": "递归"})
    print(f"[基础链] {result}")

    # 查看链的结构
    print(f"\n链的类型: {type(chain).__name__}")
    print(f"链的步骤数: {len(chain.steps)}")


# =============================================================================
# Part 2：RunnableParallel（并行分支）
# =============================================================================

def demo_parallel():
    """同时运行多个分支，结果合并为字典"""
    # 两个不同的后续处理分支
    pros_chain = (
        ChatPromptTemplate.from_template("列举{topic}的3个优点，每条一句话")
        | llm | StrOutputParser()
    )
    cons_chain = (
        ChatPromptTemplate.from_template("列举{topic}的3个缺点，每条一句话")
        | llm | StrOutputParser()
    )

    # RunnableParallel 同时调用两个链
    parallel = RunnableParallel(
        pros=pros_chain,
        cons=cons_chain,
    )

    result = parallel.invoke({"topic": "Python"})
    print(f"[并行链] Python 分析:")
    print(f"\n优点:\n{result['pros']}")
    print(f"\n缺点:\n{result['cons']}")


# =============================================================================
# Part 3：RunnablePassthrough（透传）
# =============================================================================

def demo_passthrough():
    """把原始输入与处理结果同时传递给下一步"""
    def format_docs(docs):
        return "\n".join(docs)

    # 模拟检索到的文档
    fake_docs = ["Python是解释型语言", "Python支持面向对象", "Python有丰富的库"]

    chain = (
        {
            "context": RunnableLambda(lambda _: format_docs(fake_docs)),
            "question": RunnablePassthrough(),   # 原样透传输入
        }
        | ChatPromptTemplate.from_template(
            "根据以下背景回答问题:\n\n背景:{context}\n\n问题:{question}"
        )
        | llm
        | StrOutputParser()
    )

    result = chain.invoke("Python适合做什么？")
    print(f"[透传链] {result}")


# =============================================================================
# Part 4：链的组合（链套链）
# =============================================================================

def demo_chain_composition():
    """把两个链组合成更长的链"""
    # 第一条链：生成文章大纲
    outline_chain = (
        ChatPromptTemplate.from_template("为主题「{topic}」生成一个3点文章大纲，每点10字内")
        | llm
        | StrOutputParser()
    )

    # 第二条链：根据大纲写摘要
    summary_chain = (
        ChatPromptTemplate.from_template("根据以下大纲写一句话摘要:\n{outline}")
        | llm
        | StrOutputParser()
    )

    # 串联：第一条链的输出作为第二条链的输入
    full_chain = outline_chain | RunnableLambda(lambda x: {"outline": x}) | summary_chain

    result = full_chain.invoke({"topic": "机器学习入门"})
    print(f"[组合链] {result}")


def main():
    print("=" * 60)
    print("Part 1：基础链 prompt | llm | parser")
    print("=" * 60)
    demo_basic_chain()

    print("\n" + "=" * 60)
    print("Part 2：RunnableParallel（并行分支）")
    print("=" * 60)
    demo_parallel()

    print("\n" + "=" * 60)
    print("Part 3：RunnablePassthrough（透传）")
    print("=" * 60)
    demo_passthrough()

    print("\n" + "=" * 60)
    print("Part 4：链的组合（链套链）")
    print("=" * 60)
    demo_chain_composition()


if __name__ == "__main__":
    main()
