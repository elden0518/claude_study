"""
主题：Output Parsers —— 把模型输出变成结构化数据

学习目标：
  1. 理解为什么需要 Output Parser（模型输出是字符串，应用需要结构化数据）
  2. 掌握 StrOutputParser（字符串，最简单）
  3. 掌握 JsonOutputParser（解析 JSON 输出）
  4. 掌握 PydanticOutputParser（强类型结构化输出，含格式指令）
  5. 理解格式指令（format_instructions）的作用

核心概念：
  模型只能输出文本 → Output Parser 负责把文本转换成 Python 对象
  PydanticOutputParser 会自动生成格式指令注入 prompt，
  告诉模型"我需要你输出这种格式的 JSON"
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
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from typing import List
from pydantic import BaseModel, Field

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=512)


# =============================================================================
# Part 1：StrOutputParser（字符串）
# =============================================================================

def demo_str_parser():
    """最简单的 parser：把 AIMessage 变成纯字符串"""
    chain = (
        ChatPromptTemplate.from_template("用一句话解释{concept}")
        | llm
        | StrOutputParser()
    )
    result = chain.invoke({"concept": "封装"})
    print(f"[StrOutputParser] 类型: {type(result).__name__}")
    print(f"  内容: {result}")


# =============================================================================
# Part 2：JsonOutputParser
# =============================================================================

def demo_json_parser():
    """解析 JSON 格式的输出，返回 Python dict/list"""
    parser = JsonOutputParser()

    chain = (
        ChatPromptTemplate.from_template(
            "返回一个 JSON 对象，包含以下字段：\n"
            "name（语言名）, year（发明年份）, paradigm（编程范式列表）\n"
            "介绍编程语言：{language}\n"
            "只输出 JSON，不要其他内容。"
        )
        | llm
        | parser
    )

    result = chain.invoke({"language": "Python"})
    print(f"[JsonOutputParser] 类型: {type(result).__name__}")
    print(f"  name: {result.get('name')}")
    print(f"  year: {result.get('year')}")
    print(f"  paradigm: {result.get('paradigm')}")


# =============================================================================
# Part 3：PydanticOutputParser（强类型）
# =============================================================================

class BookReview(BaseModel):
    """书评结构"""
    title: str = Field(description="书名")
    author: str = Field(description="作者")
    rating: int = Field(description="评分，1-10分")
    summary: str = Field(description="一句话简介")
    pros: List[str] = Field(description="优点列表，3条")
    cons: List[str] = Field(description="缺点列表，2条")


def demo_pydantic_parser():
    """PydanticOutputParser：输出强类型的 Pydantic 对象"""
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=BookReview)

    # parser.get_format_instructions() 自动生成格式要求
    print(f"[格式指令预览]\n{parser.get_format_instructions()[:200]}...\n")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位书评家。\n\n{format_instructions}"),
        ("human", "请给《{book}》写一篇结构化书评"),
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    result = chain.invoke({"book": "Python编程：从入门到实践"})

    print(f"[PydanticOutputParser] 类型: {type(result).__name__}")
    print(f"  书名: {result.title}")
    print(f"  作者: {result.author}")
    print(f"  评分: {result.rating}/10")
    print(f"  简介: {result.summary}")
    print(f"  优点: {result.pros}")
    print(f"  缺点: {result.cons}")


def main():
    print("=" * 60)
    print("Part 1：StrOutputParser")
    print("=" * 60)
    demo_str_parser()

    print("\n" + "=" * 60)
    print("Part 2：JsonOutputParser")
    print("=" * 60)
    demo_json_parser()

    print("\n" + "=" * 60)
    print("Part 3：PydanticOutputParser（强类型）")
    print("=" * 60)
    demo_pydantic_parser()


if __name__ == "__main__":
    main()
