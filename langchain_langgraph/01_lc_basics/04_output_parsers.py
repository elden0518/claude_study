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

MODEL = "xiaomi/mimo-v2.5-pro"
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
    import json

    parser = PydanticOutputParser(pydantic_object=BookReview)

    # parser.get_format_instructions() 自动生成格式要求
    format_instr = parser.get_format_instructions()
    print(f"[格式指令预览]\n{format_instr[:200]}...\n")

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "你是一位专业的书评家。你的任务是生成结构化的书评。\n\n"
         "重要：你必须只输出一个有效的JSON对象，不要包含任何其他文本、解释或markdown标记。\n\n"
         "{format_instructions}\n\n"
         "示例输出格式：\n"
         '{{\n'
         '  "title": "书名",\n'
         '  "author": "作者名",\n'
         '  "rating": 8,\n'
         '  "summary": "一句话简介",\n'
         '  "pros": ["优点1", "优点2", "优点3"],\n'
         '  "cons": ["缺点1", "缺点2"]\n'
         '}}'),
        ("human", "请给《{book}》写一篇结构化书评。记住：只输出JSON，不要其他任何内容。"),
    ]).partial(format_instructions=format_instr)

    chain = prompt | llm | StrOutputParser()
    
    try:
        # 先获取原始文本
        raw_text = chain.invoke({"book": "Python编程：从入门到实践"})
        print(f"[原始输出]\n{raw_text}\n")
        
        # 清理可能的markdown代码块标记
        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```"):
            # 移除 ```json 和 ``` 标记
            lines = cleaned_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned_text = '\n'.join(lines).strip()
        
        # 如果还是空的，说明模型没有返回有效内容
        if not cleaned_text:
            print("[警告] 模型未返回任何内容，使用示例数据演示")
            # 创建示例数据来演示Pydantic功能
            sample_data = {
                "title": "Python编程：从入门到实践",
                "author": "Eric Matthes",
                "rating": 9,
                "summary": "一本适合初学者的Python入门经典教材",
                "pros": ["循序渐进的教学方式", "丰富的实践项目", "清晰的代码示例"],
                "cons": ["部分内容较为基础", "高级主题覆盖有限"]
            }
            result = BookReview(**sample_data)
        else:
            # 解析JSON并转换为Pydantic对象
            json_data = json.loads(cleaned_text)
            result = BookReview(**json_data)
        
        print(f"[PydanticOutputParser] 类型: {type(result).__name__}")
        print(f"  书名: {result.title}")
        print(f"  作者: {result.author}")
        print(f"  评分: {result.rating}/10")
        print(f"  简介: {result.summary}")
        print(f"  优点: {result.pros}")
        print(f"  缺点: {result.cons}")
    except json.JSONDecodeError as e:
        print(f"[错误] JSON解析失败: {e}")
        print(f"提示：当前模型可能不支持严格的JSON输出格式")
        print(f"建议使用支持structured output的模型（如Claude 3系列）")
    except Exception as e:
        print(f"[错误] 解析失败: {type(e).__name__}: {e}")
        print(f"提示：某些模型可能不支持严格的JSON输出格式")


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
