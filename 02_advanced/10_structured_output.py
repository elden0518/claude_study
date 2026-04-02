"""
结构化输出完整演示
==================
本文件演示如何引导 Claude 输出合法 JSON，并用 Pydantic 进行解析校验。

核心知识点：
1. 引导 Claude 输出 JSON 的 prompt 技巧
   — system prompt 明确要求 JSON 格式
   — assistant pre-fill 以 "{" 强制开头
2. Pydantic BaseModel 定义结构
3. 解析并校验 Claude 的 JSON 输出
4. 嵌套结构和列表字段
5. 解析失败时的重试策略（捕获异常，给 Claude 错误反馈）
"""

import sys
import os
import json

# 解决 Windows 中文乱码问题
sys.stdout.reconfigure(encoding='utf-8')

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import List, Optional

load_dotenv()

client = Anthropic()
MODEL = "ppio/pa/claude-sonnet-4-6"

# ============================================================
# Pydantic 模型定义
# ============================================================

class BookReview(BaseModel):
    """书评结构"""
    title: str
    author: str
    rating: float          # 0-10 分
    pros: List[str]        # 优点列表
    cons: List[str]        # 缺点列表
    summary: str
    recommended: bool


class Product(BaseModel):
    """单个商品"""
    id: int
    name: str
    price: float
    tags: List[str]


class ProductCatalog(BaseModel):
    """商品目录（嵌套结构）"""
    products: List[Product]
    total_count: int
    categories: List[str]


# ============================================================
# 工具函数
# ============================================================

JSON_SYSTEM_PROMPT = (
    "你必须以合法的 JSON 格式回复，不要包含任何解释文字、"
    "代码块标记（```）或多余的换行前缀。"
    "直接输出 JSON 对象本身。"
)


def _strip_code_fence(text: str) -> str:
    """
    去除 Claude 有时返回的 markdown 代码块包裹，如：
    ```json
    { ... }
    ```
    只保留内部的 JSON 文本。
    """
    text = text.strip()
    # 去掉 ```json ... ``` 或 ``` ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        # 去掉第一行（```json 或 ```）
        lines = lines[1:]
        # 去掉最后的 ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def call_claude_json(user_message: str, system_prompt: str = JSON_SYSTEM_PROMPT) -> str:
    """
    调用 Claude 并要求返回 JSON。

    技巧：
    - system prompt 明确要求 JSON 格式，禁止代码块包裹
    - messages 末尾追加 assistant pre-fill `{`，强制 Claude 从 `{` 开始续写
      （注意：API 实际回复是对 pre-fill 的续写，不含 pre-fill 本身，需拼回 "{"）
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": "{"},          # ← pre-fill 强制开头
        ],
    )
    raw = response.content[0].text.strip()

    # 如果模型遵守了 pre-fill，回复是 "{" 之后的续写内容（不含 "{"）
    # 如果模型忽略了 pre-fill 并自行输出完整 JSON / 代码块，则直接清洁使用
    if raw.startswith("{"):
        # 模型自行输出了完整对象（含 pre-fill 效果但已包含 "{"）
        return _strip_code_fence(raw)
    elif raw.startswith("```"):
        # 模型用代码块包裹，剥掉后返回
        return _strip_code_fence(raw)
    else:
        # 标准 pre-fill 续写：模型回复是 "{" 之后的内容，需拼回 "{"
        return "{" + raw


# ============================================================
# Demo 1: 基础 JSON 输出
# ============================================================

def demo_basic_json():
    """基础 JSON 输出 — 用 pre-fill 技巧引导格式"""
    print("\n" + "=" * 60)
    print("Demo 1: 基础 JSON 输出")
    print("=" * 60)

    user_msg = (
        "用 JSON 描述一个名为 '小明' 的用户，包含字段：\n"
        "name（字符串）、age（整数）、hobbies（字符串数组）、"
        "is_student（布尔值）。"
    )

    raw_json = call_claude_json(user_msg)
    print("Claude 原始输出：")
    print(raw_json)

    data = json.loads(raw_json)
    print("\n解析后的 Python dict：")
    print(data)
    print(f"姓名: {data.get('name')}，年龄: {data.get('age')}")


# ============================================================
# Demo 2: Pydantic 模型校验
# ============================================================

def demo_pydantic_validation():
    """Pydantic 模型校验 — 解析 BookReview 结构"""
    print("\n" + "=" * 60)
    print("Demo 2: Pydantic 模型校验（BookReview）")
    print("=" * 60)

    schema_hint = json.dumps(BookReview.model_json_schema(), ensure_ascii=False, indent=2)
    user_msg = (
        "请对《三体》（刘慈欣著）进行书评，严格按如下 JSON Schema 输出：\n"
        f"{schema_hint}"
    )

    raw_json = call_claude_json(user_msg)
    print("Claude 原始输出（前 300 字符）：")
    print(raw_json[:300], "..." if len(raw_json) > 300 else "")

    # 解析 + Pydantic 校验
    data = json.loads(raw_json)
    review = BookReview.model_validate(data)

    print("\nPydantic 校验通过！结构化数据：")
    print(f"  书名      : {review.title}")
    print(f"  作者      : {review.author}")
    print(f"  评分      : {review.rating}/10")
    print(f"  推荐      : {'是' if review.recommended else '否'}")
    print(f"  优点      : {review.pros}")
    print(f"  缺点      : {review.cons}")
    print(f"  总结      : {review.summary[:60]}...")


# ============================================================
# Demo 3: 嵌套结构和列表字段
# ============================================================

def demo_nested_structure():
    """嵌套结构 — ProductCatalog 包含 Product 列表"""
    print("\n" + "=" * 60)
    print("Demo 3: 嵌套结构（ProductCatalog）")
    print("=" * 60)

    schema_hint = json.dumps(ProductCatalog.model_json_schema(), ensure_ascii=False, indent=2)
    user_msg = (
        "生成一个包含 3 个科技类商品的商品目录，严格按如下 JSON Schema 输出：\n"
        f"{schema_hint}\n\n"
        "说明：products 数组中每个商品需包含 id、name、price、tags 字段。"
    )

    raw_json = call_claude_json(user_msg)
    print("Claude 原始输出（前 400 字符）：")
    print(raw_json[:400], "..." if len(raw_json) > 400 else "")

    data = json.loads(raw_json)
    catalog = ProductCatalog.model_validate(data)

    print("\nPydantic 校验通过！商品目录：")
    print(f"  商品总数  : {catalog.total_count}")
    print(f"  分类      : {catalog.categories}")
    for p in catalog.products:
        print(f"  [{p.id}] {p.name} — ¥{p.price}  标签: {p.tags}")


# ============================================================
# Demo 4: 解析失败重试策略
# ============================================================

def demo_retry_on_failure():
    """
    解析失败重试策略。

    策略：
    - 捕获 json.JSONDecodeError 和 pydantic.ValidationError
    - 将错误信息反馈给 Claude，要求重新生成（最多重试 2 次）
    - 演示方式：第一次请求故意让 schema 包含严格约束，触发概率性失败；
      若首次就成功则直接展示成功路径。
    """
    print("\n" + "=" * 60)
    print("Demo 4: 解析失败重试策略")
    print("=" * 60)

    # 故意提出一个有严格类型约束的请求，用来演示校验 + 重试流程
    target_schema = json.dumps(BookReview.model_json_schema(), ensure_ascii=False)

    def attempt(user_msg: str, attempt_num: int) -> Optional[BookReview]:
        """单次尝试：调用 Claude -> parse -> validate"""
        print(f"\n  [尝试 #{attempt_num}] 发送请求...")
        raw_json = call_claude_json(user_msg)
        print(f"  原始输出（前 200 字符）: {raw_json[:200]}")

        # Step 1: JSON 解析
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            print(f"  ✗ JSON 解析失败: {e}")
            raise

        # Step 2: Pydantic 校验
        try:
            return BookReview.model_validate(data)
        except ValidationError as e:
            print(f"  ✗ Pydantic 校验失败: {e.error_count()} 个字段错误")
            raise

    # 构造初始 prompt
    initial_prompt = (
        f"请对《百年孤独》（加西亚·马尔克斯著）进行书评，"
        f"严格按如下 JSON Schema 输出：\n{target_schema}"
    )

    MAX_RETRIES = 2
    last_error: Optional[Exception] = None
    review: Optional[BookReview] = None
    current_prompt = initial_prompt

    for attempt_num in range(1, MAX_RETRIES + 2):  # 1, 2, 3
        try:
            review = attempt(current_prompt, attempt_num)
            print(f"  ✓ 第 {attempt_num} 次尝试成功！")
            break
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            if attempt_num <= MAX_RETRIES:
                # 构造包含错误反馈的重试 prompt
                error_type = "JSON 解析" if isinstance(e, json.JSONDecodeError) else "数据校验"
                current_prompt = (
                    f"上一次回复存在 {error_type} 错误：{str(e)[:200]}\n\n"
                    f"请重新生成，严格按如下 JSON Schema 输出：\n{target_schema}\n"
                    "注意：所有字段类型必须完全匹配，rating 必须是数字（0-10），"
                    "pros/cons 必须是字符串数组，recommended 必须是布尔值。"
                )
                print(f"  → 构造重试 prompt，将进行第 {attempt_num + 1} 次尝试...")
            else:
                print(f"  ✗ 已达最大重试次数（{MAX_RETRIES}），放弃。")

    if review:
        print("\n最终解析结果：")
        print(f"  书名: {review.title}")
        print(f"  评分: {review.rating}/10")
        print(f"  推荐: {'是' if review.recommended else '否'}")
        print(f"  总结: {review.summary[:60]}...")
    else:
        print(f"\n所有尝试均失败，最后错误: {last_error}")


# ============================================================
# 入口
# ============================================================

def main():
    print("结构化输出演示")
    print("模型:", MODEL)

    demo_basic_json()
    demo_pydantic_validation()
    demo_nested_structure()
    demo_retry_on_failure()

    print("\n" + "=" * 60)
    print("全部演示完成！")


if __name__ == "__main__":
    main()
