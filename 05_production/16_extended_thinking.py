"""
主题：Extended Thinking（扩展思维）—— 让 Claude 深度推理

学习目标：
  1. 理解 Extended Thinking 的工作原理：模型先"想"后"答"
  2. 掌握 thinking 参数的两个字段：type 和 budget_tokens
  3. 学会解析响应中的 thinking block 和 text block
  4. 理解 budget_tokens 对输出质量和费用的影响
  5. 对比普通回答与 thinking 回答的差异

适用场景：
  - 数学/逻辑推理题
  - 多步骤分析（商业决策、代码架构设计）
  - 需要自我检验的任务（写作、翻译质量评估）
  - 复杂的 agentic 任务规划

前置知识：
  - 已完成 01_basics/01_hello_claude.py
  - 了解 response.content 是一个列表（包含多种 block 类型）
"""

# ── 0. Windows 控制台编码修复 ──────────────────────────────────────────────────
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── 1. 导入 ────────────────────────────────────────────────────────────────────
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
# Extended Thinking 需要 claude-3-7-sonnet 及以上；claude-sonnet-4-6 同样支持
MODEL = "ppio/pa/claude-sonnet-4-6"


# =============================================================================
# Part 1：基础用法 —— 开启 Extended Thinking
# =============================================================================

def demo_basic_thinking():
    """最简单的 Extended Thinking 示例：解一道数学推理题。"""

    print("=" * 60)
    print("Part 1：Extended Thinking 基础用法")
    print("=" * 60)

    # thinking 参数说明：
    #   type        : 固定为 "enabled"，开启扩展思维
    #   budget_tokens: 允许模型用于"思考"的最大 token 数
    #                  范围：1024 ~ 32000（部分模型支持更高）
    #                  越高 → 思考更深入，但也更贵
    #                  建议从 5000 开始，复杂任务用 10000+

    QUESTION = """
    一个水池有三个进水管：
    - A 管单独注满需 2 小时
    - B 管单独注满需 3 小时
    - C 管单独注满需 6 小时
    三管同时开，同时还有一个排水管 D，D 单独排空满池需 4 小时。
    问：四管同时工作，几小时能注满水池？
    """

    print(f"\n题目：{QUESTION.strip()}")
    print("\n正在调用 Extended Thinking（budget_tokens=8000）...\n")

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,       # max_tokens 必须 > budget_tokens
        thinking={
            "type": "enabled",
            "budget_tokens": 8000   # 给模型 8000 token 的思考空间
        },
        messages=[
            {"role": "user", "content": QUESTION}
        ]
    )

    # ── 解析响应：content 列表中包含两种 block ──────────────────────────────
    # ThinkingBlock : type="thinking", 包含 .thinking 属性（模型的思考过程）
    # TextBlock     : type="text",     包含 .text 属性（最终答案）

    thinking_text = ""
    answer_text = ""

    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            answer_text = block.text

    # 打印思考过程（前 500 字，完整思考可能很长）
    print("── Claude 的思考过程（前 500 字）─────────────────────────")
    print(thinking_text[:500] + ("..." if len(thinking_text) > 500 else ""))

    print("\n── Claude 的最终答案 ──────────────────────────────────────")
    print(answer_text)

    print(f"\n── Token 用量 ──────────────────────────────────────────────")
    print(f"  输入 tokens    : {response.usage.input_tokens}")
    print(f"  思考 tokens    : {getattr(response.usage, 'cache_read_input_tokens', 0)}")
    print(f"  输出 tokens    : {response.usage.output_tokens}")


# =============================================================================
# Part 2：对比实验 —— 有无 Thinking 的回答差异
# =============================================================================

def compare_with_without_thinking():
    """对比普通回答与 Extended Thinking 回答，看推理深度差异。"""

    print("\n" + "=" * 60)
    print("Part 2：有无 Extended Thinking 的对比")
    print("=" * 60)

    HARD_QUESTION = """
    请分析：一家初创公司应该先做 A（快速迭代、低质量产品先出市场）
    还是先做 B（打磨产品、高质量再出市场）？给出你的决策建议。
    """

    print(f"\n问题：{HARD_QUESTION.strip()}\n")

    # ── 2a. 普通回答 ───────────────────────────────────────────────────────────
    print("── 普通回答（无 thinking）──────────────────────────────────")
    resp_normal = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": HARD_QUESTION}]
    )
    print(resp_normal.content[0].text)

    # ── 2b. Extended Thinking 回答 ─────────────────────────────────────────────
    print("\n── Extended Thinking 回答 ──────────────────────────────────")
    resp_thinking = client.messages.create(
        model=MODEL,
        max_tokens=10000,
        thinking={"type": "enabled", "budget_tokens": 6000},
        messages=[{"role": "user", "content": HARD_QUESTION}]
    )
    for block in resp_thinking.content:
        if block.type == "text":
            print(block.text)


# =============================================================================
# Part 3：在多轮对话中使用 Thinking（thinking block 必须完整回传）
# =============================================================================

def demo_thinking_in_conversation():
    """
    多轮对话中使用 Extended Thinking 的注意事项：
    thinking block 必须原样放回 messages 历史中，否则 API 报错。
    """

    print("\n" + "=" * 60)
    print("Part 3：多轮对话中的 Extended Thinking")
    print("=" * 60)

    messages = []

    # ── 第一轮 ─────────────────────────────────────────────────────────────────
    messages.append({
        "role": "user",
        "content": "请帮我设计一个电商网站的数据库表结构，先列出核心实体。"
    })

    resp1 = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        thinking={"type": "enabled", "budget_tokens": 4000},
        messages=messages
    )

    # 关键：将 assistant 的完整 content（含 thinking block）放入历史
    # 不能只放 text block！否则 API 会报 400 错误。
    messages.append({
        "role": "assistant",
        "content": resp1.content   # ← 直接传入 block 列表，SDK 会自动序列化
    })

    print("第一轮 Claude 回答：")
    for block in resp1.content:
        if block.type == "text":
            print(block.text[:300] + "...")

    # ── 第二轮 ─────────────────────────────────────────────────────────────────
    messages.append({
        "role": "user",
        "content": "好的，请重点展开「订单」相关的表，包括字段和索引设计。"
    })

    resp2 = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        thinking={"type": "enabled", "budget_tokens": 4000},
        messages=messages
    )

    print("\n第二轮 Claude 回答：")
    for block in resp2.content:
        if block.type == "text":
            print(block.text[:400] + "...")

    print("\n✓ 多轮对话中 Extended Thinking 运行成功")


# =============================================================================
# 主入口
# =============================================================================

def main():
    demo_basic_thinking()
    compare_with_without_thinking()
    demo_thinking_in_conversation()


if __name__ == "__main__":
    main()