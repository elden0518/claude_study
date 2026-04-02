"""
Tool Use（工具调用）完整演示
============================
本文件演示 Claude API 的 Tool Use 功能，这是构建 AI Agent 的核心机制。

工作流程（Agentic Loop）：
1. 用户发送消息，附带可用工具列表
2. Claude 分析任务，决定是否需要调用工具
3. 若 stop_reason == "tool_use"，解析工具调用请求
4. 本地执行对应的 Python 函数，获取结果
5. 将结果以 tool_result 格式回传给 Claude
6. Claude 根据工具结果继续生成，直到 stop_reason == "end_turn"

本文件展示的工具：
- calculate：计算数学表达式
- get_current_time：获取当前时间（支持时区）
- count_words：统计文本词数和字符数
"""

import sys
import os
import math
import json
from datetime import datetime

# 解决 Windows 中文乱码问题
sys.stdout.reconfigure(encoding='utf-8')

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
MODEL = "ppio/pa/claude-sonnet-4-6"

# ============================================================
# 第一部分：定义工具列表（JSON Schema 格式）
# ============================================================
# tools 参数是一个列表，每个工具包含：
#   - name: 工具的唯一标识符（Claude 用这个名字来调用工具）
#   - description: 详细描述工具的功能和使用场景（非常重要，Claude 靠它判断何时调用）
#   - input_schema: JSON Schema 格式，描述工具接受的参数

TOOLS = [
    {
        "name": "calculate",
        "description": (
            "计算数学表达式，支持加减乘除、幂运算、括号等。"
            "输入合法的 Python 数学表达式字符串，返回计算结果。"
            "例如：'2 ** 10 + 100'、'(3.14 * 5 ** 2)'。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如 '(2**10 + 100) / 3'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_time",
        "description": (
            "获取指定时区的当前时间，返回格式化的日期时间字符串。"
            "默认时区为 Asia/Shanghai（北京时间）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "时区名称，例如 'Asia/Shanghai'、'America/New_York'、'UTC'",
                    "default": "Asia/Shanghai"
                }
            },
            "required": []
        }
    },
    {
        "name": "count_words",
        "description": (
            "统计给定文本的词数（按空格分割）和字符数（不含空格）。"
            "返回包含词数和字符数的统计报告。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "需要统计的文本内容"
                }
            },
            "required": ["text"]
        }
    }
]


# ============================================================
# 第二部分：实现工具对应的 Python 函数
# ============================================================

def calculate(expression: str) -> str:
    """
    计算数学表达式。
    使用 eval() 配合白名单 math 模块，安全执行数学运算。
    """
    try:
        # 只允许访问 math 模块和基本内置函数，防止代码注入
        safe_globals = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow,
        }
        # 将 math 模块中的常用函数加入白名单
        for name in ["sqrt", "log", "log2", "log10", "sin", "cos", "tan",
                     "pi", "e", "ceil", "floor"]:
            safe_globals[name] = getattr(math, name)

        result = eval(expression, safe_globals)
        return f"表达式 `{expression}` 的计算结果为：{result}"
    except Exception as e:
        return f"计算出错：{str(e)}"


def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """
    获取当前时间。
    为保持轻量（不依赖 pytz），使用 datetime.utcnow() 加上固定偏移模拟常见时区。
    """
    # 常见时区与 UTC 偏移（小时）的映射
    tz_offsets = {
        "Asia/Shanghai": 8,
        "Asia/Tokyo": 9,
        "Asia/Seoul": 9,
        "UTC": 0,
        "America/New_York": -4,   # EDT（夏令时），EST 为 -5
        "America/Los_Angeles": -7, # PDT（夏令时），PST 为 -8
        "Europe/London": 1,        # BST（夏令时），GMT 为 0
        "Europe/Paris": 2,         # CEST（夏令时）
    }

    offset_hours = tz_offsets.get(timezone, 8)  # 默认北京时间 UTC+8
    now_utc = datetime.utcnow()

    from datetime import timedelta
    now_local = now_utc + timedelta(hours=offset_hours)

    sign = "+" if offset_hours >= 0 else ""
    return (
        f"当前时间（{timezone}，UTC{sign}{offset_hours}）：\n"
        f"{now_local.strftime('%Y年%m月%d日 %H:%M:%S')}"
    )


def count_words(text: str) -> str:
    """
    统计文本的词数（按空格分割）和字符数（不含空格）。
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text.replace(" ", ""))
    total_chars = len(text)

    return (
        f"文本统计结果：\n"
        f"  - 词数（按空格分割）：{word_count} 词\n"
        f"  - 字符数（不含空格）：{char_count} 字符\n"
        f"  - 总字符数（含空格）：{total_chars} 字符"
    )


# ============================================================
# 第三部分：工具分发函数
# ============================================================

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """
    根据工具名称，将调用分发到对应的 Python 函数。
    返回工具执行结果（字符串）。
    """
    if tool_name == "calculate":
        return calculate(**tool_input)
    elif tool_name == "get_current_time":
        return get_current_time(**tool_input)
    elif tool_name == "count_words":
        return count_words(**tool_input)
    else:
        return f"未知工具：{tool_name}"


# ============================================================
# 第四部分：Agentic Loop（核心）
# ============================================================

def run_agent(user_message: str) -> str:
    """
    运行完整的 Agentic Loop，直到模型返回最终答案。

    循环逻辑：
      1. 调用 API，携带工具定义
      2. 检查 stop_reason：
         - "end_turn"  → 模型完成，提取文本回复，退出循环
         - "tool_use"  → 模型要调用工具，执行工具，追加结果，继续循环
      3. 构造 tool_result 消息并追加到 messages 列表
    """
    print(f"\n{'='*60}")
    print(f"用户：{user_message}")
    print('='*60)

    # 初始化消息历史
    messages = [
        {"role": "user", "content": user_message}
    ]

    # ── Agentic Loop 开始 ──────────────────────────────────
    iteration = 0
    while True:
        iteration += 1
        print(f"\n[轮次 {iteration}] 调用 Claude API...")

        # 步骤 1：向 Claude 发送消息（附带工具定义）
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,          # 告知 Claude 可以使用哪些工具
            messages=messages
        )

        print(f"  stop_reason = {response.stop_reason!r}")

        # ── 情况 A：模型完成，直接返回 ───────────────────────
        if response.stop_reason == "end_turn":
            # 从 response.content 中提取文本类型的 block
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            print(f"\n{'='*60}")
            print("Claude 最终回复：")
            print(final_text)
            print('='*60)
            return final_text

        # ── 情况 B：模型要调用工具 ───────────────────────────
        if response.stop_reason == "tool_use":
            # 步骤 2：将模型的回复（含 tool_use block）追加到消息历史
            # 这一步很关键：必须先把 assistant 的消息（含工具调用）记录下来，
            # 才能在下一条 user 消息中附上 tool_result
            messages.append({
                "role": "assistant",
                "content": response.content  # 包含 TextBlock 和 ToolUseBlock
            })

            # 步骤 3：遍历所有工具调用，逐一执行
            # response.content 可能包含多个 ToolUseBlock（Claude 可以一次调用多个工具）
            tool_results = []

            for block in response.content:
                # 只处理 ToolUseBlock，跳过 TextBlock
                if block.type != "tool_use":
                    continue

                # 解析工具调用信息
                tool_id = block.id       # 工具调用的唯一 ID，回传时必须匹配
                tool_name = block.name   # 工具名称，对应 TOOLS 中的 name 字段
                tool_input = block.input # 工具参数，对应 input_schema 定义的字段

                print(f"  → 调用工具：{tool_name}，参数：{json.dumps(tool_input, ensure_ascii=False)}")

                # 步骤 4：本地执行工具函数
                result = dispatch_tool(tool_name, tool_input)
                print(f"  ← 工具结果：{result}")

                # 步骤 5：构造 tool_result block
                # 格式固定：type="tool_result"，tool_use_id 必须与请求中的 id 一致
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,  # 关键：必须与 ToolUseBlock.id 对应
                    "content": result         # 工具执行结果（字符串）
                })

            # 步骤 6：将所有工具结果作为一条 user 消息追加到历史
            # 注意：tool_result 消息的 role 是 "user"，content 是列表
            # 格式示例：
            # {
            #   "role": "user",
            #   "content": [
            #     {"type": "tool_result", "tool_use_id": "toolu_xxx", "content": "结果文本"},
            #     {"type": "tool_result", "tool_use_id": "toolu_yyy", "content": "结果文本"},
            #   ]
            # }
            messages.append({
                "role": "user",
                "content": tool_results  # content 是列表，可包含多个 tool_result
            })

            # 继续循环，让 Claude 根据工具结果生成下一步回复
            continue

        # ── 情况 C：未预期的 stop_reason ────────────────────
        print(f"  [警告] 未预期的 stop_reason: {response.stop_reason}")
        break

    return ""


# ============================================================
# 第五部分：测试用例
# ============================================================

def main():
    print("Tool Use 演示")
    print("本示例展示 Claude 如何调用本地工具完成复杂任务\n")

    # 测试用例 1：同时触发 calculate 和 get_current_time
    print("\n【测试用例 1】多工具调用")
    run_agent("计算 (2**10 + 100) / 3 的结果，并告诉我现在几点了（北京时间）")

    print("\n" + "─" * 60)

    # 测试用例 2：触发 count_words
    print("\n【测试用例 2】文本统计")
    run_agent("统计这段话的词数：'Python is a great programming language'")


if __name__ == "__main__":
    main()
