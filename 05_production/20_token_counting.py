"""
主题：Token Counting（预估 Tokens）—— 在发送前知道费用

学习目标：
  1. 掌握 client.messages.count_tokens() 的用法
  2. 理解 token 计数与上下文窗口管理的关系
  3. 学会在多轮对话中动态检测是否接近上下文限制
  4. 掌握工具定义对 token 的影响（tool tokens 不可忽视）
  5. 实现自动修剪对话历史，防止上下文溢出

核心价值：
  - 成本预测  : 发送前估算费用，避免意外超支
  - 上下文管理: 精确判断何时需要截断历史
  - 调试工具  : 理解 system prompt / tool 定义消耗了多少 tokens

模型上下文窗口：
  claude-sonnet-4-6 : 200,000 tokens（输入）
  claude-haiku-4-5  : 200,000 tokens
  claude-opus-4     : 200,000 tokens

前置知识：
  - 已完成 01_hello_claude.py 和 09_conversation.py
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
MODEL = "ppio/pa/claude-sonnet-4-6"

# claude-sonnet-4-6 的上下文窗口限制
CONTEXT_WINDOW = 200_000
# 安全阈值：保留 20% 给模型输出，80% 用于输入
MAX_INPUT_TOKENS = int(CONTEXT_WINDOW * 0.8)


# =============================================================================
# Part 1：基础 Token 计数
# =============================================================================

def demo_basic_counting():
    """演示对不同消息组合进行 token 计数。"""

    print("=" * 60)
    print("Part 1：基础 Token 计数")
    print("=" * 60)

    # ── 1a. 简单消息 ────────────────────────────────────────────────────────────
    simple_messages = [
        {"role": "user", "content": "你好，请介绍一下自己。"}
    ]
    resp = client.messages.count_tokens(
        model=MODEL,
        messages=simple_messages
    )
    print(f"\n简单消息：{simple_messages[0]['content']!r}")
    print(f"  → {resp.input_tokens} tokens")

    # ── 1b. 含 system prompt 的消息 ─────────────────────────────────────────────
    system = "你是一位专业的 Python 程序员，只用 Python 解答问题，回答简洁。"
    messages_with_system = [
        {"role": "user", "content": "如何读取 CSV 文件？"}
    ]
    resp2 = client.messages.count_tokens(
        model=MODEL,
        system=system,
        messages=messages_with_system
    )
    resp2_no_system = client.messages.count_tokens(
        model=MODEL,
        messages=messages_with_system
    )
    print(f"\n含 system prompt（{len(system)} 字）：")
    print(f"  有 system : {resp2.input_tokens} tokens")
    print(f"  无 system : {resp2_no_system.input_tokens} tokens")
    print(f"  system 开销：{resp2.input_tokens - resp2_no_system.input_tokens} tokens")

    # ── 1c. 多轮对话 token 增长 ─────────────────────────────────────────────────
    print("\n多轮对话 token 累积：")
    history = []
    for i in range(1, 6):
        history.append({"role": "user", "content": f"这是第 {i} 轮对话的问题，内容稍长。"})
        history.append({"role": "assistant", "content": f"这是第 {i} 轮对话的回答，回答也有一定长度。"})
        count = client.messages.count_tokens(model=MODEL, messages=history)
        print(f"  第 {i} 轮后：{len(history)} 条消息 → {count.input_tokens} tokens")


# =============================================================================
# Part 2：工具定义的 Token 消耗
# =============================================================================

def demo_tool_tokens():
    """演示工具定义对 token 数量的影响（工具描述越详细，token 越多）。"""

    print("\n" + "=" * 60)
    print("Part 2：工具定义的 Token 消耗")
    print("=" * 60)

    base_messages = [{"role": "user", "content": "帮我查一下北京今天的天气。"}]

    # 不带工具
    no_tools = client.messages.count_tokens(model=MODEL, messages=base_messages)

    # 带 1 个简单工具
    simple_tool = [{
        "name": "get_weather",
        "description": "获取天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"}
            },
            "required": ["city"]
        }
    }]
    with_simple = client.messages.count_tokens(
        model=MODEL, messages=base_messages, tools=simple_tool
    )

    # 带 1 个详细描述的工具
    detailed_tool = [{
        "name": "get_weather",
        "description": (
            "获取指定城市的实时天气信息，包括温度、湿度、风速、天气状况描述。"
            "支持全国所有城市和主要国际城市。返回当前时刻的气象数据。"
            "注意：数据更新频率为每小时一次，极端天气时可能有延迟。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如「北京」「上海」「New York」"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位，默认 celsius（摄氏度）"
                }
            },
            "required": ["city"]
        }
    }]
    with_detailed = client.messages.count_tokens(
        model=MODEL, messages=base_messages, tools=detailed_tool
    )

    # 带 5 个工具
    five_tools = simple_tool * 5
    with_five = client.messages.count_tokens(
        model=MODEL, messages=base_messages, tools=five_tools
    )

    print(f"\n  无工具           : {no_tools.input_tokens:5d} tokens")
    print(f"  1 个简单工具     : {with_simple.input_tokens:5d} tokens  (+{with_simple.input_tokens - no_tools.input_tokens})")
    print(f"  1 个详细工具     : {with_detailed.input_tokens:5d} tokens  (+{with_detailed.input_tokens - no_tools.input_tokens})")
    print(f"  5 个简单工具     : {with_five.input_tokens:5d} tokens  (+{with_five.input_tokens - no_tools.input_tokens})")
    print("\n  → 工具描述会显著增加 token 消耗，实际项目中要权衡描述详细度与成本。")


# =============================================================================
# Part 3：上下文窗口管理器 —— 自动修剪历史
# =============================================================================

class ContextAwareConversation:
    """
    带 token 感知的多轮对话管理器。
    在每次发送前检测 token 数，超过阈值时自动修剪旧消息。
    """

    def __init__(self, system: str = "", max_input_tokens: int = MAX_INPUT_TOKENS):
        self.system = system
        self.max_input_tokens = max_input_tokens
        self.messages: list[dict] = []
        self.turn_count = 0

    def _count_tokens(self) -> int:
        """预估当前 messages 的 token 数。"""
        kwargs = {"model": MODEL, "messages": self.messages}
        if self.system:
            kwargs["system"] = self.system
        resp = client.messages.count_tokens(**kwargs)
        return resp.input_tokens

    def _trim_history(self):
        """
        从历史中删除最旧的完整对话轮次（user+assistant 一对），
        直到 token 数低于阈值。
        保留最新的 2 条消息（最近一轮）。
        """
        while len(self.messages) > 2:
            current_tokens = self._count_tokens()
            if current_tokens <= self.max_input_tokens:
                break
            # 删除最旧的一轮（user + assistant = 2 条）
            removed = self.messages[:2]
            self.messages = self.messages[2:]
            print(
                f"  [修剪] 删除第 1 轮消息（{removed[0]['content'][:20]}...）"
                f"，当前 tokens：{self._count_tokens()}"
            )

    def chat(self, user_message: str) -> str:
        """发送一条用户消息，自动管理上下文。"""
        self.turn_count += 1
        self.messages.append({"role": "user", "content": user_message})

        # 发送前检测 tokens
        before_tokens = self._count_tokens()
        print(f"\n── 第 {self.turn_count} 轮（发送前 {before_tokens} tokens）──")

        if before_tokens > self.max_input_tokens:
            print(f"  ⚠ 超过阈值（{self.max_input_tokens}），开始修剪历史...")
            self._trim_history()
            print(f"  修剪后 tokens：{self._count_tokens()}")

        # 正式调用
        resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=self.system,
            messages=self.messages
        )
        answer = resp.content[0].text
        self.messages.append({"role": "assistant", "content": answer})
        return answer


def demo_context_management():
    """演示 ContextAwareConversation 的自动修剪功能。"""

    print("\n" + "=" * 60)
    print("Part 3：带 Token 感知的对话管理器")
    print("=" * 60)

    # 设置较小的阈值（500 tokens）以便快速触发修剪演示
    conv = ContextAwareConversation(
        system="你是一位简洁的助手，每次回答不超过50个汉字。",
        max_input_tokens=500    # 实际生产中应设为 160000
    )

    questions = [
        "Python 的列表和元组有什么区别？",
        "什么是装饰器？",
        "GIL 是什么？有什么影响？",
        "asyncio 和多线程的区别是什么？",
        "怎么用 dataclass？",
    ]

    for q in questions:
        answer = conv.chat(q)
        print(f"  Q: {q}")
        print(f"  A: {answer[:80]}...")


# =============================================================================
# Part 4：成本估算工具
# =============================================================================

def estimate_cost(messages: list, system: str = "", tools: list = None) -> dict:
    """
    估算一次 API 调用的成本（以 claude-sonnet-4-6 定价为例）。

    claude-sonnet-4-6 定价（2025年，每百万 tokens）：
      输入  : $3.00 / MTok
      输出  : $15.00 / MTok
      缓存写: $3.75 / MTok
      缓存读: $0.30 / MTok
    """
    INPUT_PRICE_PER_M = 3.00
    OUTPUT_PRICE_PER_M = 15.00

    kwargs = {"model": MODEL, "messages": messages}
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools

    count_resp = client.messages.count_tokens(**kwargs)
    input_tokens = count_resp.input_tokens

    # 假设输出 500 tokens（实际以 usage.output_tokens 为准）
    estimated_output = 500
    cost = (input_tokens * INPUT_PRICE_PER_M + estimated_output * OUTPUT_PRICE_PER_M) / 1_000_000

    return {
        "input_tokens": input_tokens,
        "estimated_output_tokens": estimated_output,
        "estimated_cost_usd": round(cost, 6),
        "estimated_cost_rmb": round(cost * 7.2, 5),
    }


def demo_cost_estimation():
    """演示成本估算。"""

    print("\n" + "=" * 60)
    print("Part 4：成本估算")
    print("=" * 60)

    scenarios = [
        ("简单问答", [], ""),
        ("含 system prompt", [], "你是一位专业顾问，请提供详细分析。" * 50),
    ]

    for name, tools, sys_p in scenarios:
        msgs = [{"role": "user", "content": "请帮我分析一下当前市场趋势。"}]
        info = estimate_cost(msgs, system=sys_p, tools=tools if tools else None)
        print(f"\n  场景：{name}")
        print(f"    输入 tokens    : {info['input_tokens']}")
        print(f"    预估成本 (USD) : ${info['estimated_cost_usd']}")
        print(f"    预估成本 (CNY) : ¥{info['estimated_cost_rmb']}")


# =============================================================================
# 主入口
# =============================================================================

def main():
    demo_basic_counting()
    demo_tool_tokens()
    demo_context_management()
    demo_cost_estimation()


if __name__ == "__main__":
    main()