import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
主题：Callbacks —— 追踪和监控 LangChain 的每一步执行

学习目标：
  1. 理解 Callback 机制：在链执行的关键节点插入自定义逻辑
  2. 掌握 StdOutCallbackHandler（打印所有执行步骤）
  3. 学会自定义 BaseCallbackHandler（记录 token 消耗）
  4. 理解 Callback 的应用场景（监控、日志、计费）
  5. 了解 LangSmith 追踪（生产级监控工具）

核心概念：
  Callback = 在链执行时触发的钩子函数

  关键事件：
  on_chain_start / on_chain_end   → 链开始/结束
  on_llm_start / on_llm_end       → LLM 调用开始/结束
  on_tool_start / on_tool_end     → 工具调用开始/结束

  用途：日志记录、性能监控、token 计费、错误追踪

前置知识：已完成 03_lcel_chains.py
"""

from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler, StdOutCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ============================================================
# 自定义 TokenCounterCallback（全局定义，供多个 Part 使用）
# ============================================================

class TokenCounterCallback(BaseCallbackHandler):
    """
    自定义 Callback Handler：统计 LLM 调用的 token 消耗。

    继承 BaseCallbackHandler 并重写感兴趣的事件方法。
    未重写的方法保持默认（空操作），不会产生任何输出。

    on_llm_end 在每次 LLM 调用结束时触发，
    response 参数包含完整的响应信息，包括 token 使用量。
    """

    def __init__(self):
        super().__init__()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        self.call_log: List[Dict[str, Any]] = []

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """LLM 调用开始时触发。"""
        self.call_count += 1
        print(f"  [TokenCounter] LLM 调用 #{self.call_count} 开始")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        LLM 调用结束时触发。

        LLMResult 的结构：
          response.llm_output        → 包含 token 使用量的字典
          response.generations       → 生成结果列表（可能有多个候选）

        Anthropic 的 llm_output 通常包含：
          {"usage": {"input_tokens": N, "output_tokens": M}}
        """
        # 尝试从 llm_output 获取 token 使用信息
        usage = {}
        if response.llm_output:
            # 不同模型提供商的字段名可能不同
            usage = (
                response.llm_output.get("usage", {})
                or response.llm_output.get("token_usage", {})
            )

        # 如果 llm_output 没有，尝试从 generation_info 获取
        if not usage and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "generation_info") and gen.generation_info:
                        usage = gen.generation_info.get("usage", {})
                        if usage:
                            break

        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        call_info = {
            "call": self.call_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        self.call_log.append(call_info)

        print(
            f"  [TokenCounter] LLM 调用 #{self.call_count} 结束 "
            f"| 输入 tokens: {input_tokens} "
            f"| 输出 tokens: {output_tokens}"
        )

    def print_summary(self):
        """打印 token 使用汇总。"""
        print(f"  ┌─ Token 使用汇总 ──────────────────────")
        print(f"  │  总调用次数：{self.call_count}")
        print(f"  │  总输入 tokens：{self.total_input_tokens}")
        print(f"  │  总输出 tokens：{self.total_output_tokens}")
        print(f"  │  合计 tokens：{self.total_input_tokens + self.total_output_tokens}")
        print(f"  └───────────────────────────────────────")


# ============================================================
# Part 1: StdOutCallbackHandler —— 打印执行追踪
# ============================================================

def part1_stdout_callback():
    """
    StdOutCallbackHandler 是 LangChain 内置的追踪处理器。

    它实现了所有 on_* 方法，把每个执行事件打印到标准输出。
    非常适合开发和调试阶段，帮助理解链的执行流程。

    Callback 传递方式：
      在 chain.invoke() 的 config 参数中传入
      config={"callbacks": [handler]}
    """
    print("=" * 60)
    print("Part 1: StdOutCallbackHandler —— 观察执行追踪")
    print("=" * 60)

    llm = ChatAnthropic(
        model="ppio/pa/claude-sonnet-4-6",
        max_tokens=100,
    )

    prompt = ChatPromptTemplate.from_template(
        "用一句话解释：{topic}"
    )

    chain = prompt | llm | StrOutputParser()

    # 创建 StdOutCallbackHandler
    stdout_handler = StdOutCallbackHandler()

    print("运行链（使用 StdOutCallbackHandler 追踪）：")
    print("问题：用一句话解释 LangChain")
    print()
    print("【执行追踪输出】")
    print("-" * 40)

    # 通过 config 参数传入 callback
    result = chain.invoke(
        {"topic": "LangChain"},
        config={"callbacks": [stdout_handler]},
    )

    print("-" * 40)
    print()
    print(f"【最终结果】{result}")
    print()


# ============================================================
# Part 2: 自定义 TokenCounterCallback
# ============================================================

def part2_custom_token_counter():
    """
    自定义 BaseCallbackHandler 是 LangChain Callback 系统的核心能力。

    通过继承 BaseCallbackHandler 并重写特定的 on_* 方法，
    可以在不修改链代码的前提下插入监控、日志、计费等逻辑。

    这种设计模式称为"观察者模式"（Observer Pattern），
    将业务逻辑（链）与横切关注点（监控）解耦。
    """
    print("=" * 60)
    print("Part 2: 自定义 TokenCounterCallback —— Token 计费")
    print("=" * 60)

    llm = ChatAnthropic(
        model="ppio/pa/claude-sonnet-4-6",
        max_tokens=100,
    )

    prompt = ChatPromptTemplate.from_template(
        "请用中文简短解释：{concept}"
    )
    chain = prompt | llm | StrOutputParser()

    # 创建自定义 callback
    token_counter = TokenCounterCallback()

    # 进行两次调用，观察 token 累计
    questions = ["什么是 RAG", "什么是 Agent"]

    print("进行 2 次 LLM 调用，观察 token 累积计数：")
    print()

    for q in questions:
        print(f"问题：{q}")
        result = chain.invoke(
            {"concept": q},
            config={"callbacks": [token_counter]},
        )
        print(f"回答：{result[:80]}{'...' if len(result) > 80 else ''}")
        print()

    print("【Token 使用汇总】")
    token_counter.print_summary()
    print()


# ============================================================
# Part 3: Callback 作用域 —— 链级 vs 调用级
# ============================================================

def part3_callback_scoping():
    """
    Callback 可以在两个不同的级别传入，它们的作用域不同：

    (a) 调用级（invoke level）：
        chain.invoke(input, config={"callbacks": [handler]})
        只在这一次调用中生效。
        适合：按需追踪某次特定调用。

    (b) 链级（chain level）：
        chain.with_config({"callbacks": [handler]})
        返回一个新链，该链的每次调用都会使用这些 callbacks。
        适合：始终监控某个链的所有调用（如生产环境全局日志）。

    注意：链级 callback 和调用级 callback 可以同时存在，两者都会触发。
    """
    print("=" * 60)
    print("Part 3: Callback 作用域 —— 链级 vs 调用级")
    print("=" * 60)

    llm = ChatAnthropic(
        model="ppio/pa/claude-sonnet-4-6",
        max_tokens=80,
    )

    prompt = ChatPromptTemplate.from_template("用10个字描述：{subject}")
    base_chain = prompt | llm | StrOutputParser()

    # --- (a) 调用级 Callback ---
    print("(a) 调用级 Callback：config 参数传入，仅本次调用生效")
    print()

    invoke_handler = TokenCounterCallback()

    result1 = base_chain.invoke(
        {"subject": "Python"},
        config={"callbacks": [invoke_handler]},
    )
    print(f"  结果：{result1}")
    print()

    # 再次调用，不传 callback，handler 不会被触发
    result2 = base_chain.invoke({"subject": "JavaScript"})
    print(f"  第二次调用（不传 callback）：{result2}")
    print(f"  invoke_handler.call_count = {invoke_handler.call_count}（仅记录了 1 次）")
    print()

    # --- (b) 链级 Callback ---
    print("(b) 链级 Callback：with_config() 绑定，每次调用都生效")
    print()

    chain_handler = TokenCounterCallback()

    # with_config 返回一个新链，callback 永久绑定在这个链上
    monitored_chain = base_chain.with_config({"callbacks": [chain_handler]})

    monitored_chain.invoke({"subject": "LangChain"})
    monitored_chain.invoke({"subject": "LangGraph"})

    print(f"  chain_handler.call_count = {chain_handler.call_count}（记录了 2 次）")
    print()
    print("  【链级 Token 汇总】")
    chain_handler.print_summary()
    print()

    print("【结论】")
    print("  • 调用级：灵活，适合临时调试某次请求")
    print("  • 链级：稳定，适合生产环境持续监控某个链")
    print()


# ============================================================
# main
# ============================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       12_callbacks.py —— Callbacks 追踪监控演示           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    part1_stdout_callback()
    part2_custom_token_counter()
    part3_callback_scoping()

    print("=" * 60)
    print("Callbacks 演示完毕！")
    print()
    print("关键要点：")
    print("  • Callback 是在链执行的关键节点插入自定义逻辑的钩子")
    print("  • StdOutCallbackHandler 内置，打印所有执行步骤")
    print("  • 自定义 Handler 继承 BaseCallbackHandler，重写 on_* 方法")
    print("  • 调用级（invoke config）vs 链级（with_config）两种传入方式")
    print("  • 生产环境推荐使用 LangSmith 进行全链路追踪")
    print("=" * 60)


if __name__ == "__main__":
    main()
