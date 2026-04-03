"""
项目：生产级模式 —— 让多 Agent 系统在真实环境中可靠运行

项目目标：
  演示生产环境中必不可少的可靠性模式

涵盖内容：
  1. 错误重试（RetryPolicy）：节点失败后自动重试
  2. 超时与循环控制：防止 Agent 无限循环
  3. 结构化日志：记录每个节点的执行时间和输入输出
  4. 错误恢复节点：专门处理异常情况
  5. LangSmith 追踪：一行配置开启全链路监控（可选）

学习重点：
  - 生产代码与学习代码的差距在哪里
  - 如何让 AI 应用更健壮、更可观测
  - RetryPolicy 的参数含义

前置知识：已完成 24_research_team.py、25_code_review_team.py
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
import time
import logging
from datetime import datetime
from typing import TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import RetryPolicy
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=256)

# ── 结构化日志配置 ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("multi_agent")


# =============================================================================
# Part 1：RetryPolicy —— 节点自动重试
# =============================================================================

class RetryState(TypedDict):
    input: str
    output: str
    attempt_log: list


_attempt_counter = {"count": 0}

def flaky_node(state: RetryState) -> dict:
    """模拟不稳定的节点：前2次失败，第3次成功"""
    _attempt_counter["count"] += 1
    attempt = _attempt_counter["count"]
    logger.info(f"flaky_node 第 {attempt} 次尝试")

    if attempt < 3:
        raise ValueError(f"模拟网络错误（第 {attempt} 次）")

    response = llm.invoke(f"用一句话回答：{state['input']}")
    return {
        "output": response.content,
        "attempt_log": [f"第{attempt}次：成功"],
    }


def demo_retry_policy():
    """演示 RetryPolicy：节点失败后自动重试"""
    print("\n── Part 1：RetryPolicy 自动重试 ──────────────────────────")
    print("  flaky_node 前两次会抛出异常，第三次成功")
    print(f"  RetryPolicy(max_attempts=3, backoff_factor=0.5)")

    _attempt_counter["count"] = 0  # 重置计数器

    graph = StateGraph(RetryState)
    graph.add_node(
        "flaky",
        flaky_node,
        retry=RetryPolicy(
            max_attempts=3,       # 最多尝试3次
            backoff_factor=0.5,   # 每次等待时间：0.5^(attempt-1) 秒
            retry_on=ValueError,  # 只重试 ValueError（可指定异常类型）
        )
    )
    graph.add_edge(START, "flaky")
    graph.add_edge("flaky", END)

    app = graph.compile()
    result = app.invoke({
        "input": "什么是重试机制？",
        "output": "",
        "attempt_log": [],
    })

    print(f"\n  最终成功！输出：{result['output'][:60]}...")


# =============================================================================
# Part 2：结构化日志 + 执行时间追踪
# =============================================================================

class LoggedState(TypedDict):
    messages: Annotated[list, add_messages]
    execution_log: Annotated[list, lambda x, y: x + y]  # 追加日志


def make_logged_node(node_name: str, task_prompt: str):
    """工厂函数：创建带日志记录的节点"""
    def logged_node(state: LoggedState) -> dict:
        start_time = time.time()
        logger.info(f"节点 [{node_name}] 开始执行")

        try:
            response = llm.invoke(task_prompt.format(
                input=state["messages"][-1].content if state["messages"] else ""
            ))
            elapsed = time.time() - start_time

            log_entry = {
                "node": node_name,
                "status": "success",
                "elapsed_ms": round(elapsed * 1000),
                "output_len": len(response.content),
                "timestamp": datetime.now().isoformat(),
            }
            logger.info(f"节点 [{node_name}] 完成，耗时 {log_entry['elapsed_ms']}ms")

            return {
                "messages": [AIMessage(content=response.content, name=node_name)],
                "execution_log": [log_entry],
            }
        except Exception as e:
            elapsed = time.time() - start_time
            log_entry = {
                "node": node_name,
                "status": "error",
                "error": str(e),
                "elapsed_ms": round(elapsed * 1000),
                "timestamp": datetime.now().isoformat(),
            }
            logger.error(f"节点 [{node_name}] 失败：{e}")
            raise

    return logged_node


def demo_structured_logging():
    """演示结构化日志：记录每个节点的执行时间和状态"""
    print("\n── Part 2：结构化日志 ──────────────────────────────────────")

    graph = StateGraph(LoggedState)

    # 用工厂函数创建3个带日志的节点
    graph.add_node("step1", make_logged_node("step1", "用一句话介绍：{input}"))
    graph.add_node("step2", make_logged_node("step2", "把以下内容翻译成英文：{input}"))
    graph.add_node("step3", make_logged_node("step3", "把以下内容改写成正式风格：{input}"))

    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)

    app = graph.compile()
    result = app.invoke({
        "messages": [HumanMessage(content="Python")],
        "execution_log": [],
    })

    print("\n  执行日志汇总：")
    for entry in result["execution_log"]:
        status_icon = "+" if entry["status"] == "success" else "x"
        print(f"  {status_icon} [{entry['node']}] {entry['elapsed_ms']}ms | {entry['status']}")


# =============================================================================
# Part 3：LangSmith 追踪配置
# =============================================================================

def demo_langsmith_setup():
    """演示 LangSmith 追踪的配置方式"""
    print("\n── Part 3：LangSmith 追踪配置 ──────────────────────────────")
    print("""
  LangSmith 是 LangChain 官方的可观测性平台，一行配置开启全链路追踪。

  配置方式（在 .env 中添加）：
  ──────────────────────────────────────────────
  LANGCHAIN_API_KEY=ls__your_key_here
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_PROJECT=my-multi-agent-project
  ──────────────────────────────────────────────

  配置后，每次运行 LangGraph 图，LangSmith 会自动记录：
  - 完整的图执行轨迹（每个节点的输入/输出）
  - 每次 LLM 调用的 token 消耗和耗时
  - 错误信息和重试记录
  - 多 Agent 的调度顺序

  生产价值：
  + 调试复杂的多 Agent 交互
  + 监控 token 消耗，控制成本
  + 发现性能瓶颈（哪个节点最慢）
  + 追踪线上用户的完整对话
    """)

    is_configured = bool(os.getenv("LANGCHAIN_API_KEY"))
    if is_configured:
        print(f"  已配置 LANGCHAIN_API_KEY，当前追踪已开启")
        print(f"  项目名: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    else:
        print(f"  未配置 LANGCHAIN_API_KEY，追踪未开启")
        print(f"  → 访问 https://smith.langchain.com 获取免费 API Key")


def main():
    print("=" * 60)
    print("生产级模式 —— 让多 Agent 系统更可靠")
    print("=" * 60)

    demo_retry_policy()
    demo_structured_logging()
    demo_langsmith_setup()


if __name__ == "__main__":
    main()
