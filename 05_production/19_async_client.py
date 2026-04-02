"""
主题：Async Client（异步客户端）—— 并发请求，大幅提升吞吐量

学习目标：
  1. 掌握 AsyncAnthropic 的初始化和基本用法
  2. 使用 asyncio.gather() 并发发送多个请求
  3. 理解并发 vs 串行的性能差异（实测对比）
  4. 掌握异步流式输出（async streaming）
  5. 理解并发限制和速率控制（Semaphore）

核心概念：
  同步客户端（Anthropic）   : 每次调用阻塞等待，串行执行
  异步客户端（AsyncAnthropic）: 发出请求后不阻塞，可同时等待多个响应

适用场景：
  - Web 服务（FastAPI / aiohttp 中调用 Claude）
  - 批量处理但需要实时返回（不能用 Batch API 的场景）
  - 并发测试多个 Prompt 变体
  - 高吞吐量 pipeline（数据处理、内容生成）

前置知识：
  - 已完成 01_hello_claude.py
  - 了解 Python async/await 基础语法
"""

# ── 0. Windows 控制台编码修复 ──────────────────────────────────────────────────
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── 1. 导入 ────────────────────────────────────────────────────────────────────
import os
import time
import asyncio
from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "ppio/pa/claude-sonnet-4-6"

# 同步客户端（用于串行对比实验）
sync_client = Anthropic()

# 异步客户端（用于并发实验）
async_client = AsyncAnthropic()


# =============================================================================
# Part 1：串行 vs 并发 性能对比
# =============================================================================

# 5 个独立问题，互不依赖，非常适合并发
QUESTIONS = [
    "用一句话解释什么是 TCP/IP 协议。",
    "用一句话解释什么是 RESTful API。",
    "用一句话解释什么是 Docker 容器。",
    "用一句话解释什么是 Git 版本控制。",
    "用一句话解释什么是机器学习。",
]


def serial_requests():
    """串行发送 5 个请求（传统同步方式）。"""
    results = []
    for q in QUESTIONS:
        resp = sync_client.messages.create(
            model=MODEL, max_tokens=100,
            messages=[{"role": "user", "content": q}]
        )
        results.append(resp.content[0].text)
    return results


async def concurrent_requests():
    """并发发送 5 个请求（asyncio.gather）。"""

    async def single_request(question: str) -> str:
        resp = await async_client.messages.create(
            model=MODEL, max_tokens=100,
            messages=[{"role": "user", "content": question}]
        )
        return resp.content[0].text

    # asyncio.gather 同时发起所有请求，等待所有完成
    # 耗时 ≈ max(单个请求时间)，而非 sum(所有请求时间)
    results = await asyncio.gather(*[single_request(q) for q in QUESTIONS])
    return results


def demo_performance_comparison():
    """实测串行 vs 并发的时间差异。"""

    print("=" * 60)
    print("Part 1：串行 vs 并发 性能对比")
    print("=" * 60)

    # 串行
    print("\n── 串行请求（5 次，顺序执行）──────────────────────────────")
    t0 = time.time()
    serial_results = serial_requests()
    serial_time = time.time() - t0
    print(f"  ✓ 完成，耗时：{serial_time:.2f}s")

    # 并发
    print("\n── 并发请求（5 次，同时发出）──────────────────────────────")
    t0 = time.time()
    concurrent_results = asyncio.run(concurrent_requests())
    concurrent_time = time.time() - t0
    print(f"  ✓ 完成，耗时：{concurrent_time:.2f}s")

    print(f"\n── 结果对比 ────────────────────────────────────────────────")
    print(f"  串行耗时  : {serial_time:.2f}s")
    print(f"  并发耗时  : {concurrent_time:.2f}s")
    print(f"  速度提升  : {serial_time / concurrent_time:.1f}x")

    print("\n── 并发回答 ─────────────────────────────────────────────────")
    for q, ans in zip(QUESTIONS, concurrent_results):
        print(f"  Q: {q}")
        print(f"  A: {ans[:80]}...\n")


# =============================================================================
# Part 2：并发限速（Semaphore）
# =============================================================================

async def rate_limited_requests(questions: list, max_concurrent: int = 3):
    """
    使用 asyncio.Semaphore 限制最大并发数。

    为什么需要限速：
    - API 有速率限制（RPM / TPM）
    - 过多并发请求可能触发 429 错误
    - 实际生产中需要根据 API tier 调整并发数
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def request_with_limit(question: str) -> tuple[str, str]:
        async with semaphore:   # 确保同时最多 max_concurrent 个请求在运行
            resp = await async_client.messages.create(
                model=MODEL, max_tokens=100,
                messages=[{"role": "user", "content": question}]
            )
            return question, resp.content[0].text

    tasks = [request_with_limit(q) for q in questions]
    return await asyncio.gather(*tasks)


def demo_rate_limiting():
    """演示限速并发，避免触发 API 速率限制。"""

    print("=" * 60)
    print("Part 2：并发限速（Semaphore）")
    print("=" * 60)

    # 10 个问题，最多同时 3 个并发
    more_questions = QUESTIONS * 2   # 10 个
    print(f"  {len(more_questions)} 个问题，最大并发：3\n")

    t0 = time.time()
    results = asyncio.run(rate_limited_requests(more_questions, max_concurrent=3))
    elapsed = time.time() - t0

    print(f"  ✓ 全部完成，耗时：{elapsed:.2f}s（{len(results)} 条结果）")
    for q, a in results[:3]:   # 只打印前 3 条
        print(f"\n  Q: {q}")
        print(f"  A: {a[:80]}")


# =============================================================================
# Part 3：异步流式输出
# =============================================================================

async def async_streaming():
    """异步流式：在 async 函数中使用流式输出。"""

    print("\n" + "=" * 60)
    print("Part 3：异步流式输出")
    print("=" * 60)

    print("\n实时输出（流式）：\n")
    async with async_client.messages.stream(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": "请用 3 点介绍异步编程的优势。"}]
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)

    final = await stream.get_final_message()
    print(f"\n\n  输出 tokens：{final.usage.output_tokens}")


# =============================================================================
# Part 4：在 FastAPI 中使用异步客户端（代码示例，不运行）
# =============================================================================

def show_fastapi_example():
    """展示在 FastAPI web 服务中集成 AsyncAnthropic 的典型模式。"""

    print("\n" + "=" * 60)
    print("Part 4：FastAPI 集成示例（展示代码，不运行）")
    print("=" * 60)

    code = '''
from fastapi import FastAPI
from anthropic import AsyncAnthropic
from pydantic import BaseModel

app = FastAPI()
# AsyncAnthropic 客户端应在应用级别创建（复用连接池）
client = AsyncAnthropic()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    """FastAPI 路由中直接 await AsyncAnthropic，不会阻塞事件循环。"""
    response = await client.messages.create(
        model="ppio/pa/claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": request.message}]
    )
    return {"reply": response.content[0].text}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式响应版本（需配合 StreamingResponse）。"""
    from fastapi.responses import StreamingResponse

    async def generate():
        async with client.messages.stream(
            model="ppio/pa/claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": request.message}]
        ) as stream:
            async for text in stream.text_stream:
                yield text

    return StreamingResponse(generate(), media_type="text/plain")
'''
    print(code)


# =============================================================================
# 主入口
# =============================================================================

def main():
    demo_performance_comparison()
    demo_rate_limiting()
    asyncio.run(async_streaming())
    show_fastapi_example()


if __name__ == "__main__":
    main()