"""
主题：Batch API（批量处理）—— 高效处理大量请求，费用降低 50%

学习目标：
  1. 理解 Batch API 的工作模式：异步提交 → 轮询状态 → 获取结果
  2. 掌握 MessageBatchRequestParam 的构造（custom_id + params）
  3. 学会提交批量任务、查询状态、获取结果
  4. 理解适用场景和限制（处理时间 ≤ 24h，无法流式）
  5. 对比批量 API 与普通 API 的成本差异

核心优势：
  - 价格：相比实时 API 节省 50% 费用
  - 容量：不占用实时 API 的速率限制
  - 规模：单批次最多 10,000 条请求

适用场景：
  - 数据集标注（批量分类、打标签）
  - 内容审核（批量分析文本）
  - 批量翻译 / 摘要生成
  - 报告生成（每日数百份报告）
  - 离线评估（LLM-as-a-judge 批量评分）

限制：
  - 最大处理时间：24 小时（通常远快于此）
  - 不支持流式输出
  - 不支持 Extended Thinking（Batch API 不支持 thinking 参数）

前置知识：
  - 已完成 01_hello_claude.py（基础调用）
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
import json
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request as MessageBatchRequestParam
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
MODEL = "ppio/pa/claude-sonnet-4-6"


# =============================================================================
# Part 1：提交批量任务
# =============================================================================

def create_batch():
    """
    构造批量任务并提交。
    每条请求包含：
      custom_id: 你自定义的唯一标识（用于在结果中匹配对应请求）
      params    : 与普通 messages.create() 完全相同的参数
    """

    print("=" * 60)
    print("Part 1：提交批量任务")
    print("=" * 60)

    # 示例：批量为 5 个产品生成一句话简介
    products = [
        ("prod_001", "iPhone 16 Pro Max", "旗舰智能手机"),
        ("prod_002", "MacBook Air M3", "轻薄笔记本电脑"),
        ("prod_003", "AirPods Pro 2", "主动降噪耳机"),
        ("prod_004", "Apple Watch Ultra 2", "户外运动智能手表"),
        ("prod_005", "iPad Pro M4", "专业平板电脑"),
    ]

    # 构造批量请求列表
    requests = []
    for custom_id, product_name, category in products:
        requests.append(
            MessageBatchRequestParam(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=MODEL,
                    max_tokens=128,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"请用一句话（不超过30个汉字）描述产品「{product_name}」"
                                f"（{category}）的核心卖点。"
                                f"只输出一句话，不要加任何前缀。"
                            )
                        }
                    ]
                )
            )
        )

    print(f"  准备提交 {len(requests)} 条请求...\n")
    for _, name, _ in products:
        print(f"  · {name}")

    # 提交批量任务
    batch = client.messages.batches.create(requests=requests)

    print(f"\n  ✓ 批量任务已提交")
    print(f"  批次 ID       : {batch.id}")
    print(f"  状态          : {batch.processing_status}")
    print(f"  请求数        : {batch.request_counts.processing}")
    print(f"  预计最长等待  : 24 小时（通常几分钟内完成）")

    return batch.id


# =============================================================================
# Part 2：查询批次状态
# =============================================================================

def poll_batch_status(batch_id: str, max_wait_seconds: int = 300):
    """
    轮询批次状态，直到处理完成或超时。

    状态说明：
      in_progress  : 正在处理中
      ended        : 已完成（所有请求都有了结果，不一定全部成功）
      canceling    : 正在取消
      canceled     : 已取消
    """

    print("\n" + "=" * 60)
    print("Part 2：轮询批次状态")
    print("=" * 60)
    print(f"  批次 ID：{batch_id}")
    print(f"  每 10 秒查询一次，最多等待 {max_wait_seconds} 秒...\n")

    start = time.time()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts

        elapsed = int(time.time() - start)
        print(
            f"  [{elapsed:3d}s] 状态：{batch.processing_status:12s} | "
            f"处理中：{counts.processing} | "
            f"成功：{counts.succeeded} | "
            f"失败：{counts.errored}"
        )

        if batch.processing_status == "ended":
            print("\n  ✓ 批次处理完成！")
            return batch

        if elapsed >= max_wait_seconds:
            print(f"\n  ✗ 超过等待上限（{max_wait_seconds}s），退出轮询。")
            print(f"  可以稍后用 retrieve('{batch_id}') 继续查询。")
            return None

        time.sleep(10)


# =============================================================================
# Part 3：获取并解析批次结果
# =============================================================================

def fetch_results(batch_id: str):
    """
    获取批次的所有结果，按 custom_id 解析。

    每条结果的 result.type：
      succeeded  : 成功，result.message 包含完整 Message 对象
      errored    : 失败，result.error 包含错误信息
      canceled   : 被取消
      expired    : 超过 24h 未处理
    """

    print("\n" + "=" * 60)
    print("Part 3：解析批次结果")
    print("=" * 60)

    results = {}

    # client.messages.batches.results() 返回一个可迭代对象（流式）
    for item in client.messages.batches.results(batch_id):
        custom_id = item.custom_id
        result = item.result

        if result.type == "succeeded":
            text = result.message.content[0].text
            tokens_out = result.message.usage.output_tokens
            results[custom_id] = {"status": "ok", "text": text, "tokens": tokens_out}
            print(f"\n  [{custom_id}] ✓ ({tokens_out} tokens)")
            print(f"  {text}")

        elif result.type == "errored":
            error_msg = str(result.error)
            results[custom_id] = {"status": "error", "error": error_msg}
            print(f"\n  [{custom_id}] ✗ 错误：{error_msg}")

        else:
            results[custom_id] = {"status": result.type}
            print(f"\n  [{custom_id}] {result.type}")

    print(f"\n── 汇总：成功 {sum(1 for v in results.values() if v['status']=='ok')} / {len(results)} 条")
    return results


# =============================================================================
# Part 4：取消批次（演示用，不实际调用）
# =============================================================================

def demo_cancel(batch_id: str):
    """演示如何取消正在处理中的批次。"""

    print("\n" + "=" * 60)
    print("Part 4：取消批次（仅演示，不实际执行）")
    print("=" * 60)
    print("  如果需要取消正在处理的批次，调用：")
    print(f"  client.messages.batches.cancel('{batch_id}')")
    print()
    print("  注意：已完成的请求结果不会因取消而丢失。")
    print("  适用场景：发现数据错误、成本控制、紧急中止。")


# =============================================================================
# Part 5：列出所有批次（历史管理）
# =============================================================================

def list_recent_batches():
    """列出最近的批次，用于管理历史任务。"""

    print("\n" + "=" * 60)
    print("Part 5：列出近期批次")
    print("=" * 60)

    batches_page = client.messages.batches.list(limit=5)
    batches = list(batches_page)

    if not batches:
        print("  暂无历史批次")
        return

    print(f"  最近 {len(batches)} 个批次：\n")
    for b in batches:
        counts = b.request_counts
        print(
            f"  {b.id[:20]}... | {b.processing_status:12s} | "
            f"成功:{counts.succeeded} 失败:{counts.errored} | "
            f"创建：{b.created_at}"
        )


# =============================================================================
# 主入口
# =============================================================================

def main():
    # 1. 提交批量任务
    batch_id = create_batch()

    # 2. 轮询直到完成
    batch = poll_batch_status(batch_id, max_wait_seconds=300)

    # 3. 获取结果
    if batch and batch.processing_status == "ended":
        fetch_results(batch_id)
    else:
        print(f"\n任务仍在进行中，稍后可以运行：")
        print(f"  fetch_results('{batch_id}')")

    # 4. 演示取消
    demo_cancel(batch_id)

    # 5. 列出历史批次
    list_recent_batches()


if __name__ == "__main__":
    main()