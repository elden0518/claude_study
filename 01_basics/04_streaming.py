"""
主题：流式输出（Streaming）—— 让 Claude 边想边说

学习目标：
  1. 掌握 with client.messages.stream() as stream 的 context manager 用法
  2. 学会用 stream.text_stream 逐 token 实时打印文本（最简方式）
  3. 了解 stream.get_final_message() 在流结束后获取完整消息对象（含 usage）
  4. 学会回调风格写法：继承 MessageStream（EventHandler 子类）重写 on_text()
  5. 理解流式 vs 非流式的适用场景，以及为什么流式对用户体验更友好

前置知识：
  - 已完成 01_hello_claude.py（基础调用）
  - 了解 Python 的 with 语句（context manager）和 for 循环迭代器

课程顺序：这是 01_basics 模块的第四个文件，建议按序学习。

核心概念速记：
  非流式（sync）：等模型全部生成完毕 → 一次性返回完整结果
  流式（stream）：模型生成一个 token 就立即推送 → 用户看到"打字机"效果
"""

# ── 0. Windows 控制台编码修复 ──────────────────────────────────────────────────
# Windows 默认控制台编码（GBK）无法正常显示 Unicode 字符，强制设为 UTF-8
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── 1. 导入依赖 ────────────────────────────────────────────────────────────────
import os
import time          # 用于流式 vs 非流式的耗时对比演示

import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# 统一使用的模型（通过代理前缀 ppio/pa/ 路由到 claude-sonnet-4-6）
MODEL = "ppio/pa/claude-sonnet-4-6"


def section(title: str):
    """打印分隔标题，方便阅读输出。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Demo 1 : 最简单的流式输出 —— 逐 token 打印
# ─────────────────────────────────────────────────────────────────────────────

def demo_basic_streaming():
    """
    最简单的流式输出：逐 token 打印

    关键 API：with client.messages.stream() as stream
    ─────────────────────────────────────────────────
    与 messages.create() 的区别：
      - messages.create()  → 阻塞等待全部生成完毕，返回完整 Message 对象
      - messages.stream()  → 立即返回 MessageStreamManager，边生成边推送 token

    为什么流式对用户体验很重要？
      想象用户提问后，需要等待 10 秒才看到 Claude 的回答（非流式）。
      而流式模式下，用户 0.5 秒内就能看到第一个字出现，
      即使总耗时相同，"感知延迟"也大幅降低 —— 这是现代 AI 产品的标配体验。

    stream.text_stream：
      - 这是最简单的迭代方式，直接 yield 纯文本片段（str）
      - 每次 yield 的内容可能是 1~多个 token 合并后的字符串
      - 适合只关心最终文字、不需要处理事件细节的场景
    """
    section("Demo 1 : 最简单的流式输出 —— 逐 token 打印")

    prompt = "请用 3 句话介绍一下量子计算的基本原理，语言尽量通俗易懂。"
    print(f"\n提问：{prompt}")
    print("\nClaude 的回答（流式输出）：")

    # with 语句确保流式连接在退出代码块时被正确关闭（释放 HTTP 连接）
    # 参数与 messages.create() 完全一致：model / max_tokens / messages
    with client.messages.stream(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        # stream.text_stream 是一个生成器（generator），每次 yield 一个文本片段
        # flush=True 非常关键：强制立即将缓冲区内容输出到终端
        # 否则 Python 会等缓冲区满或换行符才真正打印，看不到"逐字出现"的效果
        for text_chunk in stream.text_stream:
            print(text_chunk, end="", flush=True)

    # 流结束后换行，避免下一行输出紧跟在回答末尾
    print()
    print("\n[Demo 1 结束] 流式输出完毕。")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2 : 流式输出 + 最终统计（usage）
# ─────────────────────────────────────────────────────────────────────────────

def demo_stream_with_stats():
    """
    流式输出 + 最终统计（usage）

    关键 API：stream.get_final_message()
    ─────────────────────────────────────
    在 with 代码块内（流还未关闭时）调用 get_final_message() 会：
      1. 等待流全部接收完毕（如果还没结束的话）
      2. 返回一个完整的 Message 对象，与 messages.create() 的返回值结构完全相同
      3. 该对象包含 .usage（input_tokens / output_tokens），可用于费用估算和调试

    典型用途：
      - 流式展示给用户看的同时，后台记录 token 消耗
      - 在日志系统中同时保存完整响应文本和 usage 统计
    """
    section("Demo 2 : 流式输出 + 最终统计（usage）")

    prompt = "请列出 5 个提高编程效率的实用小技巧，每条不超过 20 字。"
    print(f"\n提问：{prompt}")
    print("\nClaude 的回答（流式输出）：")

    with client.messages.stream(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        # 先正常流式打印 —— 用户能实时看到内容
        for text_chunk in stream.text_stream:
            print(text_chunk, end="", flush=True)

        # 流接收完毕后，立即获取完整消息对象
        # 注意：必须在 with 代码块内调用，退出块后流已关闭
        final_message = stream.get_final_message()

    # 流结束后换行
    print()

    # 打印统计信息
    print("\n── 本次调用统计 ─────────────────────────────────────")
    print(f"  stop_reason  : {final_message.stop_reason}")
    print(f"  输入 tokens  : {final_message.usage.input_tokens}")
    print(f"  输出 tokens  : {final_message.usage.output_tokens}")
    print(f"  合计 tokens  : {final_message.usage.input_tokens + final_message.usage.output_tokens}")
    print(f"  响应 ID      : {final_message.id}")
    print("\n[Demo 2 结束]")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3 : 回调风格（MessageStream 子类 + on_text()）
# ─────────────────────────────────────────────────────────────────────────────

def demo_callback_style():
    """
    回调风格：直接迭代原始 SSE 事件流，在不同事件节点插入自定义逻辑

    为什么不用 event_handler 子类？
    ─────────────────────────────────────────────────────
    当前版本的 Anthropic SDK（>=0.40）已简化了流式 API，
    messages.stream() 不再接受 event_handler 参数。
    取而代之的是直接迭代 MessageStream 对象本身——
    每次 yield 一个 ParsedMessageStreamEvent，包含事件类型和原始数据。

    这种方式等价于"回调风格"——你可以在 for 循环中按事件类型分支处理，
    实现与 on_text / on_message 回调完全相同的逻辑，且更加透明易读。

    常见事件类型（event.type）：
      "message_start"          —— 流开始，包含初始 Message 对象（无内容）
      "content_block_start"    —— 内容块开始（通常是 text 类型块）
      "content_block_delta"    —— 文本增量（event.delta.text 即新增文本）
      "content_block_stop"     —— 内容块结束
      "message_delta"          —— 消息级增量（包含 stop_reason 和 usage）
      "message_stop"           —— 整个流结束
    """
    section("Demo 3 : 回调风格（逐事件处理 + 统计 TTFT）")

    prompt = "请解释一下为什么 Python 的 GIL（全局解释器锁）会影响多线程性能，并给出规避方案。"
    print(f"\n提问：{prompt}")
    print("\nClaude 的回答（事件驱动流式输出）：")

    token_count = 0           # 记录收到的文本增量批次数
    first_chunk_time = None   # 首 token 延迟（TTFT: Time To First Token）
    start_time = time.time()

    with client.messages.stream(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        # 直接迭代 stream 对象可以获取所有原始 SSE 事件
        # 每个 event 都有 .type 属性，根据类型做不同处理
        for event in stream:
            if event.type == "content_block_delta":
                # content_block_delta 事件携带文本增量
                # event.delta.type == "text_delta" 表示这是文本内容（区别于工具调用）
                if event.delta.type == "text_delta":
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                    token_count += 1
                    # 在"回调点"打印文本，flush=True 体现实时效果
                    print(event.delta.text, end="", flush=True)

            # 其他事件（message_start / content_block_start / message_stop 等）
            # 此处不做额外处理，但你可以在这里记录日志、更新进度条等

        # 流结束后获取完整消息（usage 等统计数据）
        final_message = stream.get_final_message()

    elapsed = time.time() - start_time

    # 换行并打印统计
    print()
    print("\n── 事件驱动统计 ─────────────────────────────────────")
    print(f"  首 token 延迟（TTFT） : {first_chunk_time:.3f} 秒")
    print(f"  总耗时               : {elapsed:.3f} 秒")
    print(f"  收到的文本增量批次    : {token_count}")
    print(f"  输入 tokens          : {final_message.usage.input_tokens}")
    print(f"  输出 tokens          : {final_message.usage.output_tokens}")
    print("\n[Demo 3 结束]")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 4 : 流式 vs 非流式对比
# ─────────────────────────────────────────────────────────────────────────────

def demo_streaming_vs_sync():
    """
    对比说明：流式 vs 非流式 —— 什么场景选哪种？

    结论：
    ─────────────────────────────────────────────────────────────
    ┌──────────────┬────────────────────────────┬────────────────────────────┐
    │              │        非流式（sync）        │       流式（streaming）     │
    ├──────────────┼────────────────────────────┼────────────────────────────┤
    │ 总网络耗时   │ 相同（模型计算速度一样）     │ 相同                       │
    │ 感知延迟     │ 高（需等全部生成完毕）       │ 低（第一个 token 即可显示） │
    │ 代码复杂度   │ 低（直接用返回值）           │ 中（需处理迭代/回调）       │
    │ 适用场景     │ 批处理、后台任务、单元测试   │ 聊天 UI、实时 API、长文生成 │
    └──────────────┴────────────────────────────┴────────────────────────────┘

    选择建议：
      ✅ 使用流式：聊天机器人界面、实时翻译、代码补全工具、长篇文章生成
      ✅ 使用非流式：批量数据处理、后台摘要生成、单元测试、结果需要整体处理时
    """
    section("Demo 4 : 流式 vs 非流式对比（时间感知实验）")

    prompt = "请用 100 字左右，介绍一下黑洞的形成过程。"
    print(f"\n提问：{prompt}")

    # ── 非流式：等待全部生成完毕后一次性返回 ─────────────────────────────────
    print("\n[非流式] 等待中...")
    t0 = time.time()
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    t_sync = time.time() - t0
    sync_text = response.content[0].text
    print(f"[非流式] 耗时 {t_sync:.2f}s 后一次性输出：\n{sync_text}")

    print()

    # ── 流式：立即开始打印，直到结束 ────────────────────────────────────────
    print("[流式] 立即开始输出（注意第一个字出现的速度）：")
    t0 = time.time()
    first_chunk_elapsed = None

    with client.messages.stream(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text_chunk in stream.text_stream:
            if first_chunk_elapsed is None:
                first_chunk_elapsed = time.time() - t0
            print(text_chunk, end="", flush=True)

    t_stream_total = time.time() - t0
    print()

    # 打印对比结论
    print(f"\n── 对比结果 ─────────────────────────────────────────")
    print(f"  非流式总耗时           : {t_sync:.2f}s（用户等待时间）")
    print(f"  流式首 token 延迟      : {first_chunk_elapsed:.2f}s（用户感知到响应的时间）")
    print(f"  流式总耗时             : {t_stream_total:.2f}s")
    print(f"\n  结论：流式首 token 延迟通常比非流式总耗时低 {((t_sync - first_chunk_elapsed) / t_sync * 100):.0f}%+，")
    print(f"         用户几乎感觉不到等待 —— 这就是流式的核心价值。")
    print("\n[Demo 4 结束]")


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Claude API 流式输出（Streaming）教学 Demo")
    print(f"  模型：{MODEL.split('/')[-1]}")
    print("=" * 60)

    demo_basic_streaming()
    demo_stream_with_stats()
    demo_callback_style()
    demo_streaming_vs_sync()

    print("\n" + "=" * 60)
    print("  所有 Demo 运行完毕！")
    print("  下一步：尝试将 demo_basic_streaming() 接入你自己的 CLI 工具，")
    print("          体验实时打字机效果带来的用户体验提升。")
    print("=" * 60)


if __name__ == "__main__":
    main()
