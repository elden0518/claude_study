"""
主题：API 参数调优 —— 控制模型行为的关键旋钮

学习目标：
  1. 掌握 temperature 参数：控制输出的随机性/创造性
  2. 掌握 max_tokens 参数：控制输出的最大长度
  3. 了解 top_p（核采样）参数：与 temperature 的关系
  4. 了解模型选择策略：速度/成本 vs 质量/均衡
  5. 掌握 stop_sequences 参数：自定义停止词

前置知识：
  - 已完成 01_hello_claude.py 和 02_messages_format.py
  - 了解基本 API 调用结构

课程顺序：这是 01_basics 模块的第三个文件，建议按序学习。
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
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# 统一使用的模型（通过代理前缀 ppio/pa/ 路由到 claude-sonnet-4-6）
MODEL = "ppio/pa/claude-sonnet-4-6"


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def call(prompt: str, **kwargs) -> str:
    """封装 API 调用，返回纯文本内容。kwargs 透传给 messages.create()。"""
    resp = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return resp.content[0].text


def section(title: str):
    """打印分隔标题，方便阅读输出。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1 : temperature —— 确定性 vs 创造性
# ─────────────────────────────────────────────────────────────────────────────
def demo_temperature():
    """
    temperature 参数详解
    ─────────────────────
    取值范围：0.0 ~ 1.0（部分 API 可达 2.0，Claude 推荐不超过 1.0）
    默认值：1.0

    含义：
      - 低值（0）：模型倾向于选择概率最高的 token，输出稳定、可预测、适合
                   事实问答、代码生成、数据提取等需要准确性的场景。
      - 高值（1）：模型从更广的概率分布中采样，输出更多样、更有创意，
                   适合故事创作、头脑风暴、广告文案等需要新颖性的场景。

    最佳实践：
      - 写代码、提取信息 → temperature=0
      - 日常对话、问答   → temperature=0.3~0.7
      - 创意写作、诗歌   → temperature=0.8~1.0
      - 避免同时调高 temperature 和 top_p（会叠加随机性，结果难以预期）
    """
    section("Demo 1 : temperature — 确定性 vs 创造性")

    prompt = "用一句话描述月亮（20字以内）"

    # ── temperature=0：确定性输出 ──────────────────────────────────────────────
    # max_tokens 限为 60，足够容纳一句话，节省 token
    out_low = call(prompt, max_tokens=60, temperature=0)
    print(f"\n[temperature=0  确定性] {out_low}")

    # ── temperature=1：创造性输出 ──────────────────────────────────────────────
    out_high = call(prompt, max_tokens=60, temperature=1)
    print(f"[temperature=1  创造性] {out_high}")

    print("\n  -> 注意观察：temperature=0 的两次结果几乎相同；")
    print("              temperature=1 可能更有诗意或用词更多变。")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2 : max_tokens —— 控制输出长度（截断实验）
# ─────────────────────────────────────────────────────────────────────────────
def demo_max_tokens():
    """
    max_tokens 参数详解
    ─────────────────────
    取值范围：1 ~ 模型上限（claude-sonnet-4-6 上限 8192）
    默认值：无默认，必须显式指定

    含义：
      - 限制模型本次回复的最大 token 数量（token ≈ 0.75 个英文单词 / 约 1.5 个汉字）
      - 模型会在达到上限时"硬截断"，可能截断在句子中间
      - 并非越大越好：过大会增加费用，实际任务应根据预期输出长度合理设置

    最佳实践：
      - 短回答/分类：50~150
      - 一般问答：256~512
      - 长文生成/总结：1024~4096
      - 避免设置远大于实际需要的值，以免产生不必要费用
    """
    section("Demo 2 : max_tokens — 控制输出长度")

    prompt = "请详细介绍一下人工智能的发展历史，包括重要里程碑事件。"

    # ── max_tokens=50：非常短，极可能被截断 ───────────────────────────────────
    out_50 = call(prompt, max_tokens=50, temperature=0)
    print(f"\n[max_tokens=50  极短输出，可能截断]\n{out_50}")

    # ── max_tokens=200：中等长度 ───────────────────────────────────────────────
    out_200 = call(prompt, max_tokens=200, temperature=0)
    print(f"\n[max_tokens=200 中等长度]\n{out_200}")

    # ── max_tokens=500：较完整回答 ────────────────────────────────────────────
    out_500 = call(prompt, max_tokens=500, temperature=0)
    print(f"\n[max_tokens=500 较完整回答]\n{out_500}")

    print("\n  -> 注意观察：50 个 token 往往说到一半就被截断；")
    print("              500 个 token 能覆盖更多历史细节。")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3 : top_p —— 核采样说明
# ─────────────────────────────────────────────────────────────────────────────
def demo_top_p():
    """
    top_p（核采样）参数详解
    ─────────────────────────
    取值范围：0.0 ~ 1.0
    默认值：1.0（不截断，使用全部词汇概率分布）

    原理：
      模型在生成每个 token 时，会先把所有候选词按概率从高到低排列，
      然后累加概率，直到累计概率 >= top_p 为止，只在这个"核"范围内采样。
      - top_p=1.0 → 考虑全部候选词（无截断）
      - top_p=0.9 → 只在累计概率达到 90% 的词中采样，过滤掉低概率长尾
      - top_p=0.1 → 只考虑最高概率的少数几个词，输出极保守

    与 temperature 的关系：
      两者都会影响采样范围，但作用机制不同：
        temperature：拉伸/压缩整个概率分布（类比"调节分布形状"）
        top_p：直接截断低概率候选词（类比"限制候选词池大小"）

    最佳实践（Anthropic 官方建议）：
      ★ 每次只调节其中一个，另一个保持默认值！
        - 若想控制随机性 → 调 temperature，保持 top_p=1（默认）
        - 若想控制词汇多样性 → 调 top_p，保持 temperature=1（默认）
        - 同时修改两者会使行为难以预期，不推荐
    """
    section("Demo 3 : top_p — 核采样说明（演示默认 vs 保守）")

    prompt = "写一个关于宇宙的有趣冷知识（一句话）"

    # ── top_p=1.0（默认）：完整分布采样 ──────────────────────────────────────
    out_full = call(prompt, max_tokens=80, top_p=1.0, temperature=1)
    print(f"\n[top_p=1.0  完整分布，词汇多样] {out_full}")

    # ── top_p=0.3：只考虑高概率词汇 ──────────────────────────────────────────
    # 注意：此时 temperature 保持默认(1)，仅演示 top_p 的截断效果
    out_narrow = call(prompt, max_tokens=80, top_p=0.3, temperature=1)
    print(f"[top_p=0.3  保守采样，措辞更常规] {out_narrow}")

    print("\n  -> 建议：日常开发只需调 temperature；")
    print("           top_p 留给需要精细控制词汇多样性的高级场景。")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 4 : stop_sequences —— 自定义停止词
# ─────────────────────────────────────────────────────────────────────────────
def demo_stop_sequences():
    """
    stop_sequences 参数详解
    ─────────────────────────
    类型：list[str]，最多 5 个字符串
    默认值：[] （不设置任何自定义停止词）

    含义：
      当模型输出中出现列表中任意一个字符串时，立即停止生成，
      且该停止词本身不会出现在最终输出中。

    常见用途：
      1. 解析结构化输出：生成列表时遇到 "END" 就停止，方便后续解析
      2. 多轮对话分隔：遇到 "User:" 停止，避免模型自己扮演用户
      3. 代码生成：遇到特定标记停止，截取第一个完整代码块
      4. 模板填充：只填充到特定占位符为止

    最佳实践：
      - 停止词要足够独特，避免在正常回复中意外触发
      - 可结合 assistant pre-fill 一起使用，实现精准格式控制
    """
    section("Demo 4 : stop_sequences — 自定义停止词")

    # ── 示例：让模型列举事项，遇到 "END" 停止 ────────────────────────────────
    prompt = (
        "请依次列出三件让你开心的小事，每条以数字和点号开头，"
        "列完后在最后一行单独写 END，然后继续写其他任何内容。"
    )

    # 不设置 stop_sequences：会输出完整内容（包括 END 后的部分）
    out_no_stop = call(prompt, max_tokens=200, temperature=0)
    print(f"\n[无 stop_sequences  完整输出]\n{out_no_stop}")

    # 设置 stop_sequences=["END"]：遇到 END 立即停止
    out_with_stop = call(prompt, max_tokens=200, temperature=0, stop_sequences=["END"])
    print(f"\n[stop_sequences=['END']  遇到 END 后截断]\n{out_with_stop}")

    print("\n  -> 注意观察：设置停止词后，'END' 及其之后的内容不会出现在输出中。")

    # ── 补充说明：多轮对话场景 ───────────────────────────────────────────────
    print("\n  [进阶用法] 多轮对话中防止角色混淆：")
    print("    stop_sequences=['\\nUser:', '\\nHuman:']")
    print("    模型生成到换行+User: 时自动停止，避免 AI 自演用户发言。")


# ─────────────────────────────────────────────────────────────────────────────
# 附录注释：模型选择策略（无需实际调用，节省 token）
# ─────────────────────────────────────────────────────────────────────────────
# 模型名称                          速度    成本    能力    适用场景
# ─────────────────────────────────────────────────────────────────────────────
# claude-haiku-4-5-20251001          最快    最低    基础    分类、提取、简单问答、
#                                                           高并发/低延迟应用
# claude-sonnet-4-6（本文件使用）     均衡    中等    强     日常开发、复杂推理、
#                                                           代码生成、长文分析
# ─────────────────────────────────────────────────────────────────────────────
# 选择建议：
#   - 原型阶段先用 sonnet 保证效果，上线优化时根据任务复杂度选择 haiku 降低成本
#   - 对延迟敏感（< 1s 响应）的实时场景优先选 haiku
#   - 需要深度推理、长文档理解则坚持用 sonnet 或更高级模型


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Claude API 参数调优教学 Demo")
    print("  模型：", MODEL.split("/")[-1])  # 只显示模型简称
    print("=" * 60)

    demo_temperature()
    demo_max_tokens()
    demo_top_p()
    demo_stop_sequences()

    print("\n" + "=" * 60)
    print("  所有 Demo 运行完毕！")
    print("=" * 60)


if __name__ == "__main__":
    main()
