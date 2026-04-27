"""
主题：为什么需要 Transformer？—— 从 RNN 到 Transformer 的演进

学习目标：
  1. 理解序列模型（RNN/LSTM）处理文本的方式及其根本缺陷
  2. 掌握 Transformer 解决了哪些核心问题
  3. 了解 "Attention Is All You Need" 论文的历史意义
  4. 直观感受"顺序处理"与"并行处理"的性能差异

前置知识：
  - Python 基础（列表、循环、函数）
  - 对"神经网络"有基本概念即可（知道是机器学习模型）

课程顺序：这是 06_transformer 模块的第 1 个文件，建议从此开始。
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import time

# ══════════════════════════════════════════════════════════════════════════════
# 第一部分：RNN 的工作方式 —— 像人类逐字阅读
# ══════════════════════════════════════════════════════════════════════════════

def demo_rnn_sequential(sentence):
    """
    模拟 RNN（循环神经网络）处理句子的方式：
    - 每次只能读一个词
    - 必须等上一个词处理完才能处理下一个词
    - 用 "hidden_state"（隐藏状态）记忆之前读过的内容
    """
    print("\n【RNN 处理方式】逐词顺序处理（必须等待上一步完成）")
    print("-" * 50)

    words = sentence.split()
    hidden_state = "空记忆"   # 初始时记忆为空

    for i, word in enumerate(words):
        # 模拟每个词需要一定时间处理
        time.sleep(0.05)

        # RNN 核心：新的隐藏状态 = f(当前词, 上一个隐藏状态)
        # 这就是为什么必须顺序处理 —— 下一步依赖上一步的结果
        hidden_state = f"记住了前{i+1}个词"

        print(f"  步骤 {i+1:2d}: 读入 [{word:10s}] → 当前记忆: {hidden_state}")

    print(f"\n  ✓ 处理完毕，总共 {len(words)} 步，必须按顺序完成")
    return hidden_state


# ══════════════════════════════════════════════════════════════════════════════
# 第二部分：RNN 的三大致命缺陷
# ══════════════════════════════════════════════════════════════════════════════

def explain_rnn_problems():
    """
    用具体例子说明 RNN 的核心问题
    """
    print("\n\n【RNN 的三大缺陷】")
    print("=" * 60)

    # 缺陷1：长距离依赖问题（梯度消失）
    print("\n缺陷 1：遗忘问题（长距离依赖 / 梯度消失）")
    print("-" * 50)

    long_sentence = (
        "今天  早上  我  去  了  那家  我  三年前  和  朋友  "
        "一起  去过  的  ___  咖啡馆"
    )
    print(f"  句子：{long_sentence}")
    print()
    print("  问题：要填写 ___ 处的词，需要联系到句子开头的'今天早上'")
    print("  RNN 必须把这个信息保存在隐藏状态里，经过 15 个词后才用到")
    print("  → 经过多步传递，早期信息会逐渐'稀释'，导致模型遗忘")
    print("  → 这就是'梯度消失'问题的直觉理解")

    # 缺陷2：无法并行
    print("\n缺陷 2：无法并行计算（训练极慢）")
    print("-" * 50)
    print("  RNN 必须等第 t 步完成，才能开始第 t+1 步")
    print("  就像流水线上每道工序必须等前一道完成")
    print("  → 即使有 1000 个 GPU，也无法加速！")
    print("  → 训练一个大型 RNN 可能需要数周甚至数月")

    # 缺陷3：序列越长越难处理
    print("\n缺陷 3：长文本处理能力差")
    print("-" * 50)
    print("  RNN 用固定大小的'隐藏状态'记住所有历史信息")
    print("  对于 10 个词：还好")
    print("  对于 100 个词：开始困难")
    print("  对于 10000 个词（一篇文章）：几乎失败")
    print("  → Transformer 可以直接处理数万词的上下文！")


# ══════════════════════════════════════════════════════════════════════════════
# 第三部分：Transformer 的革命 —— 并行处理 + 全局注意力
# ══════════════════════════════════════════════════════════════════════════════

def demo_transformer_parallel(sentence):
    """
    模拟 Transformer 处理句子的方式：
    - 所有词同时处理（真正的并行）
    - 每个词可以直接"看到"句子中的所有其他词
    - 不需要依赖隐藏状态逐步传递
    """
    print("\n\n【Transformer 处理方式】所有词同时并行处理")
    print("-" * 50)

    words = sentence.split()
    print(f"  输入句子：{sentence}")
    print(f"  共 {len(words)} 个词，全部同时处理 ↓")
    print()

    # 模拟并行处理：所有词同时开始
    print("  时刻 T=1（同时处理所有词）：")
    for i, word in enumerate(words):
        # 关键区别：每个词可以直接看到所有其他词
        other_words = [w for j, w in enumerate(words) if j != i]
        attention_targets = "、".join(other_words[:3]) + ("..." if len(other_words) > 3 else "")
        print(f"    [{word:8s}] 同时关注: {attention_targets}")

    print()
    print(f"  ✓ 一步完成！无论句子多长，都只需 1 步并行处理")
    print(f"  ✓ GPU 可以完全发挥并行优势，训练速度提升数十倍")


# ══════════════════════════════════════════════════════════════════════════════
# 第四部分：并行 vs 顺序性能对比
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_sequential_vs_parallel(text_lengths):
    """
    用时间测量直观展示并行与顺序处理的性能差距
    """
    print("\n\n【性能对比】顺序处理 vs 并行处理")
    print("=" * 60)
    print(f"  {'文本长度':>8s} | {'RNN顺序(模拟)':>12s} | {'Transformer并行':>15s} | {'加速比':>8s}")
    print("-" * 55)

    for n_words in text_lengths:
        # 模拟 RNN：每个词串行处理
        rnn_time = n_words * 0.001  # 1ms 每个词

        # 模拟 Transformer：所有词并行，只需固定时间
        # 实际上注意力计算是 O(n²)，但对比 RNN 的 O(n) 串行，
        # GPU 并行后 Transformer 仍快得多
        transformer_time = max(0.005, n_words * 0.0001)  # 基础延迟 + 极少增量

        speedup = rnn_time / transformer_time

        print(f"  {n_words:>8d} 词 | {rnn_time*1000:>10.1f} ms | {transformer_time*1000:>13.1f} ms | {speedup:>6.0f}x")


# ══════════════════════════════════════════════════════════════════════════════
# 第五部分："Attention Is All You Need" —— 2017 年的革命
# ══════════════════════════════════════════════════════════════════════════════

def explain_transformer_history():
    """
    简述 Transformer 的历史背景和影响
    """
    print("\n\n【历史背景】Transformer 如何改变了 AI 世界")
    print("=" * 60)

    timeline = [
        ("2013", "Word2Vec",     "词向量出现，开始用向量表示单词含义"),
        ("2015", "Attention",    "注意力机制首次用于机器翻译（配合RNN使用）"),
        ("2017", "Transformer",  "Google 发布《Attention Is All You Need》—— 革命！"),
        ("2018", "BERT",         "Google 发布 BERT，NLP 各项任务刷新纪录"),
        ("2018", "GPT-1",        "OpenAI 发布 GPT，开启语言生成新时代"),
        ("2019", "GPT-2",        "GPT-2 因'过于强大'而延迟发布"),
        ("2020", "GPT-3",        "1750亿参数，少样本学习能力震惊业界"),
        ("2022", "ChatGPT",      "基于 GPT-3.5，对话式AI产品爆发"),
        ("2023", "GPT-4/Claude", "多模态大模型，接近甚至超越人类专家水平"),
        ("2024", "Claude 3",     "Anthropic 发布 Opus/Sonnet/Haiku 系列"),
        ("2025", "Claude 4+",    "你正在用的这个 Claude 模型！"),
    ]

    for year, name, desc in timeline:
        marker = "★" if year == "2017" else " "
        print(f"  {marker} {year}  [{name:12s}]  {desc}")

    print()
    print("  ★ 标注处是最关键的节点：Transformer 让一切成为可能")
    print()
    print("  核心论文三要素：")
    print("    · 标题：《Attention Is All You Need》（只需要注意力）")
    print("    · 作者：Vaswani 等 8 人（Google Brain / Google Research）")
    print("    · 核心：用纯注意力机制替代 RNN，不需要循环结构")


# ══════════════════════════════════════════════════════════════════════════════
# 第六部分：Transformer 的核心组件预览
# ══════════════════════════════════════════════════════════════════════════════

def preview_transformer_components():
    """
    提前展示 Transformer 的完整结构，后续课程会逐一深入
    """
    print("\n\n【架构预览】Transformer 的核心组件")
    print("=" * 60)

    diagram = """
    输入文本："我 爱 自然语言处理"
         │
         ▼
    ┌─────────────────────────────────┐
    │  词嵌入 (Embedding)              │ ← 把词转换为向量
    │  + 位置编码 (Positional Encoding) │ ← 告诉模型词的位置
    └──────────────┬──────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────┐
    │         编码器 (Encoder)         │
    │  ┌──────────────────────────┐   │
    │  │  多头自注意力 (Multi-Head │   │ ← 让每个词关注所有词
    │  │  Self-Attention)          │   │
    │  └──────────────────────────┘   │
    │  ┌──────────────────────────┐   │
    │  │  前馈神经网络 (FFN)       │   │ ← 进一步提炼特征
    │  └──────────────────────────┘   │
    └──────────────┬──────────────────┘
                   │  （可堆叠多层，如 12 层、24 层）
                   ▼
    ┌─────────────────────────────────┐
    │         解码器 (Decoder)         │
    │  [用于生成输出，翻译/对话等任务]  │
    └──────────────┬──────────────────┘
                   │
                   ▼
    输出预测："I love natural language processing"
    """

    print(diagram)

    print("  后续课程将逐步深入每个组件：")
    print("  · 第02课：注意力机制直觉（无需数学）")
    print("  · 第03课：自注意力的数学推导（Q/K/V）")
    print("  · 第04课：多头注意力 + 位置编码")
    print("  · 第05课：完整架构 Encoder + Decoder")
    print("  · 第06课：现代大模型 BERT / GPT / Claude")


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Transformer 第01课：为什么需要 Transformer？")
    print("=" * 60)

    # 演示句子
    example = "the cat sat on the mat and looked at the bird"

    # 1. 展示 RNN 顺序处理
    demo_rnn_sequential(example)

    # 2. 分析 RNN 的缺陷
    explain_rnn_problems()

    # 3. 展示 Transformer 并行处理
    demo_transformer_parallel(example)

    # 4. 性能对比
    benchmark_sequential_vs_parallel([10, 50, 100, 500, 1000])

    # 5. 历史背景
    explain_transformer_history()

    # 6. 组件预览
    preview_transformer_components()

    print("\n" + "=" * 60)
    print("  本课小结：")
    print("  · RNN 顺序处理 → 无法并行 + 容易遗忘")
    print("  · Transformer 并行处理 → 速度快 + 全局注意力")
    print("  · 2017年《Attention Is All You Need》开启了LLM时代")
    print()
    print("  下一课：注意力机制直觉理解（02_attention_intuition.py）")
    print("=" * 60)


if __name__ == "__main__":
    main()
