"""
主题：Transformer 完整架构 —— Encoder 与 Decoder 深度解析

学习目标：
  1. 掌握 Encoder Block 的4个组件：多头注意力、FFN、残差连接、LayerNorm
  2. 掌握 Decoder Block 的额外组件：掩码注意力和交叉注意力
  3. 理解残差连接为什么能防止梯度消失
  4. 通过 ASCII 图理解完整的 Transformer 数据流

前置知识：
  - 完成 01-04 课
  - 理解自注意力和多头注意力

依赖：pip install numpy
课程顺序：这是 06_transformer 模块的第 5 个文件。
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import math

np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# 第一部分：Encoder Block 的4个组件
# ══════════════════════════════════════════════════════════════════════════════

def explain_encoder_components():
    print("=" * 60)
    print("  Encoder Block：4个核心组件")
    print("=" * 60)

    print("""
  完整的 Encoder Block 结构：

  ┌────────────────────────────────────────────┐
  │             Encoder Block                  │
  │                                            │
  │  输入 X ─────────────────────┐             │
  │       │                      │ (残差连接)   │
  │       ▼                      │             │
  │  [多头自注意力]               │             │
  │       │                      │             │
  │       └──────────────────────┘             │
  │       │                                    │
  │  [Layer Normalization]                     │
  │       │                                    │
  │       ├──────────────────────┐             │
  │       │                      │ (残差连接)   │
  │       ▼                      │             │
  │  [前馈神经网络 FFN]            │             │
  │       │                      │             │
  │       └──────────────────────┘             │
  │       │                                    │
  │  [Layer Normalization]                     │
  │       │                                    │
  │  输出 Z ────────────────────────────────── │
  └────────────────────────────────────────────┘

  公式：
    Z1 = LayerNorm(X + MultiHeadAttn(X))     # 注意力 + 残差
    Z2 = LayerNorm(Z1 + FFN(Z1))             # FFN + 残差
  """)


def explain_residual_connection():
    """
    残差连接：output = F(x) + x

    核心作用：防止梯度消失，让信息可以"跳过"某些层
    就像高速公路：可以走小路（F(x)），也可以走主干道（x）
    """
    print("\n\n" + "=" * 60)
    print("  残差连接（Residual Connection）")
    print("=" * 60)

    print("""
  普通网络（没有残差）：
    x → [层1] → [层2] → [层3] → ... → [层N] → 输出
    问题：梯度反向传播时，经过太多层会变得极小（梯度消失）
         模型深度越深，训练越困难

  残差网络：
    x → [层1] ──→ + → [层2] ──→ + → ... → 输出
         └──────→ ↑    └──────→ ↑
         （原始 x 直接加到输出）

  数学形式：
    output = F(x) + x  （而非 output = F(x)）

  为什么有效：
  · 梯度反向传播时，"+ x" 提供了一条不衰减的直接通道
  · 即使 F(x) 的梯度很小，总梯度也不会消失
  · 最差情况：F(x) 学到 0，相当于该层被跳过，不会使性能变差
  · 使得模型可以轻松训练 100+ 层的深度网络
  """)

    # 数值演示：梯度消失对比
    print("  数值演示：梯度在 10 层网络中的衰减")
    print()

    n_layers = 10
    gradient_per_layer = 0.5  # 每层梯度乘以0.5（模拟衰减）

    grad_no_residual = 1.0
    grad_with_residual = 1.0

    print(f"  {'层数':>4s}  {'无残差梯度':>12s}  {'有残差梯度（简化）':>18s}")
    print("  " + "-" * 42)
    for layer in range(1, n_layers + 1):
        grad_no_residual *= gradient_per_layer
        # 有残差时：梯度 = F(x)的梯度 + 残差的梯度(≈1)
        # 简化模型：每层至少保留 0.9
        grad_with_residual = grad_with_residual * gradient_per_layer + 0.5
        bar_no = "█" * max(0, int(grad_no_residual * 20))
        bar_yes = "█" * min(20, int(grad_with_residual * 5))
        print(f"  层{layer:>2d}   {grad_no_residual:>12.5f}   {grad_with_residual:>8.3f}  {bar_no or '(消失)'}  {bar_yes}")

    print()
    print("  → 无残差：10 层后梯度 ≈ 0（无法学习）")
    print("  → 有残差：梯度保持稳定，深层网络可正常训练")


def explain_layer_norm():
    """
    Layer Normalization：对每个样本的特征维度做归一化
    作用：稳定训练过程，防止激活值爆炸或消失
    """
    print("\n\n" + "=" * 60)
    print("  Layer Normalization（层归一化）")
    print("=" * 60)

    print("""
  归一化：将数据调整为均值=0、方差=1 的分布

  LayerNorm 公式：
    y = (x - mean(x)) / sqrt(var(x) + ε)
    然后用可学习参数 γ 和 β 缩放和平移

  为什么需要归一化：
  · 不同批次的数据分布不同，模型训练不稳定
  · 激活值如果太大或太小，梯度会爆炸或消失
  · 归一化后，每层输入都在相似范围内，训练更稳定
  """)

    # 数值演示
    x = np.array([2.0, 5.0, 3.0, 8.0, 1.0, 4.0])
    mean = x.mean()
    std = x.std()
    x_norm = (x - mean) / (std + 1e-8)

    print(f"  原始向量：{x}")
    print(f"  均值 = {mean:.2f}，标准差 = {std:.2f}")
    print(f"  归一化后：{np.round(x_norm, 3)}")
    print(f"  归一化后均值 ≈ {x_norm.mean():.6f}，标准差 ≈ {x_norm.std():.3f}")


def explain_ffn():
    """
    前馈神经网络（Feed-Forward Network）
    结构：Linear → ReLU → Linear
    作用：在每个位置独立地对特征进行非线性变换
    """
    print("\n\n" + "=" * 60)
    print("  前馈神经网络（FFN）")
    print("=" * 60)

    print("""
  FFN 结构：
    FFN(x) = max(0, x·W₁ + b₁) · W₂ + b₂

    · 第一层将维度从 d_model 扩展到 4×d_model（如 768 → 3072）
    · 中间使用 ReLU 激活函数引入非线性
    · 第二层将维度压缩回 d_model

  作用解释：
  · 注意力层负责"信息聚合"（把其他词的信息集中过来）
  · FFN 层负责"信息提炼"（对聚合后的信息做非线性变换）
  · 研究表明：FFN 存储了大量"事实知识"，如"巴黎是法国首都"

  参数量（以 GPT-3 为例）：
    d_model = 12288，FFN 隐藏层 = 4×12288 = 49152
    FFN 参数量 = 2 × 12288 × 49152 ≈ 12亿（占模型的约60%！）
  """)


# ══════════════════════════════════════════════════════════════════════════════
# 第二部分：Decoder Block —— 额外的两种注意力
# ══════════════════════════════════════════════════════════════════════════════

def explain_decoder_components():
    print("\n\n" + "=" * 60)
    print("  Decoder Block：三种注意力层")
    print("=" * 60)

    print("""
  Decoder Block（比 Encoder 多了掩码自注意力和交叉注意力）：

  ┌──────────────────────────────────────────────────┐
  │              Decoder Block                       │
  │                                                  │
  │  目标序列输入 ──────────────────────┐             │
  │       │                            │ (残差)       │
  │       ▼                            │             │
  │  [掩码多头自注意力]                  │             │
  │  (Masked Multi-Head Self-Attn)     │             │
  │       │                            │             │
  │       └────────────────────────────┘             │
  │       │                                          │
  │  [Layer Norm]                                    │
  │       │                                          │
  │       ├────────────────────────────┐             │
  │       │         Encoder输出 ─→ K,V │ (残差)       │
  │       ▼                            │             │
  │  [多头交叉注意力]                   │             │
  │  (Multi-Head Cross-Attn)          │             │
  │  Q=来自上层，K/V=来自Encoder       │             │
  │       │                            │             │
  │       └────────────────────────────┘             │
  │       │                                          │
  │  [Layer Norm]                                    │
  │       │                                          │
  │       ├────────────────────────────┐             │
  │       │                            │ (残差)       │
  │       ▼                            │             │
  │  [前馈神经网络 FFN]                  │             │
  │       │                            │             │
  │       └────────────────────────────┘             │
  │       │                                          │
  │  [Layer Norm]                                    │
  │       │                                          │
  │  输出 ────────────────────────────────────────── │
  └──────────────────────────────────────────────────┘

  三种注意力：
  1. 掩码自注意力（Masked Self-Attention）：
     · Q/K/V 都来自当前已生成的目标序列
     · "掩码"：生成第t个词时，只能看到前t-1个词，不能"偷看"未来
     · 训练时用于预测下一个词

  2. 交叉注意力（Cross-Attention）：
     · Q 来自解码器（当前生成状态）
     · K/V 来自编码器输出（原始输入的表示）
     · 作用：让生成的词能"参考"原始输入

  3. 前馈网络（FFN）：与 Encoder 中相同
  """)


def demo_causal_mask():
    """
    演示解码器中的因果掩码（Causal Mask）
    防止在训练时"偷看"未来的词
    """
    print("\n\n" + "=" * 60)
    print("  因果掩码（Causal Mask）—— 防止偷看未来")
    print("=" * 60)

    seq_len = 5
    words = ["<SOS>", "I", "love", "Paris", "<EOS>"]

    print(f"\n  目标序列：{' '.join(words)}")
    print()
    print("  训练时的目标：给定前面的词，预测下一个词")
    print("  例如：给定 '<SOS>' → 预测 'I'")
    print("       给定 '<SOS> I' → 预测 'love'")
    print("       给定 '<SOS> I love' → 预测 'Paris'")
    print()

    # 因果掩码：下三角矩阵（包含对角线），上三角=0（被遮蔽）
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)  # 上三角（不含对角）为1（遮蔽）

    print("  因果掩码矩阵（1=被遮蔽，0=可见）：")
    print(f"  {'':8s}", end="")
    for w in words:
        print(f"  {w:>6s}", end="")
    print()
    for i, row_word in enumerate(words):
        print(f"  {row_word:8s}", end="")
        for j in range(seq_len):
            if mask[i][j] == 1:
                print(f"  {'██':>6s}", end="")  # 被遮蔽
            else:
                print(f"  {'可见':>6s}", end="")
        print()

    print()
    print("  解读：")
    print("  · 第0行（'<SOS>'）：只能看到自己，不能看后面的词")
    print("  · 第2行（'love'）：可以看到 '<SOS>', 'I', 'love' 自身，但不能看 'Paris', '<EOS>'")
    print("  · 第4行（'<EOS>'）：可以看到整个句子")
    print()
    print("  这样训练的效果：")
    print("  · 模型被迫只用历史信息预测当前词")
    print("  · 避免了信息泄露，保证生成时的合理性")


# ══════════════════════════════════════════════════════════════════════════════
# 第三部分：完整数据流 —— 机器翻译任务
# ══════════════════════════════════════════════════════════════════════════════

def demo_full_transformer_flow():
    """
    用机器翻译任务演示完整的 Transformer 数据流
    原始论文的任务：英文→法文
    """
    print("\n\n" + "=" * 60)
    print("  完整 Transformer 数据流（机器翻译：英→法）")
    print("=" * 60)

    steps = [
        ("输入",     "英文：I love Paris"),
        ("词嵌入",   "I=[0.8,0.2,...], love=[0.3,0.7,...], Paris=[0.5,0.4,...]"),
        ("位置编码", "每个词加上位置向量，使模型感知词序"),
        ("Encoder层1", "多头自注意力：每个词关注全句，获得语境信息"),
        ("Encoder层2", "FFN + LayerNorm：对特征进行非线性提炼"),
        ("(...重复N层)", "GPT: 12-96层  BERT: 12-24层  GPT-3: 96层"),
        ("Encoder输出", "每个英文词都有包含全部上下文的向量表示"),
        ("Decoder输入", "法文开始符 <SOS>"),
        ("掩码自注意力", "解码器内部的自注意力（只看历史）"),
        ("交叉注意力",  "Q=当前解码状态, K/V=Encoder输出（参考原文）"),
        ("FFN+输出层",  "线性层 + Softmax → 每个词的概率分布"),
        ("生成词",    "选概率最大的词：J'aime"),
        ("循环生成",  "将生成的词加入输入，继续生成：J'aime Paris"),
        ("停止条件",  "直到生成 <EOS> 或达到最大长度"),
    ]

    print()
    for step, desc in steps:
        marker = "━━" if "步骤" not in step and step.startswith("(") else "→ "
        prefix = "  " if "重复" in step else ""
        print(f"  {prefix}[{step:12s}] {desc}")

    print()
    print("  整体数据流总结：")
    print("""
  输入序列 → [词嵌入 + 位置编码] → N× Encoder Block → Encoder输出
                                                          ↓
  目标序列 → [词嵌入 + 位置编码] → N× Decoder Block → 线性+Softmax → 预测词
              （已生成的部分）        （包含掩码自注意力和交叉注意力）
  """)


# ══════════════════════════════════════════════════════════════════════════════
# 第四部分：真实模型的规模对比
# ══════════════════════════════════════════════════════════════════════════════

def show_model_scales():
    print("\n\n" + "=" * 60)
    print("  真实 Transformer 模型规模对比")
    print("=" * 60)

    models = [
        # (模型名,  类型,   层数,  头数,  d_model, 参数量,   上下文长度)
        ("BERT-base",   "Enc",  12,  12,  768,    "110M",   "512 词"),
        ("BERT-large",  "Enc",  24,  16,  1024,   "340M",   "512 词"),
        ("GPT-2 small", "Dec",  12,  12,  768,    "117M",   "1024 词"),
        ("GPT-2 XL",    "Dec",  48,  25,  1600,   "1.5B",   "1024 词"),
        ("GPT-3",       "Dec",  96,  96,  12288,  "175B",   "2048 词"),
        ("GPT-4",       "Dec",  "~120","~96","未公开","~1T",  "128K 词"),
        ("Claude 3",    "Dec",  "未公开","未公开","未公开","~100B-1T","200K 词"),
        ("Llama 3 8B",  "Dec",  32,  32,  4096,   "8B",     "128K 词"),
    ]

    print(f"\n  {'模型':12s}  {'类型':4s}  {'层数':>6s}  {'头数':>6s}  {'d_model':>8s}  {'参数量':>8s}  {'上下文'}")
    print("  " + "-" * 75)
    for name, arch, layers, heads, d_model, params, ctx in models:
        print(f"  {name:12s}  {arch:4s}  {str(layers):>6s}  {str(heads):>6s}  {str(d_model):>8s}  {params:>8s}  {ctx}")

    print()
    print("  类型说明：Enc=Encoder-Only，Dec=Decoder-Only")
    print()
    print("  关键观察：")
    print("  · 层数越深 + d_model越大 + 数据越多 → 能力越强")
    print("  · GPT-3 的 175B 参数 = 1750亿个数字（约 700GB）")
    print("  · Claude 等现代模型可以处理 20万+ 词的超长上下文")


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Transformer 第05课：完整的 Encoder-Decoder 架构")
    print("=" * 60)

    explain_encoder_components()
    explain_residual_connection()
    explain_layer_norm()
    explain_ffn()
    explain_decoder_components()
    demo_causal_mask()
    demo_full_transformer_flow()
    show_model_scales()

    print("\n" + "=" * 60)
    print("  本课小结：")
    print("  · Encoder Block = 多头注意力 + FFN + 残差 + LayerNorm")
    print("  · Decoder Block 额外加了：掩码注意力 + 交叉注意力")
    print("  · 残差连接 防止梯度消失，让模型可以深达 100+ 层")
    print("  · LayerNorm 稳定训练，让各层输入保持相似范围")
    print("  · 因果掩码 防止解码时偷看未来词")
    print()
    print("  下一课：BERT vs GPT —— 现代大模型如何演化")
    print("          (06_bert_gpt_llm.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
