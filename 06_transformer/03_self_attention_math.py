"""
主题：自注意力的数学推导 —— 用 NumPy 手写 Q/K/V 计算

学习目标：
  1. 理解向量和矩阵的基本概念（从零开始）
  2. 掌握缩放点积注意力（Scaled Dot-Product Attention）的 4 步计算
  3. 亲手用 NumPy 实现自注意力，看到实际数字
  4. 理解 softmax 函数和温度参数的作用
  5. 可视化注意力权重矩阵

前置知识：
  - Python 基础
  - 完成 01/02 课
  - 首次使用 NumPy（本课会逐步介绍）

依赖：pip install numpy
课程顺序：这是 06_transformer 模块的第 3 个文件。
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import math

# 固定随机种子，保证每次运行结果一致（方便学习）
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 第一部分：NumPy 快速入门 —— 向量和矩阵
# ══════════════════════════════════════════════════════════════════════════════

def numpy_quick_intro():
    """
    对于没有 NumPy 基础的同学，先了解基本操作
    """
    print("=" * 60)
    print("  NumPy 快速入门（自注意力所需的数学工具）")
    print("=" * 60)

    # 向量：一维数组，可以表示一个词的含义
    print("\n── 1. 向量（Vector）：用数字列表表示一个对象 ──────────────")
    word_cat    = np.array([0.8, 0.2, 0.9, 0.1])   # "猫"的向量表示（假设）
    word_dog    = np.array([0.7, 0.3, 0.8, 0.2])   # "狗"的向量表示
    word_flower = np.array([0.1, 0.9, 0.1, 0.8])   # "花"的向量表示

    print(f"  '猫' 的向量：{word_cat}")
    print(f"  '狗' 的向量：{word_dog}")
    print(f"  '花' 的向量：{word_flower}")
    print("  （实际的词向量维度是 768 或 4096，这里用 4 维简化）")

    # 点积：衡量两个向量的相似度
    print("\n── 2. 点积（Dot Product）：衡量相似度 ─────────────────────")
    sim_cat_dog    = np.dot(word_cat, word_dog)
    sim_cat_flower = np.dot(word_cat, word_flower)
    print(f"  猫 · 狗   = {sim_cat_dog:.3f}  （数值较大 → 语义相近）")
    print(f"  猫 · 花   = {sim_cat_flower:.3f}  （数值较小 → 语义相差较远）")

    # 矩阵：二维数组，将多个向量堆叠
    print("\n── 3. 矩阵（Matrix）：多个向量的堆叠 ──────────────────────")
    sentence_matrix = np.stack([word_cat, word_dog, word_flower])
    print(f"  句子矩阵（3个词 × 4维向量）：")
    print(f"  shape = {sentence_matrix.shape}  意思是：3行（词数）× 4列（维度）")
    print(sentence_matrix)

    # 矩阵乘法：一次性计算所有词的相似度
    print("\n── 4. 矩阵乘法（@ 或 matmul）：批量计算相似度 ─────────────")
    similarity_matrix = sentence_matrix @ sentence_matrix.T
    print(f"  相似度矩阵 shape = {similarity_matrix.shape}  (3词 × 3词)")
    print("  每个格子 [i,j] = 第i个词与第j个词的点积相似度：")
    words = ["猫", "狗", "花"]
    print(f"  {'':4s} {'猫':>8s} {'狗':>8s} {'花':>8s}")
    for i, row_word in enumerate(words):
        vals = "  ".join(f"{v:>8.3f}" for v in similarity_matrix[i])
        print(f"  {row_word:4s} {vals}")


# ══════════════════════════════════════════════════════════════════════════════
# 第二部分：Softmax 函数详解
# ══════════════════════════════════════════════════════════════════════════════

def explain_softmax():
    """
    Softmax：将任意数值转换为概率分布（每个值∈(0,1)，总和=1）

    公式：softmax(x_i) = exp(x_i) / Σ exp(x_j)
    """
    print("\n\n" + "=" * 60)
    print("  Softmax：将分数变成概率")
    print("=" * 60)

    def softmax(x):
        # 减去最大值（数值稳定性技巧，防止 exp 溢出）
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # 示例：4个词的注意力分数
    raw_scores = np.array([2.0, 1.0, 0.1, 0.5])
    probs = softmax(raw_scores)

    print(f"\n  原始分数：{raw_scores}")
    print(f"  softmax后：{np.round(probs, 3)}")
    print(f"  总和 = {probs.sum():.3f}（等于1，这是softmax的保证）")
    print()
    print("  可视化：")
    for i, (score, prob) in enumerate(zip(raw_scores, probs)):
        bar = "█" * int(prob * 30)
        print(f"  词{i+1}: 原始分数={score:.1f} → 概率={prob:.3f}  {bar}")

    # 温度参数
    print("\n  【温度参数（Temperature）的作用】")
    print("  公式中除以 √d_k，相当于控制'温度'")
    print()
    temperatures = [0.5, 1.0, 2.0, 5.0]
    print(f"  相同原始分数：{raw_scores}")
    print(f"  {'温度':>8s}  {'结果概率（4个值）':>40s}  分布特征")
    print("  " + "-" * 75)
    for T in temperatures:
        scaled = raw_scores / T
        probs_t = softmax(scaled)
        desc = "更极端（几乎只关注最高分）" if T < 1 else \
               "标准"                     if T == 1 else \
               "更平均（关注所有词）"
        print(f"  {T:>8.1f}  {np.round(probs_t, 3)}  {desc}")

    print()
    print("  Transformer 中默认温度 = √d_k（d_k 是注意力头的维度）")
    print("  例如 d_k=64 → 温度 = √64 = 8，防止分数过大导致梯度消失")


# ══════════════════════════════════════════════════════════════════════════════
# 第三部分：手写缩放点积注意力（4步）
# ══════════════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(Q, K, V, mask=None, verbose=True):
    """
    缩放点积注意力的完整实现

    Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

    参数：
      Q: Query 矩阵  shape=(seq_len, d_k)
      K: Key   矩阵  shape=(seq_len, d_k)
      V: Value 矩阵  shape=(seq_len, d_v)
      mask: 可选，用于遮蔽未来位置（解码器使用）

    返回：
      output:  注意力输出  shape=(seq_len, d_v)
      weights: 注意力权重  shape=(seq_len, seq_len)
    """
    d_k = Q.shape[-1]  # Key 的维度

    # ── 步骤1：计算注意力分数 ─────────────────────────────────────────
    # scores[i,j] = 第i个词（作为Query）和第j个词（作为Key）的相似度
    scores = Q @ K.T   # shape=(seq_len, seq_len)

    if verbose:
        print(f"\n  步骤1：计算 Q·K^T 原始分数（shape={scores.shape}）")
        print("  每个格子 = 一对词之间的相似度得分")
        _print_matrix(scores, "原始分数")

    # ── 步骤2：缩放（除以 √d_k）─────────────────────────────────────
    # 防止 d_k 较大时点积值过大，导致 softmax 梯度消失
    scaled_scores = scores / math.sqrt(d_k)

    if verbose:
        print(f"\n  步骤2：除以 √{d_k} = {math.sqrt(d_k):.2f}（缩放）")
        _print_matrix(scaled_scores, "缩放后分数")

    # ── 步骤3：应用 Mask（可选，解码器用）─────────────────────────────
    if mask is not None:
        # 将被遮蔽位置设为极小值，softmax后近似为0
        scaled_scores = scaled_scores + mask * (-1e9)
        if verbose:
            print(f"\n  步骤3：应用掩码（屏蔽未来位置）")
            _print_matrix(scaled_scores, "Mask后分数")
    elif verbose:
        print(f"\n  步骤3：无掩码（编码器自注意力）")

    # ── 步骤4：Softmax + 加权求和 ─────────────────────────────────────
    # 对每行做 softmax，得到该词对所有词的注意力权重
    weights = np.array([_softmax(row) for row in scaled_scores])

    if verbose:
        print(f"\n  步骤4a：Softmax → 注意力权重（每行总和=1）")
        _print_matrix(weights, "注意力权重", fmt=".3f")

    # 加权求和：output[i] = Σ_j (weights[i,j] × V[j])
    output = weights @ V

    if verbose:
        print(f"\n  步骤4b：weights · V → 最终输出（shape={output.shape}）")
        _print_matrix(output, "注意力输出")

    return output, weights


def _softmax(x):
    """数值稳定版 softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _print_matrix(matrix, title, fmt=".2f"):
    """用 ASCII 可视化矩阵"""
    print(f"  {title}:")
    for row in matrix:
        vals = "  ".join(f"{v:{fmt}}" for v in row)
        print(f"    [{vals}]")


# ══════════════════════════════════════════════════════════════════════════════
# 第四部分：完整演示 —— 句子"I love Paris"的自注意力
# ══════════════════════════════════════════════════════════════════════════════

def demo_full_self_attention():
    """
    用一个简单例子演示完整的自注意力计算流程
    """
    print("\n\n" + "=" * 60)
    print("  完整演示：'I love Paris' 的自注意力计算")
    print("=" * 60)

    # ── 准备输入 ─────────────────────────────────────────────────────
    # 3个词，每个词用4维向量表示（实际应用中是768维）
    words = ["I", "love", "Paris"]
    d_model = 4    # 词向量维度
    d_k = 4        # 注意力头的键/查询维度

    print(f"\n  输入句子：{' '.join(words)}")
    print(f"  词向量维度：d_model = {d_model}")
    print(f"  注意力维度：d_k = {d_k}")

    # 词嵌入矩阵（实际中通过训练学习，这里随机初始化）
    X = np.random.randn(len(words), d_model)   # shape=(3, 4)
    print(f"\n  词嵌入矩阵 X（shape={X.shape}）：")
    for i, word in enumerate(words):
        print(f"    {word:6s}: {np.round(X[i], 3)}")

    # ── 生成 Q/K/V 权重矩阵 ─────────────────────────────────────────
    # 三个可训练权重矩阵，通过训练学习
    W_Q = np.random.randn(d_model, d_k)
    W_K = np.random.randn(d_model, d_k)
    W_V = np.random.randn(d_model, d_k)

    print(f"\n  Q/K/V 投影矩阵（可训练权重，shape each={W_Q.shape}）：")
    print(f"  W_Q、W_K、W_V 在训练中学习，决定如何提取'关注什么'")

    # ── 计算 Q/K/V ─────────────────────────────────────────────────
    Q = X @ W_Q   # shape=(3, 4)：每个词的"查询向量"
    K = X @ W_K   # shape=(3, 4)：每个词的"键向量"
    V = X @ W_V   # shape=(3, 4)：每个词的"值向量"

    print(f"\n  Q = X · W_Q  (shape={Q.shape})：每个词的'查询'")
    print(f"  K = X · W_K  (shape={K.shape})：每个词的'键'")
    print(f"  V = X · W_V  (shape={V.shape})：每个词的'值'")

    # ── 执行自注意力 ─────────────────────────────────────────────────
    print(f"\n  {'='*50}")
    print(f"  开始执行缩放点积注意力...")
    print(f"  {'='*50}")

    output, weights = scaled_dot_product_attention(Q, K, V, verbose=True)

    # ── 可视化注意力权重 ─────────────────────────────────────────────
    print(f"\n  【注意力热力图可视化】")
    print(f"  （行=Query词，列=Key词，值=注意力权重）")
    print()
    print(f"  {'':8s}", end="")
    for w in words:
        print(f"  {w:>8s}", end="")
    print()
    print("  " + "-" * 40)
    for i, row_word in enumerate(words):
        print(f"  {row_word:8s}", end="")
        for j, weight in enumerate(weights[i]):
            # 用颜色深度表示权重大小
            intensity = int(weight * 9)
            chars = [" ", "░", "░", "▒", "▒", "▓", "▓", "█", "█", "█"]
            cell = f"  {chars[intensity]}{weight:.3f}"
            print(cell, end="")
        print()

    print()
    print(f"  输出矩阵 shape = {output.shape}")
    print(f"  每个词都获得了融合上下文信息的新表示（不再是孤立的词向量）")


# ══════════════════════════════════════════════════════════════════════════════
# 第五部分：公式总结
# ══════════════════════════════════════════════════════════════════════════════

def formula_summary():
    print("\n\n" + "=" * 60)
    print("  核心公式总结")
    print("=" * 60)
    print("""
  缩放点积注意力公式：

    Attention(Q, K, V) = softmax( Q · K^T / √d_k ) · V

  逐步分解：
    ┌─────────────────────────────────────────────────────┐
    │  1. 相似度分数：  score = Q · K^T                  │
    │     → 衡量每对词之间的"相关程度"                   │
    │                                                     │
    │  2. 缩放：        scaled = score / √d_k             │
    │     → 防止维度过大导致 softmax 梯度消失             │
    │                                                     │
    │  3. 归一化：      weights = softmax(scaled)         │
    │     → 将分数变成概率，每行总和 = 1                  │
    │                                                     │
    │  4. 加权求和：    output = weights · V              │
    │     → 用注意力权重综合所有词的"值"信息              │
    └─────────────────────────────────────────────────────┘

  维度规律（以单头注意力为例）：
    X 输入：     (seq_len, d_model)   = (句子长度, 词向量维度)
    W_Q/W_K/W_V：(d_model, d_k)      = (词向量维度, 注意力维度)
    Q/K/V：      (seq_len, d_k)       = (句子长度, 注意力维度)
    scores：     (seq_len, seq_len)   = 每对词的相似度
    weights：    (seq_len, seq_len)   = 每对词的注意力权重
    output：     (seq_len, d_k)       = 融合上下文后的新表示
  """)


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Transformer 第03课：自注意力数学推导（NumPy实现）")
    print("=" * 60)

    numpy_quick_intro()
    explain_softmax()
    demo_full_self_attention()
    formula_summary()

    print("\n" + "=" * 60)
    print("  本课小结：")
    print("  · 向量点积 = 相似度度量")
    print("  · Q·K^T / √d_k → softmax → weights")
    print("  · output = weights · V（加权合并值向量）")
    print("  · 一次矩阵运算就能处理整个句子（并行！）")
    print()
    print("  下一课：多头注意力 + 位置编码 (04_multihead_positional.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
