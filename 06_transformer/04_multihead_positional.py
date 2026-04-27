"""
主题：多头注意力 + 位置编码 —— Transformer 的两个关键创新

学习目标：
  1. 理解为什么需要多个注意力头（多维语义）
  2. 掌握多头注意力的计算流程（分头→并行→拼接→投影）
  3. 理解为什么 Transformer 需要位置编码（它天然无位置感知）
  4. 了解正弦/余弦位置编码的原理和特性

前置知识：
  - 完成 01/02/03 课
  - 掌握 NumPy 基础（向量/矩阵）

依赖：pip install numpy
课程顺序：这是 06_transformer 模块的第 4 个文件。
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
# 第一部分：为什么需要多头注意力？
# ══════════════════════════════════════════════════════════════════════════════

def explain_why_multihead():
    """
    单头注意力的局限：只能在一个语义空间中计算相似度
    多头注意力：在多个语义子空间中并行计算，捕捉不同维度的关系
    """
    print("=" * 60)
    print("  为什么需要多头注意力？")
    print("=" * 60)

    example = "The cat sat on the mat because it was tired."

    print(f"\n  例句：{example}")
    print(f"  代词 'it' 需要解析为 'cat'（不是 'mat'）")
    print()
    print("  不同的注意力头会关注不同类型的关系：")
    print()

    heads_explanation = [
        ("头1（句法依存）",  "it",  {"cat": 0.7, "sat": 0.1, "mat": 0.1, "tired": 0.1},
         "关注句法主语关系，识别'it'=主语，对应'cat'也是主语"),
        ("头2（语义相关）",  "it",  {"cat": 0.5, "mat": 0.3, "sat": 0.1, "tired": 0.1},
         "关注语义共现，动物更可能疲惫"),
        ("头3（时序/因果）", "it",  {"tired": 0.5, "because": 0.3, "cat": 0.1, "mat": 0.1},
         "关注因果连词'because'周围的关系"),
        ("头4（位置相邻）",  "it",  {"was": 0.6, "because": 0.2, "cat": 0.1, "mat": 0.1},
         "关注局部位置相邻性"),
    ]

    for head_name, query_word, attn, desc in heads_explanation:
        print(f"  【{head_name}】")
        print(f"  分析词：'{query_word}' | 说明：{desc}")
        for word, weight in attn.items():
            bar = "▓" * int(weight * 20)
            print(f"    {word:8s}: {weight:.1f}  {bar}")
        print()

    print("  → 多头注意力将多个头的结果拼接，获得更丰富的表示")
    print("  → 最终模型综合所有头的信息来做判断")


# ══════════════════════════════════════════════════════════════════════════════
# 第二部分：多头注意力的计算流程
# ══════════════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(Q, K, V):
    """单头注意力（复用第03课的实现）"""
    d_k = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d_k)
    # 按行做 softmax
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    return weights @ V, weights


class MultiHeadAttention:
    """
    多头注意力的完整实现

    流程：
      1. 用 h 组权重矩阵，将 Q/K/V 投影到 h 个低维子空间
      2. 在每个子空间中独立计算注意力（并行）
      3. 将 h 个结果拼接（concatenate）
      4. 再经过一个线性投影，得到最终输出
    """

    def __init__(self, d_model, num_heads):
        """
        d_model:   词向量的维度（如 GPT-2: 768, GPT-3: 12288）
        num_heads: 注意力头的数量（d_model 必须能被 num_heads 整除）
        """
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # 每个头的维度

        # 每个头有独立的 Q/K/V 投影矩阵
        # 实际实现中通常用一个大矩阵再切分（等价）
        self.W_Q = np.random.randn(d_model, d_model)  # 包含所有头的权重
        self.W_K = np.random.randn(d_model, d_model)
        self.W_V = np.random.randn(d_model, d_model)
        # 输出投影矩阵
        self.W_O = np.random.randn(d_model, d_model)

    def forward(self, X, verbose=False):
        """
        X: 输入矩阵  shape=(seq_len, d_model)
        返回：输出矩阵  shape=(seq_len, d_model)
        """
        seq_len = X.shape[0]

        if verbose:
            print(f"\n  输入 X shape = {X.shape}  (seq_len={seq_len}, d_model={self.d_model})")
            print(f"  注意力头数 = {self.num_heads}，每头维度 = {self.d_k}")

        # ── 步骤1：线性投影 ────────────────────────────────────────
        Q_all = X @ self.W_Q   # (seq_len, d_model)
        K_all = X @ self.W_K
        V_all = X @ self.W_V

        if verbose:
            print(f"\n  步骤1：线性投影")
            print(f"  Q_all shape = {Q_all.shape}（包含所有头的查询）")

        # ── 步骤2：拆分成多头 ───────────────────────────────────────
        # 将 d_model 维度拆成 (num_heads, d_k)
        # reshape: (seq_len, d_model) → (seq_len, num_heads, d_k)
        # transpose: → (num_heads, seq_len, d_k)
        def split_heads(X):
            return X.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        Q_heads = split_heads(Q_all)  # (num_heads, seq_len, d_k)
        K_heads = split_heads(K_all)
        V_heads = split_heads(V_all)

        if verbose:
            print(f"\n  步骤2：拆分为 {self.num_heads} 个注意力头")
            print(f"  每头 Q shape = {Q_heads[0].shape}")

        # ── 步骤3：每头独立计算注意力（可并行） ───────────────────
        head_outputs = []
        all_weights  = []

        for h in range(self.num_heads):
            output_h, weights_h = scaled_dot_product_attention(
                Q_heads[h], K_heads[h], V_heads[h]
            )
            head_outputs.append(output_h)
            all_weights.append(weights_h)

        if verbose:
            print(f"\n  步骤3：每头独立计算注意力")
            print(f"  每头输出 shape = {head_outputs[0].shape}")

        # ── 步骤4：拼接所有头的输出 ────────────────────────────────
        # (num_heads, seq_len, d_k) → (seq_len, d_model)
        concatenated = np.concatenate(head_outputs, axis=-1)
        # head_outputs 是 list，先 stack 再 transpose
        stacked = np.stack(head_outputs, axis=0)                # (num_heads, seq_len, d_k)
        transposed = stacked.transpose(1, 0, 2)                  # (seq_len, num_heads, d_k)
        concatenated = transposed.reshape(seq_len, self.d_model) # (seq_len, d_model)

        if verbose:
            print(f"\n  步骤4：拼接 {self.num_heads} 个头的输出")
            print(f"  拼接后 shape = {concatenated.shape}")

        # ── 步骤5：输出投影 ─────────────────────────────────────────
        output = concatenated @ self.W_O   # (seq_len, d_model)

        if verbose:
            print(f"\n  步骤5：输出线性投影")
            print(f"  最终输出 shape = {output.shape}")

        return output, all_weights


def demo_multihead_attention():
    """演示多头注意力的实际计算"""
    print("\n\n" + "=" * 60)
    print("  多头注意力计算演示")
    print("=" * 60)

    # 模型参数（GPT-2 small 规模的缩小版）
    d_model   = 8    # 词向量维度（GPT-2实际是768）
    num_heads = 2    # 注意力头数（GPT-2实际是12）
    seq_len   = 4    # 序列长度

    print(f"\n  配置：d_model={d_model}, num_heads={num_heads}, d_k={d_model//num_heads}")
    print(f"  输入：{seq_len} 个词")

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # 模拟输入："the cat sat on"
    words = ["the", "cat", "sat", "on"]
    X = np.random.randn(seq_len, d_model)

    output, all_weights = mha.forward(X, verbose=True)

    print(f"\n  所有 {num_heads} 个注意力头的权重分布：")
    for h, weights in enumerate(all_weights):
        print(f"\n  头 {h+1} 注意力权重（{seq_len}×{seq_len}）：")
        print(f"  {'':6s}", end="")
        for w in words:
            print(f"  {w:>6s}", end="")
        print()
        for i, row_word in enumerate(words):
            print(f"  {row_word:6s}", end="")
            for j in range(len(words)):
                intensity = int(weights[i][j] * 8)
                chars = [" ", "░", "▒", "▒", "▓", "▓", "█", "█", "█"]
                print(f"  {chars[intensity]}{weights[i][j]:.3f}", end="")
            print()

    print(f"\n  ✓ 完整输出 shape = {output.shape}，与输入维度相同")
    print(f"  ✓ 每个词都获得了融合了 {num_heads} 个不同语义视角的新表示")


# ══════════════════════════════════════════════════════════════════════════════
# 第三部分：位置编码 —— 告诉模型词的位置
# ══════════════════════════════════════════════════════════════════════════════

def explain_why_positional_encoding():
    """
    Transformer 的注意力机制本身对顺序无感知！
    "猫追狗" 和 "狗追猫" 在不加位置信息时，会得到相同的词嵌入和注意力
    """
    print("\n\n" + "=" * 60)
    print("  为什么需要位置编码？")
    print("=" * 60)

    print("""
  问题演示：
  句子1："猫  追  狗"
  句子2："狗  追  猫"

  如果不加位置信息：
    · "猫"的词向量在两个句子中完全相同
    · 注意力计算的结果（Q·K^T）完全相同
    · 模型无法区分这两句话！

  解决方案：给每个位置添加唯一的"位置信号"
  使得：词向量 = 语义向量 + 位置向量

  要求：
  1. 每个位置的编码必须唯一
  2. 不同位置的距离应该一致（位置1和2的距离 = 位置2和3的距离）
  3. 能推广到比训练时更长的序列
  4. 值域有界（不能无限增大）

  Transformer 的方案：正弦/余弦位置编码
  """)


def positional_encoding(max_seq_len, d_model):
    """
    计算正弦/余弦位置编码

    公式：
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    参数：
      max_seq_len: 最大序列长度
      d_model:     词向量维度

    返回：
      PE: shape=(max_seq_len, d_model) 的位置编码矩阵
    """
    PE = np.zeros((max_seq_len, d_model))

    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            # 分母：10000^(2i/d_model)
            denominator = 10000 ** (2 * i / d_model)

            # 偶数维：sin
            PE[pos, i] = math.sin(pos / denominator)

            # 奇数维：cos（如果存在）
            if i + 1 < d_model:
                PE[pos, i + 1] = math.cos(pos / denominator)

    return PE


def demo_positional_encoding():
    """演示位置编码的特性"""
    print("\n\n" + "=" * 60)
    print("  位置编码演示")
    print("=" * 60)

    d_model = 8
    seq_len = 6
    words   = ["<SOS>", "猫", "追", "着", "那只", "狗"]

    PE = positional_encoding(seq_len, d_model)

    print(f"\n  位置编码矩阵（{seq_len}个位置 × {d_model}维）：")
    print(f"  {'位置':>4s}  {'词':>6s}  {'位置编码向量（前8维）'}")
    print("  " + "-" * 65)
    for pos in range(seq_len):
        vals = " ".join(f"{v:+.3f}" for v in PE[pos])
        print(f"  pos={pos}  {words[pos]:6s}  [{vals}]")

    print()
    print("  正弦/余弦编码的关键特性：")

    # 特性1：每个位置的编码唯一
    print("\n  特性1：每个位置唯一（不同位置的向量不同）")
    for i in range(min(3, seq_len)):
        for j in range(i+1, min(4, seq_len)):
            diff = np.linalg.norm(PE[i] - PE[j])
            print(f"    pos={i} vs pos={j} 的差距（L2距离）= {diff:.3f}")

    # 特性2：相对位置编码可通过线性变换得到
    print("\n  特性2：可以通过点积计算位置相似性")
    print("  位置i和位置j的点积越大，表示它们在'位置上'越相关")
    print(f"\n  {'':6s}", end="")
    for i in range(seq_len):
        print(f"  pos{i}", end="")
    print()
    for i in range(seq_len):
        print(f"  pos{i}  ", end="")
        for j in range(seq_len):
            sim = PE[i] @ PE[j]
            intensity = min(8, max(0, int((sim + 1) * 4)))
            chars = ["░", "░", "▒", "▒", "▒", "▓", "▓", "█", "█"]
            print(f" {chars[intensity]}{sim:+.2f}", end="")
        print()

    print("\n  ✓ 对角线（自身）值最大，相邻位置相关性高于远距离位置")

    # 特性3：可视化不同频率的正弦波
    print("\n  特性3：不同维度使用不同频率（低维低频，高维高频）")
    print("  就像时钟：秒针（高频）、分针（中频）、时针（低频）")
    print()
    print("  维度0（最低频，变化最慢）：", end="")
    vals = [f"{PE[pos, 0]:+.2f}" for pos in range(seq_len)]
    print("  ".join(vals))

    print("  维度2（中等频率）：", end="")
    vals = [f"{PE[pos, 2]:+.2f}" for pos in range(seq_len)]
    print("  ".join(vals))

    if d_model > 4:
        print("  维度4（较高频率）：", end="")
        vals = [f"{PE[pos, 4]:+.2f}" for pos in range(seq_len)]
        print("  ".join(vals))


def demo_position_matters():
    """
    演示：加上位置编码后，模型能区分词序不同的句子
    """
    print("\n\n" + "=" * 60)
    print("  位置编码效果：区分'猫追狗'和'狗追猫'")
    print("=" * 60)

    d_model = 4

    # 假设词嵌入（两个句子的词完全相同，只是顺序不同）
    embed_cat  = np.array([0.8, 0.2, 0.1, 0.5])  # "猫"的词向量
    embed_chase= np.array([0.3, 0.7, 0.8, 0.2])  # "追"的词向量
    embed_dog  = np.array([0.7, 0.3, 0.1, 0.4])  # "狗"的词向量

    PE = positional_encoding(3, d_model)

    # 句子1：猫(pos0) 追(pos1) 狗(pos2)
    X1 = np.array([
        embed_cat   + PE[0],
        embed_chase + PE[1],
        embed_dog   + PE[2],
    ])

    # 句子2：狗(pos0) 追(pos1) 猫(pos2)
    X2 = np.array([
        embed_dog   + PE[0],
        embed_chase + PE[1],
        embed_cat   + PE[2],
    ])

    print("\n  句子1：猫(pos=0) 追(pos=1) 狗(pos=2)")
    print("  句子2：狗(pos=0) 追(pos=1) 猫(pos=2)")
    print()

    # 比较两个句子中"猫"的最终向量
    cat_in_s1 = X1[0]   # 猫在句子1的位置0
    cat_in_s2 = X2[2]   # 猫在句子2的位置2

    diff = np.linalg.norm(cat_in_s1 - cat_in_s2)
    print(f"  '猫' + 位置0 的向量：{np.round(cat_in_s1, 3)}")
    print(f"  '猫' + 位置2 的向量：{np.round(cat_in_s2, 3)}")
    print(f"  两者差距（L2距离）：{diff:.4f}")
    print()
    print("  ✓ 即使是同一个词，在不同位置的向量也不同！")
    print("  ✓ 模型因此可以区分词序，理解'猫追狗'≠'狗追猫'")


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Transformer 第04课：多头注意力 + 位置编码")
    print("=" * 60)

    explain_why_multihead()
    demo_multihead_attention()
    explain_why_positional_encoding()
    demo_positional_encoding()
    demo_position_matters()

    print("\n" + "=" * 60)
    print("  本课小结：")
    print("  · 多头注意力 = h 个独立注意力头并行 → 拼接 → 投影")
    print("  · 每个头关注不同的语义维度（句法/语义/位置等）")
    print("  · 位置编码用 sin/cos 给每个位置一个唯一信号")
    print("  · 输入 = 词嵌入 + 位置编码（加法，而非拼接）")
    print()
    print("  下一课：完整的 Encoder-Decoder 架构")
    print("          (05_encoder_decoder_arch.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
