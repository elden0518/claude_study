"""
主题：注意力机制直觉理解 —— 无需数学，只需类比

学习目标：
  1. 用"搜索引擎"类比理解 Query / Key / Value 三个概念
  2. 理解"自注意力"：一个词如何决定关注句子中的其他词
  3. 区分硬注意力（只选一个）和软注意力（加权综合）
  4. 直观感受注意力如何解决指代消解（"它"指什么？）

前置知识：
  - Python 基础（字典、列表、循环）
  - 完成 01_why_transformer.py

课程顺序：这是 06_transformer 模块的第 2 个文件。
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import math

# ══════════════════════════════════════════════════════════════════════════════
# 第一部分：类比1 —— 注意力就像"搜索引擎"
# ══════════════════════════════════════════════════════════════════════════════

def demo_search_engine_analogy():
    """
    注意力机制最直观的类比：数据库检索

    Query（查询）：你想搜索的内容
    Key  （索引）：数据库中每条记录的标题/标签
    Value（值）  ：实际存储的内容

    过程：
      1. 用 Query 和每个 Key 计算"相关度得分"
      2. 对得分做 softmax，变成"注意力权重"（加起来=1）
      3. 用权重对所有 Value 加权求和，得到最终输出
    """
    print("=" * 60)
    print("  类比1：注意力 = 软性数据库检索")
    print("=" * 60)

    # 模拟一个关于"动物"的知识库
    database = {
        "猫":    "小型家养动物，善于捕鼠，喜欢被抚摸",
        "狗":    "忠实的家养动物，可以训练，善于陪伴",
        "老虎":  "大型猛兽，生活在丛林，是猫科动物",
        "鸽子":  "鸟类，象征和平，可以传信",
        "金鱼":  "观赏鱼类，通常养在鱼缸里",
    }

    # Key = 知识库中每条记录的标签（这里就是动物名称）
    keys   = list(database.keys())
    # Value = 实际内容
    values = list(database.values())

    print("\n  【知识库内容】")
    for k, v in database.items():
        print(f"    Key: {k:4s}  →  Value: {v}")

    # 用户提出 Query
    query = "我想了解猫科动物"
    print(f"\n  【用户 Query】：{query}")

    # 手动定义相关度（实际中由向量点积计算）
    # 这里用数字模拟：猫=0.8，老虎=0.9，其他=0.1
    raw_scores = {"猫": 0.8, "狗": 0.1, "老虎": 0.9, "鸽子": 0.05, "金鱼": 0.05}

    # Softmax：将原始分数转换为概率（加起来=1）
    exp_scores = {k: math.exp(v * 5) for k, v in raw_scores.items()}
    total = sum(exp_scores.values())
    weights = {k: v / total for k, v in exp_scores.items()}

    print("\n  【注意力权重计算】（原始分数 → softmax → 权重）")
    print(f"  {'Key':>5s}  {'原始分数':>8s}  {'注意力权重':>10s}  {'注意力图'}")
    print("  " + "-" * 50)
    for k in keys:
        bar = "█" * int(weights[k] * 30)
        print(f"  {k:>5s}  {raw_scores[k]:>8.2f}  {weights[k]:>10.3f}  {bar}")

    print(f"\n  权重之和 = {sum(weights.values()):.3f}（总和为1，这是softmax的特性）")

    # 加权合并：最终输出 = Σ(weight_i × value_i)
    print("\n  【最终输出】：加权合并所有 Value")
    print("  '综合考虑各动物的相关性，重点参考猫和老虎的信息'")
    print("  → 这就是'软注意力'：不只选一个，而是按权重综合所有信息")


# ══════════════════════════════════════════════════════════════════════════════
# 第二部分：类比2 —— 自注意力就像"班级里的相互了解"
# ══════════════════════════════════════════════════════════════════════════════

def demo_self_attention_class_analogy():
    """
    自注意力（Self-Attention）：句子中每个词关注句子中的其他词

    比喻：一个班级里，每个同学在处理一道题时，
    会根据题目决定应该向哪些同学请教
    """
    print("\n\n" + "=" * 60)
    print("  类比2：自注意力 = 句子中词与词的相互关联")
    print("=" * 60)

    sentence = ["银行", "在", "河边", "取钱", "的", "人", "很多"]
    # "银行"这个词的含义：
    # - 如果周围有"取钱"、"储蓄"，则是"金融机构"
    # - 如果周围有"河边"、"水"，则是"河岸"

    print(f"\n  句子：{'  '.join(sentence)}")
    print(f"\n  问题：'银行'这个词到底是什么意思？")
    print(f"  → 答案取决于它和哪些词'注意力权重'更高！")

    # 模拟"银行"这个词对句子中其他词的注意力权重
    attention_weights = {
        "银行":  0.05,   # 自己对自己（有一定权重）
        "在":    0.03,
        "河边":  0.45,   # 关键词：说明是河岸而非银行
        "取钱":  0.30,   # 关键词：说明可能是金融机构（矛盾？）
        "的":    0.02,
        "人":    0.10,
        "很多":  0.05,
    }

    print(f"\n  【'银行'对各词的注意力权重】")
    print(f"  (权重越高 = 越需要参考该词来理解'银行'的含义)")
    print()
    for word, weight in attention_weights.items():
        bar = "▓" * int(weight * 40)
        marker = " ← 重要线索！" if weight > 0.2 else ""
        print(f"    {word:4s}  {weight:.2f}  {bar}{marker}")

    print()
    print("  → 由于'河边'权重最高，模型判断'银行'= 河岸（bank of river）")
    print("  → 这就是 Transformer 如何理解多义词的上下文含义！")

    print()
    print("  【与 RNN 的对比】")
    print("  RNN：'银行' 只能等读到'河边'后，通过隐藏状态间接了解语境")
    print("       距离越远，信息越弱")
    print("  Transformer：'银行' 可以直接和'河边'计算注意力，一步到位！")


# ══════════════════════════════════════════════════════════════════════════════
# 第三部分：硬注意力 vs 软注意力
# ══════════════════════════════════════════════════════════════════════════════

def demo_hard_vs_soft_attention():
    """
    硬注意力（Hard Attention）：只选一个，非0即1
    软注意力（Soft Attention）：加权综合，每个都有贡献

    Transformer 使用软注意力 —— 更可微分，可以用梯度下降训练
    """
    print("\n\n" + "=" * 60)
    print("  硬注意力 vs 软注意力")
    print("=" * 60)

    context_words = ["apple", "phone", "laptop", "fruit", "screen"]
    query_word = "iPhone"

    # 模拟相关度分数
    scores = {"apple": 0.85, "phone": 0.90, "laptop": 0.50,
              "fruit": 0.20, "screen": 0.60}

    print(f"\n  Query 词：{query_word}")
    print(f"  上下文：{context_words}")
    print()

    # 硬注意力：只选得分最高的一个
    best_word = max(scores, key=scores.get)
    hard_attn = {w: (1 if w == best_word else 0) for w in scores}

    print("  【硬注意力 Hard Attention】—— 只选最相关的一个")
    for w, v in hard_attn.items():
        bar = "█" * (20 if v == 1 else 0)
        print(f"    {w:8s}: {v}  {bar}")
    print(f"  → 结论：只用 '{best_word}' 来理解 '{query_word}'")
    print("  → 问题：丢失了 'phone'、'laptop'、'screen' 的信息！")

    # 软注意力：加权综合（Transformer 使用这种方式）
    exp_scores = {k: math.exp(v * 3) for k, v in scores.items()}
    total = sum(exp_scores.values())
    soft_attn = {k: v / total for k, v in exp_scores.items()}

    print()
    print("  【软注意力 Soft Attention】—— 加权综合所有词")
    for w, v in soft_attn.items():
        bar = "▓" * int(v * 30)
        print(f"    {w:8s}: {v:.3f}  {bar}")
    print(f"  → 结论：综合参考所有词，但重点关注 'phone'({soft_attn['phone']:.2f}) 和 'apple'({soft_attn['apple']:.2f})")
    print("  → 优势：保留了所有信息，且对训练更友好（可微分）")


# ══════════════════════════════════════════════════════════════════════════════
# 第四部分：Q/K/V 命名的由来
# ══════════════════════════════════════════════════════════════════════════════

def explain_qkv_naming():
    """
    用 Python 字典演示 Query / Key / Value 命名的来源
    """
    print("\n\n" + "=" * 60)
    print("  Q / K / V 命名来源：数据库检索的抽象")
    print("=" * 60)

    print("""
  数据库类比：
  ┌─────────────────────────────────────────────────────────┐
  │  Python dict（硬检索）：                                 │
  │    db = {"猫": "小型动物", "狗": "忠实伙伴"}             │
  │    result = db["猫"]  # 精确匹配，返回"小型动物"          │
  │                                                         │
  │  注意力机制（软检索）：                                  │
  │    Q = query_vector   # 查询向量（想找什么）             │
  │    K = key_matrix     # 所有词的键向量（每条记录的标签）  │
  │    V = value_matrix   # 所有词的值向量（实际内容）        │
  │                                                         │
  │    scores  = Q × K^T        # 计算 Q 和每个 K 的相似度   │
  │    weights = softmax(scores) # 变成概率分布               │
  │    output  = weights × V    # 加权组合所有 Value          │
  └─────────────────────────────────────────────────────────┘

  关键洞察：
  · 在"自"注意力中，Q、K、V 都来自同一个句子！
    → 句子中每个词问："我应该最关注句子中的哪些词？"

  · 在"交叉"注意力（解码器中）：
    Q 来自解码器（当前生成的词）
    K、V 来自编码器（原始输入的信息）
    → "生成每个词时，应该关注原始句子的哪个部分？"
  """)

    print("  Q/K/V 的尺寸（以 GPT-2 small 为例）：")
    print(f"    d_model（词向量维度）= 768")
    print(f"    每个注意力头的维度  = 768 / 12 = 64")
    print(f"    Q 矩阵 shape = (序列长度, 64)")
    print(f"    K 矩阵 shape = (序列长度, 64)")
    print(f"    V 矩阵 shape = (序列长度, 64)")
    print(f"  下一课将用 NumPy 实际演示这个计算过程！")


# ══════════════════════════════════════════════════════════════════════════════
# 第五部分：注意力如何解决指代消解
# ══════════════════════════════════════════════════════════════════════════════

def demo_coreference_resolution():
    """
    经典 NLP 难题：指代消解（"它"指的是什么？）
    这是展示注意力机制威力的绝佳例子
    """
    print("\n\n" + "=" * 60)
    print("  注意力的应用：指代消解")
    print("=" * 60)

    examples = [
        {
            "sentence": "法官惩罚了罪犯，因为 他 犯了罪。",
            "pronoun": "他",
            "answer": "罪犯",
            "attention": {"法官": 0.10, "惩罚": 0.05, "罪犯": 0.75, "犯": 0.10},
        },
        {
            "sentence": "法官惩罚了罪犯，因为 他 感到同情。",
            "pronoun": "他",
            "answer": "法官",
            "attention": {"法官": 0.72, "惩罚": 0.08, "罪犯": 0.12, "感到": 0.08},
        },
        {
            "sentence": "动物园拒绝了虎斑猫，因为 它 太凶猛了。",
            "pronoun": "它",
            "answer": "虎斑猫",
            "attention": {"动物园": 0.05, "拒绝": 0.05, "虎斑猫": 0.85, "凶猛": 0.05},
        },
    ]

    for ex in examples:
        print(f"\n  句子：{ex['sentence']}")
        print(f"  问题：'{ex['pronoun']}' 指的是谁？")
        print(f"  答案：'{ex['answer']}'")
        print(f"  注意力权重（模拟）：")
        for word, weight in ex["attention"].items():
            bar = "▓" * int(weight * 20)
            marker = " ← 高权重！" if weight == max(ex["attention"].values()) else ""
            print(f"    {word:6s}: {weight:.2f} {bar}{marker}")

    print()
    print("  → Transformer 通过注意力权重，让'它/他'直接关注最相关的名词")
    print("  → 不同语境下，同一个代词会产生完全不同的注意力分布")
    print("  → 这是 Transformer 理解语言语义的核心能力之一")


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Transformer 第02课：注意力机制直觉理解")
    print("=" * 60)

    demo_search_engine_analogy()
    demo_self_attention_class_analogy()
    demo_hard_vs_soft_attention()
    explain_qkv_naming()
    demo_coreference_resolution()

    print("\n" + "=" * 60)
    print("  本课小结：")
    print("  · 注意力 = 软性数据库检索（Q查询，K索引，V内容）")
    print("  · 自注意力：每个词直接关注句子中所有其他词")
    print("  · 软注意力（Transformer用）：加权综合，可微分可训练")
    print("  · 注意力能理解上下文，解决多义词、指代消解等问题")
    print()
    print("  下一课：自注意力的数学推导 —— 用NumPy实现Q/K/V计算")
    print("          (03_self_attention_math.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
