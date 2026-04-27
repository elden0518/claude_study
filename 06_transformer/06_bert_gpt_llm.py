"""
主题：BERT vs GPT vs 现代大模型 —— Transformer 的三条演化路线

学习目标：
  1. 理解 Encoder-Only、Decoder-Only、Encoder-Decoder 三种架构
  2. 掌握 BERT 的预训练方式（MLM + NSP）及其适用场景
  3. 掌握 GPT 的预训练方式（自回归语言模型）及其适用场景
  4. 了解 LLM 发展历程和 RLHF 对齐技术
  5. 理解 Claude 和 GPT-4 等现代模型的设计哲学

前置知识：
  - 完成 01-05 课
  - 理解 Encoder/Decoder 架构

课程顺序：这是 06_transformer 模块的第 6 个文件。
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ══════════════════════════════════════════════════════════════════════════════
# 第一部分：三条演化路线
# ══════════════════════════════════════════════════════════════════════════════

def explain_three_architectures():
    print("=" * 60)
    print("  Transformer 的三条演化路线")
    print("=" * 60)

    print("""
  原始 Transformer（2017）有完整的 Encoder + Decoder，用于翻译任务。
  后来，研究者发现可以单独使用其中一半，针对不同任务优化：

  ┌─────────────────────────────────────────────────────────────┐
  │  路线1：Encoder-Only（只用编码器）                          │
  │  代表：BERT, RoBERTa, ALBERT                               │
  │  特点：双向注意力（每个词能看到左右两侧的词）               │
  │  优点：对文本的"理解"能力强                                 │
  │  适合：文本分类、命名实体识别、问答、情感分析               │
  │  ────────────────────────────────────────────────────────  │
  │  路线2：Decoder-Only（只用解码器）                          │
  │  代表：GPT系列, Claude, LLaMA, Gemini                      │
  │  特点：单向注意力（每个词只能看左侧历史词，因果掩码）       │
  │  优点：对文本的"生成"能力强，可以做开放式对话               │
  │  适合：文本生成、对话、摘要、代码生成、推理                 │
  │  ────────────────────────────────────────────────────────  │
  │  路线3：Encoder-Decoder（保留完整结构）                    │
  │  代表：T5, BART, mBART                                     │
  │  特点：编码器理解输入，解码器生成输出                       │
  │  适合：翻译、摘要、问答（有明确输入→输出格式的任务）        │
  └─────────────────────────────────────────────────────────────┘
  """)


# ══════════════════════════════════════════════════════════════════════════════
# 第二部分：BERT —— 预训练-微调范式的奠基者
# ══════════════════════════════════════════════════════════════════════════════

def explain_bert():
    print("\n\n" + "=" * 60)
    print("  BERT：Bidirectional Encoder Representations from Transformers")
    print("=" * 60)

    print("""
  发布时间：2018年10月，Google
  架构：Encoder-Only，12层（base）或 24层（large）

  ── 核心创新：预训练 + 微调范式 ────────────────────────────

  阶段1：预训练（在海量无标注文本上自监督学习）
    任务1：遮蔽语言模型（MLM，Masked Language Model）
      · 随机遮蔽输入中 15% 的词（替换为 [MASK]）
      · 让模型预测被遮蔽的词
      · 例：
          输入：  "The [MASK] sat on the [MASK]"
          目标：  "The cat  sat on the mat"
      · 关键：双向上下文！'cat' 可以同时看左边和右边的词

    任务2：下一句预测（NSP，Next Sentence Prediction）
      · 给模型两段话，预测第二段是否是第一段的下一句
      · 50% 正例（真实的下一句），50% 负例（随机句子）
      · 帮助模型理解句间关系

  阶段2：微调（在特定任务的小标注数据集上fine-tune）
    · 在预训练好的 BERT 上添加一个简单的输出层
    · 用任务数据集微调所有参数（或只微调输出层）
    · 少量数据即可获得很好效果（迁移学习的威力）

  ── BERT 的局限 ─────────────────────────────────────────────
  · 双向设计使其无法做自回归生成（不能一个词一个词地生成）
  · 更适合"理解"任务，而非"生成"任务
  · 上下文窗口有限（512词）
  """)

    # 演示 BERT 的 [MASK] 预测
    print("  BERT MLM 预测演示（伪代码）：")
    examples = [
        ("I went to the [MASK] to buy some groceries.",  "store"),
        ("The sky is [MASK] today.",                     "blue / clear / overcast"),
        ("Python is a popular programming [MASK].",      "language"),
        ("The [MASK] of France is Paris.",               "capital"),
    ]
    print(f"\n  {'输入（含[MASK]）':45s}  → 预测词")
    print("  " + "-" * 65)
    for masked, prediction in examples:
        print(f"  {masked:45s}  → {prediction}")

    print()
    print("  关键：BERT 同时参考 [MASK] 左侧和右侧的词来预测！")
    print("       这就是'双向'的含义（Bidirectional）")


# ══════════════════════════════════════════════════════════════════════════════
# 第三部分：GPT —— 自回归生成的开创者
# ══════════════════════════════════════════════════════════════════════════════

def explain_gpt():
    print("\n\n" + "=" * 60)
    print("  GPT：Generative Pre-trained Transformer")
    print("=" * 60)

    print("""
  GPT-1 发布：2018年，OpenAI（与 BERT 同年！）
  架构：Decoder-Only（单向，使用因果掩码）

  ── 核心预训练任务：语言模型（Language Modeling）─────────────

  目标：给定前 n 个词，预测第 n+1 个词
  公式：P(词n+1 | 词1, 词2, ..., 词n)

  例：
    已知："The weather is"
    GPT 预测下一个词的概率：
      nice:   0.25
      cold:   0.18
      hot:    0.15
      rainy:  0.12
      ...
    选取概率最高的词 "nice"，然后继续预测...

  ── GPT 系列的演进 ──────────────────────────────────────────

  GPT-1 (2018)：1.17亿参数，12层，证明了预训练+微调的价值
  GPT-2 (2019)：15亿参数，48层，"太危险而延迟发布"
  GPT-3 (2020)：1750亿参数，96层，少样本学习（few-shot）
  GPT-3.5    ：GPT-3 + RLHF，ChatGPT 的基础（2022年底爆火）
  GPT-4      ：多模态（可以理解图片），推理能力大幅提升

  ── GPT 的关键能力 ──────────────────────────────────────────
  · 涌现能力（Emergent Abilities）：
    模型规模达到某个阈值后，突然具备了训练时没有显式教过的能力
    例：GPT-3 没有专门训练算术，但可以做简单数学题

  · 少样本学习（Few-shot Learning / In-context Learning）：
    不需要微调，只需在提示词中给几个例子，模型就能理解并完成任务
    例：
      英文→法文翻译：
      "cat" → "chat"
      "dog" → "chien"
      "bird" → ?    → GPT 输出 "oiseau" ✓
  """)


# ══════════════════════════════════════════════════════════════════════════════
# 第四部分：从 GPT-3 到 ChatGPT —— RLHF 对齐技术
# ══════════════════════════════════════════════════════════════════════════════

def explain_rlhf():
    print("\n\n" + "=" * 60)
    print("  RLHF：让 LLM 听人话 —— ChatGPT 诞生的关键")
    print("=" * 60)

    print("""
  问题：GPT-3 虽然强大，但会生成有害内容、答非所问
  原因：预训练目标只是"预测下一个词"，没有学习"如何有用地对话"

  RLHF（Reinforcement Learning from Human Feedback）解决了这个问题：

  ── 三个阶段 ─────────────────────────────────────────────────

  阶段1：有监督微调（SFT，Supervised Fine-Tuning）
    · 收集人类写的高质量对话示例
    · 用这些示例微调预训练的 GPT-3
    · 结果：模型学会了对话的基本格式

  阶段2：训练奖励模型（Reward Model）
    · 让模型对同一个问题生成多个回答
    · 人工标注员对这些回答排序（哪个更好）
    · 训练一个评分模型（奖励模型），预测人类的偏好

  阶段3：PPO 强化学习（Proximal Policy Optimization）
    · 用奖励模型给生成的回答打分
    · 用 PPO 算法更新语言模型，使其生成分数更高的回答
    · 同时加入 KL 散度惩罚，防止模型"走极端"

  视觉化流程：
  ┌─────────────────────────────────────────────────────────┐
  │  用户问题 → [GPT-3.5] → 多个候选回答                   │
  │                               ↓                        │
  │  [人工排序] → [奖励模型训练] → 奖励模型                  │
  │                               ↓                        │
  │  [PPO强化学习] → 更新 GPT-3.5 → 更好的对话模型          │
  │                 (循环多次)                              │
  └─────────────────────────────────────────────────────────┘

  效果：ChatGPT 比 GPT-3 更有帮助、更安全、更诚实

  Claude 的对齐方法：Anthropic 开发了 Constitutional AI（宪法AI）
    · 给模型一组"宪法原则"（如"避免有害内容"、"诚实"等）
    · 让模型用这些原则自我批评和修正回答
    · 减少对大量人工标注的依赖
  """)


# ══════════════════════════════════════════════════════════════════════════════
# 第五部分：主流模型对比
# ══════════════════════════════════════════════════════════════════════════════

def compare_modern_llms():
    print("\n\n" + "=" * 60)
    print("  主流大语言模型对比（2025年）")
    print("=" * 60)

    print("""
  ┌──────────────┬──────────┬──────────┬─────────────────────────────┐
  │  模型         │  开发者  │  架构    │  特点                       │
  ├──────────────┼──────────┼──────────┼─────────────────────────────┤
  │ GPT-4o       │ OpenAI   │ Dec-only │ 多模态，推理强，广泛集成    │
  │ Claude 3.5+  │Anthropic │ Dec-only │ 长上下文，安全性，代码能力  │
  │ Gemini 1.5   │ Google   │ Dec-only │ 超长上下文(1M tokens)       │
  │ LLaMA 3      │ Meta     │ Dec-only │ 开源，可本地部署            │
  │ Qwen         │ Alibaba  │ Dec-only │ 中文能力强，开源版本        │
  │ DeepSeek R1  │DeepSeek  │ Dec-only │ 开源推理模型，性价比高      │
  │ BERT系列      │ Google   │ Enc-only │ 理解任务，文本分类等        │
  │ T5/BART      │ Google   │ Enc-Dec  │ 翻译、摘要等结构化任务      │
  └──────────────┴──────────┴──────────┴─────────────────────────────┘

  你当前使用的 Claude（claude-sonnet-4-6）就是 Decoder-Only 架构！
  """)

    print("  BERT vs GPT 核心区别总结：")
    print()
    comparison = [
        ("维度",         "BERT（Encoder-Only）",       "GPT（Decoder-Only）"),
        ("注意力方向",   "双向（看全句）",             "单向（只看历史）"),
        ("预训练任务",   "预测被遮蔽的词（MLM）",      "预测下一个词（LM）"),
        ("强项",         "文本理解",                   "文本生成"),
        ("适合任务",     "分类、NER、问答（抽取式）",  "对话、生成、推理"),
        ("上下文",       "双向，理解更准确",           "单向，但可以无限生成"),
        ("缩放规律",     "规模提升理解能力",           "规模提升涌现能力"),
        ("代表模型",     "BERT, RoBERTa, ALBERT",      "GPT系列, Claude, LLaMA"),
    ]

    print(f"  {'维度':12s}  {'BERT（Encoder-Only）':24s}  {'GPT（Decoder-Only）'}")
    print("  " + "-" * 70)
    for dim, bert_val, gpt_val in comparison[1:]:
        print(f"  {dim:12s}  {bert_val:24s}  {gpt_val}")


# ══════════════════════════════════════════════════════════════════════════════
# 第六部分：现代 LLM 的关键特性
# ══════════════════════════════════════════════════════════════════════════════

def explain_modern_llm_features():
    print("\n\n" + "=" * 60)
    print("  现代 LLM 的关键特性")
    print("=" * 60)

    features = {
        "涌现能力（Emergent Abilities）": {
            "描述": "模型规模增大后突然出现的能力，无法从小模型外推",
            "例子": "数学推理、代码生成、多步逻辑推理",
            "发现": "模型参数 < 10B 时几乎没有，> 100B 后突然爆发",
        },
        "上下文学习（In-context Learning）": {
            "描述": "在提示词中给几个例子，模型无需微调就能学会新任务",
            "例子": "zero-shot（无例子）、few-shot（几个例子）",
            "发现": "GPT-3 首次展示，GPT-4 和 Claude 更强",
        },
        "思维链（Chain-of-Thought, CoT）": {
            "描述": "让模型一步步推理，而不是直接给出答案",
            "例子": "在提示中加入'Let me think step by step'可提升准确率",
            "发现": "复杂推理任务中效果显著",
        },
        "长上下文（Long Context）": {
            "描述": "处理越来越长的输入，从1K tokens到1M tokens",
            "例子": "Claude 3支持200K tokens（约15万词，一本小说）",
            "发现": "使得全文档分析、长代码库理解成为可能",
        },
        "多模态（Multimodal）": {
            "描述": "同时处理文字、图片、音频、视频等多种输入",
            "例子": "GPT-4V、Claude 3可以分析图表、截图、照片",
            "发现": "Vision Transformer(ViT)将图片分块处理，类似词的序列",
        },
    }

    for feature, info in features.items():
        print(f"\n  【{feature}】")
        print(f"  描述：{info['描述']}")
        print(f"  例子：{info['例子']}")
        print(f"  现状：{info['发现']}")


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Transformer 第06课：BERT vs GPT vs 现代大模型")
    print("=" * 60)

    explain_three_architectures()
    explain_bert()
    explain_gpt()
    explain_rlhf()
    compare_modern_llms()
    explain_modern_llm_features()

    print("\n" + "=" * 60)
    print("  本课小结：")
    print("  · Encoder-Only (BERT)：理解任务，双向注意力")
    print("  · Decoder-Only (GPT/Claude)：生成任务，单向注意力")
    print("  · Encoder-Decoder (T5)：翻译/摘要等输入→输出任务")
    print("  · RLHF 让 LLM 从'补全机器'变为'有帮助的助手'")
    print("  · 涌现能力、长上下文、多模态是现代LLM的核心特性")
    print()
    print("  下一课：HuggingFace 实战 (07_huggingface_practice.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
