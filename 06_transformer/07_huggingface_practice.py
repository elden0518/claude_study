"""
主题：HuggingFace Transformers 实战 —— 5分钟调用预训练模型

学习目标：
  1. 了解 HuggingFace 生态系统（Hub、transformers、datasets）
  2. 掌握 Pipeline API：一行代码完成 NLP 任务
  3. 理解 Tokenizer 的工作原理（分词、编码、解码）
  4. 学会加载和使用预训练模型（BERT/GPT-2）
  5. 了解中国大陆访问 HuggingFace 的镜像方案

前置知识：
  - 完成 01-06 课
  - Python 基础

依赖：pip install transformers
      （如果下载慢，请先设置镜像：见本文件第一部分）

注意事项：
  · 首次运行会自动下载模型文件（几百MB到几GB）
  · 如果网络受限，请先配置 HF_ENDPOINT 镜像
  · 模型下载到 ~/.cache/huggingface/ 目录

课程顺序：这是 06_transformer 模块的第 7 个文件。
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os

# ══════════════════════════════════════════════════════════════════════════════
# 第一部分：HuggingFace 镜像配置（中国大陆必看）
# ══════════════════════════════════════════════════════════════════════════════

def setup_hf_mirror():
    """
    配置 HuggingFace 镜像，解决中国大陆下载慢或失败的问题

    官方镜像：https://huggingface.co（可能需要代理）
    国内镜像：https://hf-mirror.com（推荐）
    """
    # 方法1：代码中设置（本脚本使用这种方式）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    print("=" * 60)
    print("  HuggingFace 镜像配置")
    print("=" * 60)
    print(f"  已设置 HF_ENDPOINT = {os.environ.get('HF_ENDPOINT')}")
    print()
    print("  其他配置方式：")
    print("  方法2：在 .env 文件中添加：HF_ENDPOINT=https://hf-mirror.com")
    print("  方法3：命令行：set HF_ENDPOINT=https://hf-mirror.com (Windows)")
    print("         命令行：export HF_ENDPOINT=https://hf-mirror.com (Linux/Mac)")
    print()
    print("  模型缓存目录（首次下载后不需要重新下载）：")
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"  {cache}")


# ══════════════════════════════════════════════════════════════════════════════
# 第二部分：HuggingFace 生态系统介绍
# ══════════════════════════════════════════════════════════════════════════════

def explain_hf_ecosystem():
    print("\n\n" + "=" * 60)
    print("  HuggingFace 生态系统")
    print("=" * 60)

    print("""
  HuggingFace（抱脸）是 NLP/AI 领域最重要的开源社区

  ── 主要产品 ─────────────────────────────────────────────────

  1. Hub（模型仓库）：https://huggingface.co
     · 35万+ 预训练模型（持续增加）
     · 7万+ 数据集
     · 类似 GitHub，任何人可以上传/下载模型

  2. transformers 库（Python）
     · pip install transformers
     · 统一的 API，支持 BERT、GPT、T5、LLaMA 等数百种模型
     · 自动下载和缓存模型权重

  3. datasets 库
     · pip install datasets
     · 快速加载 NLP 数据集

  4. PEFT 库（参数高效微调）
     · pip install peft
     · LoRA、Adapter 等微调方法，少量 GPU 即可微调大模型

  ── Pipeline API ──────────────────────────────────────────────
  最简单的调用方式：

  from transformers import pipeline

  # 情感分析（自动下载模型）
  classifier = pipeline("sentiment-analysis")
  result = classifier("I love this movie!")
  # → [{'label': 'POSITIVE', 'score': 0.9998}]

  # 文本生成
  generator = pipeline("text-generation", model="gpt2")
  result = generator("Once upon a time", max_length=50)
  # → [{'generated_text': 'Once upon a time...'}]
  """)


# ══════════════════════════════════════════════════════════════════════════════
# 第三部分：Tokenizer 详解
# ══════════════════════════════════════════════════════════════════════════════

def explain_tokenizer():
    """
    Tokenizer（分词器）：将文本转换为模型可理解的数字序列
    这是使用任何 Transformer 模型的第一步
    """
    print("\n\n" + "=" * 60)
    print("  Tokenizer：文本 → 数字的翻译器")
    print("=" * 60)

    print("""
  为什么需要 Tokenizer？
  · 神经网络只能处理数字，不能直接处理文字
  · Tokenizer 将文本切分成"token"（词片段），再转为数字

  Token ≠ 词：
  · "playing" → ["play", "##ing"]（WordPiece分词）
  · "unbelievable" → ["un", "##believe", "##able"]
  · "ChatGPT" → ["Chat", "G", "PT"]（BPE分词）
  · 中文："我爱北京天安门" → ["我", "爱", "北", "京", "天", "安", "门"]

  常见特殊 Token：
  · [CLS]：句子开头（BERT用，表示整句的语义）
  · [SEP]：句子分隔符（BERT用）
  · [PAD]：填充符（让批次中的序列等长）
  · [MASK]：掩码（BERT预训练用）
  · <s>、</s>：句子开始/结束（部分模型用）
  """)

    # 用 Python 模拟简单的 BPE 分词过程
    print("  【模拟 BPE 分词过程】（Byte-Pair Encoding，GPT 使用）")
    print()

    # 简化版 BPE 演示
    vocab_demo = {
        "hello":        100,
        "world":        200,
        "hel":          300,
        "lo":           400,
        "##world":      500,
        "natural":      600,
        "language":     700,
        "processing":   800,
        "nat":          900,
        "##ural":       910,
        "lang":         920,
        "##uage":       930,
    }

    example_texts = [
        "hello world",
        "natural language processing",
    ]

    for text in example_texts:
        print(f"  文本：'{text}'")
        # 简单按空格切分 + 显示 ID
        words = text.split()
        tokens_shown = []
        for w in words:
            if w in vocab_demo:
                tokens_shown.append(f"'{w}'(id={vocab_demo[w]})")
            else:
                tokens_shown.append(f"'{w}'(id=?)")
        print(f"  Tokens：{' + '.join(tokens_shown)}")
        print(f"  Token IDs：{[vocab_demo.get(w, 0) for w in words]}")
        print()

    print("  真实示例（BERT tokenizer 输出）：")
    bert_examples = [
        ("I love NLP",        ["[CLS]", "I", "love", "NL", "##P", "[SEP]"],        [101, 1045, 2293, 6583, 2361, 102]),
        ("unbelievable",      ["[CLS]", "un", "##believe", "##able", "[SEP]"],      [101, 4895, 9884, 3085, 102]),
    ]
    for text, tokens, ids in bert_examples:
        print(f"  输入：'{text}'")
        print(f"  Tokens：{tokens}")
        print(f"  IDs：   {ids}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# 第四部分：Pipeline API 实战演示
# ══════════════════════════════════════════════════════════════════════════════

def demo_pipeline_tasks():
    """
    演示 HuggingFace Pipeline 支持的各种任务
    注意：这些代码在安装 transformers 后可以直接运行
    """
    print("\n\n" + "=" * 60)
    print("  Pipeline API：各种 NLP 任务演示")
    print("=" * 60)

    print("""
  以下是可以直接复制运行的代码示例（安装 transformers 后）：
  ─────────────────────────────────────────────────────────

  # ── 任务1：情感分析 ────────────────────────────────────────
  from transformers import pipeline

  classifier = pipeline("sentiment-analysis")
  results = classifier([
      "I love this product, it's amazing!",
      "This is the worst movie I've ever seen.",
      "The weather is okay today.",
  ])
  # 输出示例：
  # [{'label': 'POSITIVE', 'score': 0.9998},
  #  {'label': 'NEGATIVE', 'score': 0.9991},
  #  {'label': 'POSITIVE', 'score': 0.7432}]

  # ── 任务2：文本生成（GPT-2）────────────────────────────────
  generator = pipeline("text-generation", model="gpt2")
  result = generator(
      "Artificial intelligence will",
      max_length=50,
      num_return_sequences=2,
      temperature=0.7,    # 控制随机性：越小越确定，越大越多样
  )
  # 输出示例：
  # 'Artificial intelligence will change the way we live and work...'

  # ── 任务3：问答（抽取式 QA）────────────────────────────────
  qa_model = pipeline("question-answering",
                      model="deepset/roberta-base-squad2")
  result = qa_model(
      question="What is the capital of France?",
      context="Paris is the capital and largest city of France."
  )
  # 输出：{'answer': 'Paris', 'score': 0.998, 'start': 0, 'end': 5}

  # ── 任务4：命名实体识别（NER）──────────────────────────────
  ner = pipeline("ner", grouped_entities=True)
  result = ner("Apple Inc. was founded by Steve Jobs in Cupertino, California.")
  # 输出：
  # [{'entity_group': 'ORG',  'word': 'Apple Inc.',    'score': 0.997},
  #  {'entity_group': 'PER',  'word': 'Steve Jobs',    'score': 0.999},
  #  {'entity_group': 'LOC',  'word': 'Cupertino',     'score': 0.995},
  #  {'entity_group': 'LOC',  'word': 'California',    'score': 0.998}]

  # ── 任务5：机器翻译（Helsinki-NLP 模型）──────────────────
  translator = pipeline("translation_en_to_zh",
                        model="Helsinki-NLP/opus-mt-en-zh")
  result = translator("I love machine learning and natural language processing.")
  # 输出：[{'translation_text': '我喜欢机器学习和自然语言处理。'}]

  # ── 任务6：文本摘要 ────────────────────────────────────────
  summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
  article = \"\"\"
  The Eiffel Tower is a wrought iron lattice tower on the Champ de Mars in Paris,
  France. It is named after the engineer Gustave Eiffel, whose company designed
  and built the tower from 1887 to 1889 as the centerpiece of the 1889 World's
  Fair. It was initially criticized by some of France's leading artists and
  intellectuals for its design, but it has become a global cultural icon.
  \"\"\"
  result = summarizer(article, max_length=50, min_length=20)
  # 输出：简洁的摘要

  # ── 任务7：零样本分类（无需训练！）────────────────────────
  zero_shot = pipeline("zero-shot-classification")
  result = zero_shot(
      "This tutorial explains how to use neural networks for image recognition.",
      candidate_labels=["technology", "sports", "politics", "science"],
  )
  # 输出：{'labels': ['technology', 'science', ...], 'scores': [0.72, 0.18, ...]}
  """)


# ══════════════════════════════════════════════════════════════════════════════
# 第五部分：实际可运行的代码（尝试调用 transformers）
# ══════════════════════════════════════════════════════════════════════════════

def try_run_pipeline():
    """
    尝试实际运行一个简单的 pipeline
    如果 transformers 未安装，给出安装提示
    """
    print("\n\n" + "=" * 60)
    print("  实际运行测试")
    print("=" * 60)

    try:
        from transformers import pipeline

        print("  ✓ transformers 已安装，尝试运行情感分析...")
        print("  （首次运行会下载约 280MB 的模型，请耐心等待）")
        print()

        # 使用小型模型快速测试
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        test_texts = [
            "I love learning about Transformers!",
            "This is too difficult and frustrating.",
            "The weather is fine today.",
        ]

        print("  情感分析结果：")
        print(f"  {'输入文本':45s}  结果        得分")
        print("  " + "-" * 75)
        results = classifier(test_texts)
        for text, result in zip(test_texts, results):
            label_cn = "正面" if result["label"] == "POSITIVE" else "负面"
            print(f"  {text:45s}  {label_cn}  {result['score']:.4f}")

        print()
        print("  ✓ Pipeline 运行成功！")
        print("  ✓ 你已经成功调用了一个真实的 Transformer 模型")

    except ImportError:
        print("  ✗ transformers 未安装")
        print()
        print("  安装命令：")
        print("    pip install transformers")
        print()
        print("  如果下载慢，建议配置镜像后再安装：")
        print("    pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print()
        print("  安装后重新运行本文件即可看到实际效果")

    except Exception as e:
        print(f"  运行时出错：{e}")
        print()
        print("  可能的原因：")
        print("  1. 网络问题（模型下载失败）→ 检查 HF_ENDPOINT 镜像设置")
        print("  2. 磁盘空间不足（模型文件较大）")
        print("  3. 依赖版本冲突 → pip install transformers --upgrade")


# ══════════════════════════════════════════════════════════════════════════════
# 第六部分：HuggingFace vs Anthropic API 的区别
# ══════════════════════════════════════════════════════════════════════════════

def compare_hf_vs_api():
    print("\n\n" + "=" * 60)
    print("  HuggingFace 本地模型 vs Anthropic API 对比")
    print("=" * 60)

    print("""
  ┌──────────────────┬─────────────────────┬─────────────────────┐
  │ 维度             │ HuggingFace 本地     │ Anthropic API       │
  ├──────────────────┼─────────────────────┼─────────────────────┤
  │ 成本             │ 下载后免费           │ 按 token 收费       │
  │ 模型能力         │ 依赖具体模型         │ Claude 顶级能力     │
  │ 隐私             │ 数据本地，不外传     │ 数据发送到云端      │
  │ 硬件要求         │ 需要 GPU/RAM         │ 无需本地硬件        │
  │ 部署复杂度       │ 较高                 │ 简单（API调用）     │
  │ 可定制性         │ 可微调               │ 有限（prompt工程）  │
  │ 最新模型         │ 依赖社区更新         │ 始终最新Claude      │
  │ 离线使用         │ 支持                 │ 需要网络            │
  └──────────────────┴─────────────────────┴─────────────────────┘

  选择建议：
  · 学习/研究：HuggingFace（开源，可以看到内部结构）
  · 生产环境、需要强能力：Anthropic API / OpenAI API
  · 数据敏感、需要私有化：本地部署 LLaMA/Qwen 等开源模型
  · 快速原型：HuggingFace Pipeline（几行代码完成任务）

  你在本项目中学习的 Claude API 属于最后一类：
  通过调用 API 使用 Anthropic 的顶级模型能力
  """)


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # 首先配置镜像
    setup_hf_mirror()

    print("\n" + "=" * 60)
    print("  Transformer 第07课：HuggingFace 实战")
    print("=" * 60)

    explain_hf_ecosystem()
    explain_tokenizer()
    demo_pipeline_tasks()
    try_run_pipeline()    # 尝试实际运行（需要 transformers 已安装）
    compare_hf_vs_api()

    print("\n" + "=" * 60)
    print("  本课小结：")
    print("  · HuggingFace = AI 领域的 GitHub，35万+模型可免费用")
    print("  · Pipeline API：一行代码完成情感分析、生成、翻译等任务")
    print("  · Tokenizer：文本→tokens→数字ID，是使用模型的第一步")
    print("  · 本地模型 vs API：各有优势，按需选择")
    print()
    print("  下一课：综合回顾 + 知识图谱 (08_project_summary.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
