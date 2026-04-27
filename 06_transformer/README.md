# 06_transformer — Transformer 从入门到实践

## 模块简介

本模块系统讲解 Transformer 架构，从历史背景到核心原理，从数学推导到工程实践，帮助你建立对现代大型语言模型（LLM）的完整认知。

前置要求：Python 基础即可，无需深度学习或数学背景。

---

## 学习路径（按顺序学习）

| 文件 | 主题 | 关键概念 | 难度 |
|------|------|----------|------|
| `01_why_transformer.py` | 为什么需要 Transformer？ | RNN缺陷、并行计算 | ⭐ |
| `02_attention_intuition.py` | 注意力机制直觉 | 查字典类比、软注意力 | ⭐ |
| `03_self_attention_math.py` | 自注意力数学推导 | Q/K/V矩阵、softmax | ⭐⭐ |
| `04_multihead_positional.py` | 多头注意力 + 位置编码 | 多维语义、sin/cos编码 | ⭐⭐ |
| `05_encoder_decoder_arch.py` | 编码器-解码器架构 | FFN、残差连接、LayerNorm | ⭐⭐⭐ |
| `06_bert_gpt_llm.py` | BERT vs GPT vs 现代LLM | 预训练范式、模型家族 | ⭐⭐ |
| `07_huggingface_practice.py` | HuggingFace 实战 | Pipeline、Tokenizer | ⭐⭐ |
| `08_project_summary.py` | 综合回顾 + 知识图谱 | 速查表、面试题、资源 | ⭐ |

---

## 快速运行

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 安装依赖（如果还没安装）
pip install numpy transformers

# 按顺序运行每个文件
python 06_transformer/01_why_transformer.py
python 06_transformer/02_attention_intuition.py
python 06_transformer/03_self_attention_math.py
python 06_transformer/04_multihead_positional.py
python 06_transformer/05_encoder_decoder_arch.py
python 06_transformer/06_bert_gpt_llm.py
python 06_transformer/07_huggingface_practice.py
python 06_transformer/08_project_summary.py
```

---

## 核心概念一览

```
Transformer 知识树
│
├── 注意力机制 (Attention)
│   ├── 自注意力 (Self-Attention)：词与词之间的关联
│   ├── 缩放点积注意力 (Scaled Dot-Product)：核心计算
│   └── 多头注意力 (Multi-Head)：并行捕捉多维语义
│
├── 编码器 (Encoder)
│   ├── 自注意力层
│   ├── 前馈神经网络 (FFN)
│   └── 残差连接 + LayerNorm
│
├── 解码器 (Decoder)
│   ├── 掩码自注意力 (Masked Self-Attention)
│   ├── 交叉注意力 (Cross-Attention)
│   └── 残差连接 + LayerNorm
│
└── 衍生模型
    ├── Encoder-Only：BERT（理解任务）
    ├── Decoder-Only：GPT/Claude（生成任务）
    └── Encoder-Decoder：T5/BART（翻译/摘要）
```

---

## 参考资料

- 原始论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- 可视化教程：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- HuggingFace 文档：https://huggingface.co/docs/transformers
