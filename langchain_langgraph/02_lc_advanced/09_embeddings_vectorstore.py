import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
主题：Embeddings & Vector Store —— 把文本变成向量，实现语义搜索

学习目标：
  1. 理解什么是 Embedding（文本→向量的映射）
  2. 掌握 HuggingFaceEmbeddings（本地模型，无需额外 API Key）
  3. 掌握 FAISS 向量库的构建和相似度搜索
  4. 理解 as_retriever() 的作用
  5. 理解语义搜索 vs 关键词搜索的区别

核心概念：
  Embedding = 把文本映射到高维向量空间
  语义相近的文本，在向量空间中距离也近
  FAISS = Facebook AI 开源的高效向量相似度搜索库

  注意：首次运行会下载 HuggingFace 模型（约 100MB）
  国内用户：自动使用 hf-mirror.com 镜像，无需翻墙

前置知识：已完成 08_text_splitters.py
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 尝试从 langchain_huggingface 导入，若失败则 fallback 到 langchain_community
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# 国内镜像：无法访问 huggingface.co 时自动切换到 hf-mirror.com
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 嵌入模型：BAAI/bge-small-zh-v1.5 是专为中文优化的小型嵌入模型
# 首次运行会自动下载（约 100MB），后续运行直接加载本地缓存
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 演示用的中文编程知识句子
PROGRAMMING_SENTENCES = [
    "面向对象编程是一种以对象为中心的编程范式，使用类和对象来组织代码。",
    "Python 是一种简洁易读的高级编程语言，广泛用于数据科学和 AI 开发。",
    "函数是可重用的代码块，接收输入参数并返回结果，有助于代码模块化。",
    "数据结构如列表、字典、集合是存储和组织数据的基本工具。",
    "异步编程允许程序在等待 I/O 操作时继续执行其他任务，提高并发性能。",
    "版本控制系统 Git 用于追踪代码变更，方便团队协作和代码回退。",
]


# ============================================================
# Part 1: 初始化 HuggingFaceEmbeddings 并测试向量维度
# ============================================================

def part1_embeddings_basics():
    """
    HuggingFaceEmbeddings 使用本地加载的 Transformer 模型生成文本向量。

    优点：
      - 不需要额外的 API Key（不同于 OpenAI Embeddings）
      - 可完全离线运行（模型下载后）
      - 中文模型效果好（如 BAAI/bge 系列）

    向量维度（dimension）= 模型输出的浮点数数量，代表语义空间的维数。
    bge-small-zh-v1.5 的维度为 512。
    """
    print("=" * 60)
    print("Part 1: HuggingFaceEmbeddings 基础 —— 文本转向量")
    print("=" * 60)

    print(f"加载嵌入模型：{EMBEDDING_MODEL}")
    print("（首次运行会下载模型文件，请耐心等待）")
    print()

    # 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 嵌入单个句子
    test_sentence = "Python 是一门优秀的编程语言"
    vector = embeddings.embed_query(test_sentence)

    print(f"测试句子：{test_sentence}")
    print(f"向量维度：{len(vector)}")
    print(f"向量前 5 个值：{[round(v, 4) for v in vector[:5]]}")
    print()
    print("说明：每个浮点数代表文本在语义空间某一维度上的特征值")
    print("语义相近的文本，其向量的余弦相似度更高")
    print()

    return embeddings


# ============================================================
# Part 2: 构建 FAISS 向量库并进行语义搜索
# ============================================================

def part2_faiss_similarity_search(embeddings):
    """
    FAISS（Facebook AI Similarity Search）是高效的向量相似度搜索库。

    构建流程：
      1. 将文本列表嵌入为向量
      2. 存入 FAISS 索引

    搜索流程：
      1. 将查询文本嵌入为向量
      2. 在索引中找到最相似的 k 个向量
      3. 返回对应的 Document 对象

    语义搜索 vs 关键词搜索：
      - 关键词搜索：必须包含相同词汇才能匹配
      - 语义搜索：理解意思，即使词汇不同也能匹配
        例如："面向对象" 和 "类与对象" 会被认为语义相近
    """
    print("=" * 60)
    print("Part 2: FAISS 向量库 —— 构建索引与相似度搜索")
    print("=" * 60)

    print("构建 FAISS 向量库...")
    print(f"知识库句子数量：{len(PROGRAMMING_SENTENCES)}")
    print()

    # from_texts() 自动完成：文本嵌入 → 构建 FAISS 索引
    vectorstore = FAISS.from_texts(
        texts=PROGRAMMING_SENTENCES,
        embedding=embeddings,
    )

    print("向量库构建完成！")
    print()

    # 语义搜索：查询"什么是面向对象"
    query = "什么是面向对象"
    print(f"查询：{query}")
    print(f"返回最相似的 k=3 个结果：")
    print()

    # similarity_search 返回 List[Document]，按相似度从高到低排列
    results = vectorstore.similarity_search(query, k=3)

    for i, doc in enumerate(results):
        print(f"  结果 {i + 1}：{doc.page_content}")
    print()

    # 也可以获取带分数的结果（分数越低越相似，使用 L2 距离）
    print("【带相似度分数的搜索结果（L2 距离，越小越相似）】")
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
    for doc, score in results_with_scores:
        print(f"  分数 {score:.4f}：{doc.page_content[:50]}...")
    print()

    return vectorstore


# ============================================================
# Part 3: as_retriever() + LCEL 链
# ============================================================

def part3_retriever_chain(vectorstore):
    """
    as_retriever() 把 VectorStore 转换为 LangChain 的 Retriever 接口。

    Retriever 是一个统一的接口，输入 query 字符串，返回相关 Document 列表。
    这使得不同的向量库（FAISS、Chroma、Pinecone 等）可以无缝替换。

    LCEL RAG 链的结构：
      用户问题 → Retriever 检索相关文档 → 拼接成 context
             ↓
      ChatPromptTemplate（context + 问题）→ LLM → 输出答案
    """
    print("=" * 60)
    print("Part 3: as_retriever() + LCEL 链 —— 基础 RAG")
    print("=" * 60)

    # 将向量库转为 Retriever
    # search_kwargs={"k": 2} 表示每次检索返回 2 个最相似的文档
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    print("Retriever 创建完成（基于 FAISS，k=2）")
    print()

    # 初始化 LLM
    llm = ChatAnthropic(
        model="ppio/pa/claude-sonnet-4-6",
        max_tokens=256,
    )

    # RAG Prompt 模板
    # {context} 由检索到的文档填充，{question} 是用户问题
    prompt = ChatPromptTemplate.from_template(
        """你是一个编程知识助手。请根据以下参考资料回答问题。

参考资料：
{context}

问题：{question}

请基于参考资料给出简洁准确的回答："""
    )

    # 辅助函数：将 List[Document] 转为字符串
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 构建 LCEL RAG 链
    # RunnablePassthrough() 直接传递 question 字段
    rag_chain = (
        {
            "context": retriever | format_docs,  # 检索 → 格式化文档
            "question": RunnablePassthrough(),    # 直接传递问题
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 提问
    question = "请解释什么是面向对象编程"
    print(f"问题：{question}")
    print()

    answer = rag_chain.invoke(question)
    print("【RAG 回答】")
    print(answer)
    print()


# ============================================================
# main
# ============================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   09_embeddings_vectorstore.py —— Embeddings & FAISS     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Part 1 返回 embeddings 对象，供后续复用（避免重复加载模型）
    embeddings = part1_embeddings_basics()

    # Part 2 返回 vectorstore 对象，供 Part 3 使用
    vectorstore = part2_faiss_similarity_search(embeddings)

    part3_retriever_chain(vectorstore)

    print("=" * 60)
    print("Embeddings & Vector Store 演示完毕！")
    print()
    print("关键要点：")
    print("  • Embedding 把文本映射为高维向量，语义相近则向量相近")
    print("  • HuggingFaceEmbeddings 本地运行，无需额外 API Key")
    print("  • FAISS 提供高效的向量相似度搜索能力")
    print("  • as_retriever() 统一接口，支持不同向量库无缝切换")
    print("  • 语义搜索比关键词搜索更能理解用户意图")
    print("=" * 60)


if __name__ == "__main__":
    main()
