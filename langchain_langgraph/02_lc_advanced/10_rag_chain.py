import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 国内镜像：必须在所有 huggingface 相关库导入之前设置，否则不生效
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

"""
主题：RAG Chain —— 检索增强生成，让 AI 回答你的私有文档

学习目标：
  1. 理解 RAG 的完整流程：加载→分块→嵌入→存储→检索→生成
  2. 掌握 create_retrieval_chain + create_stuff_documents_chain
  3. 学会在回答中显示文档来源（metadata）
  4. 对比有 RAG vs 无 RAG 的回答差异
  5. 理解 RAG 的局限性

核心概念：
  RAG（Retrieval-Augmented Generation）检索增强生成

  流程：
  1. 加载文档（Document Loaders）
  2. 分块（Text Splitters）
  3. 嵌入 + 存储（Embeddings + Vector Store）
  4. 检索（Retriever）
  5. 生成（LLM + Prompt）

  优势：不需要微调模型，直接扩展 LLM 的知识范围
  国内用户：Embedding 模型通过 hf-mirror.com 下载，无需翻墙

前置知识：已完成 07-09
"""

import os
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# HuggingFaceEmbeddings 导入（带 fallback）
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DOCS_DIR = os.path.join(SCRIPT_DIR, "sample_docs")
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


# ============================================================
# Part 1: 构建完整 RAG 管道（加载 → 分块 → 嵌入 → 存储 → 检索）
# ============================================================

def part1_build_rag_pipeline():
    """
    RAG 管道构建的完整步骤：

    Step 1: Document Loaders —— 把 sample_docs/ 下的所有文档加载进来
    Step 2: Text Splitters   —— 把长文档切成适合检索的小块
    Step 3: Embeddings       —— 把每个文本块转成向量
    Step 4: Vector Store     —— 把向量存入 FAISS 索引
    Step 5: Retriever        —— 封装检索接口，供 RAG 链调用

    这是一次性的"知识库建设"过程，完成后可反复查询。
    """
    print("=" * 60)
    print("Part 1: 构建 RAG 管道")
    print("=" * 60)

    # --- Step 1: 加载文档 ---
    print("Step 1: 加载 sample_docs/ 下的所有 .txt 文档...")
    loader = DirectoryLoader(
        SAMPLE_DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    docs = loader.load()
    print(f"  已加载 {len(docs)} 个文档：")
    for doc in docs:
        filename = os.path.basename(doc.metadata.get("source", "未知"))
        print(f"    - {filename} ({len(doc.page_content)} 字符)")
    print()

    # --- Step 2: 文本分块 ---
    print("Step 2: 对文档进行分块...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)
    print(f"  分块完成：{len(docs)} 个文档 → {len(chunks)} 个文本块")
    print(f"  chunk_size=300, chunk_overlap=50")
    print()

    # --- Step 3 & 4: 嵌入 + 构建 FAISS 向量库 ---
    print(f"Step 3 & 4: 嵌入文本块并构建 FAISS 向量库...")
    print(f"  使用模型：{EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"  FAISS 索引构建完成，包含 {len(chunks)} 个向量")
    print()

    # --- Step 5: 创建 Retriever ---
    print("Step 5: 创建 Retriever...")
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3},  # 每次检索返回最相关的 3 个文本块
    )
    print("  Retriever 就绪（k=3）")
    print()

    return retriever, vectorstore


# ============================================================
# Part 2: 使用 create_retrieval_chain 进行 RAG 查询
# ============================================================

def part2_query_with_rag(retriever):
    """
    create_retrieval_chain + create_stuff_documents_chain 是 LangChain
    推荐的高层 RAG 接口。

    create_stuff_documents_chain：
      - "Stuff" 策略：把所有检索到的文档直接塞（stuff）进 prompt
      - 适合文档总量不太大的场景
      - 其他策略：Map-Reduce、Refine（适合超长文档，但本例不使用）

    create_retrieval_chain：
      - 组合 Retriever + 文档处理链
      - 输入：{"input": "用户问题"}
      - 输出：{"answer": "回答", "context": [检索到的 Document 列表]}
    """
    print("=" * 60)
    print("Part 2: 使用 RAG 链回答问题")
    print("=" * 60)

    llm = ChatAnthropic(
        model="ppio/pa/claude-sonnet-4-6",
        max_tokens=512,
    )

    # RAG 专用 Prompt：要求 LLM 基于检索到的 context 回答
    # {context} 由 create_stuff_documents_chain 自动填充（检索到的文档内容）
    # {input}   是用户的原始问题
    rag_prompt = ChatPromptTemplate.from_template(
        """你是一个知识库问答助手。请严格根据以下参考文档回答问题。
如果参考文档中没有相关信息，请明确说明"文档中未找到相关信息"。

参考文档：
{context}

问题：{input}

回答："""
    )

    # 创建文档处理链（Stuff 策略）
    # 它负责：把检索到的 Document 列表格式化，填入 prompt 的 {context}
    document_chain = create_stuff_documents_chain(llm, rag_prompt)

    # 创建完整 RAG 链
    # 它负责：接收 input → 调用 retriever 检索 → 传给 document_chain
    rag_chain = create_retrieval_chain(retriever, document_chain)

    # 查询
    query = "LangGraph 有什么特点？"
    print(f"问题：{query}")
    print()

    # 调用 RAG 链
    result = rag_chain.invoke({"input": query})

    print("【RAG 回答】")
    print(result["answer"])
    print()

    # 显示文档来源（回答的依据）
    print("【参考文档来源】")
    seen_sources = set()
    for doc in result.get("context", []):
        source = doc.metadata.get("source", "未知")
        filename = os.path.basename(source)
        if filename not in seen_sources:
            seen_sources.add(filename)
            print(f"  - {filename}")
            # 显示引用片段的前 60 字符
            preview = doc.page_content.replace("\n", " ").strip()[:60]
            print(f"    引用片段：{preview}...")
    print()

    return rag_chain


# ============================================================
# Part 3: 对比 有 RAG vs 无 RAG 的回答差异
# ============================================================

def part3_rag_vs_no_rag(rag_chain):
    """
    对比实验揭示 RAG 的核心价值：

    无 RAG（直接问 LLM）：
      - LLM 只能使用训练数据中的知识
      - 对于私有文档、最新信息无法准确回答
      - 可能产生"幻觉"（hallucination）——编造看似合理但不准确的内容

    有 RAG（检索 + 生成）：
      - LLM 可以访问私有文档的具体内容
      - 回答有据可查，可溯源验证
      - 大幅减少幻觉，提高准确性

    RAG 的局限性：
      - 检索质量依赖分块策略和嵌入模型
      - 当答案分散在多个文档时，合成质量下降
      - 对于需要深度推理的问题效果有限
    """
    print("=" * 60)
    print("Part 3: 对比 有 RAG vs 无 RAG 的回答")
    print("=" * 60)

    llm = ChatAnthropic(
        model="ppio/pa/claude-sonnet-4-6",
        max_tokens=256,
    )

    # 用同一个问题对比两种方式
    question = "请介绍一下 LangChain 的主要组件有哪些？"
    print(f"问题：{question}")
    print()

    # --- 无 RAG：直接向 LLM 提问 ---
    print("【无 RAG 的回答（直接问 LLM，仅凭训练数据）】")
    direct_prompt = ChatPromptTemplate.from_template(
        "请回答以下问题（简洁，100字以内）：{question}"
    )
    direct_chain = direct_prompt | llm
    no_rag_response = direct_chain.invoke({"question": question})
    print(no_rag_response.content)
    print()

    # --- 有 RAG：检索 + 生成 ---
    print("【有 RAG 的回答（基于 sample_docs/ 文档库）】")
    rag_result = rag_chain.invoke({"input": question})
    print(rag_result["answer"])
    print()

    print("【对比分析】")
    print("  有 RAG 的回答直接引用了 sample_docs/ 中的具体内容，")
    print("  能够提供与私有文档一致的准确信息，而非模型的通用知识。")
    print()
    print("【RAG 的局限性提示】")
    print("  • 答案质量取决于文档质量和分块策略")
    print("  • 问题与文档内容差异太大时，检索可能失效")
    print("  • 文档库过大时，需要考虑向量库的性能优化")
    print()


# ============================================================
# main
# ============================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        10_rag_chain.py —— 完整 RAG 管道演示              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    retriever, vectorstore = part1_build_rag_pipeline()
    rag_chain = part2_query_with_rag(retriever)
    part3_rag_vs_no_rag(rag_chain)

    print("=" * 60)
    print("RAG Chain 演示完毕！")
    print()
    print("关键要点：")
    print("  • RAG = 检索（Retrieval）+ 增强生成（Augmented Generation）")
    print("  • 完整流程：加载 → 分块 → 嵌入 → 存储 → 检索 → 生成")
    print("  • create_stuff_documents_chain 把文档塞进 Prompt")
    print("  • create_retrieval_chain 组合检索器与文档处理链")
    print("  • metadata.source 让回答可溯源，增强可信度")
    print("=" * 60)


if __name__ == "__main__":
    main()
