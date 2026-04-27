"""
主题：RAG 高级技巧 —— 提升检索质量和回答准确性

学习目标：
  1. 掌握查询重写（Query Rewriting）：优化用户问题
  2. 掌握混合检索（Hybrid Search）：关键词 + 向量相似度
  3. 掌握重排序（Re-ranking）：精排检索结果
  4. 掌握多跳推理（Multi-hop Retrieval）：分步检索复杂问题
  5. 理解各技巧的适用场景和性能权衡

核心概念：
  基础 RAG 的问题：
  - 用户问题表述不清 → 检索不相关文档
  - 单一检索策略 → 召回率或精确率低
  - 检索结果未精排 → 包含噪声
  
  高级技巧解决：
  查询重写 → 让问题更适合检索
  混合检索 → 结合不同检索优势
  重排序 → 提高 Top-K 质量
  多跳推理 → 处理需要多步推理的问题

前置知识：已完成 02_lc_advanced/10_rag_chain.py
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 国内镜像
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=256)


# =============================================================================
# Part 1：查询重写（Query Rewriting）
# =============================================================================

def demo_query_rewriting():
    """
    查询重写的核心思想：在检索前优化用户问题。
    
    常见重写策略：
    1. 扩展缩写： "AI" → "人工智能"
    2. 补充上下文： "它的优点？" → "Python 的优点是什么？"
    3. 分解复合问题： "A 和 B 的区别？" → ["A 的特点", "B 的特点"]
    4. 转换为陈述句： "如何安装？" → "安装步骤和方法"
    
    实现方式：
    - LLM 重写（灵活但慢）
    - 规则重写（快但不灵活）
    - HyDE（假设性文档嵌入）
    """
    print("=" * 60)
    print("Part 1: Query Rewriting —— 查询重写")
    print("=" * 60)
    
    # 方法 1：LLM 重写
    print("\n【方法 1】LLM 重写用户问题")
    
    rewrite_prompt = ChatPromptTemplate.from_template(
        """你是一个查询优化专家。请重写以下用户问题，使其更适合文档检索。

要求：
1. 补充缺失的上下文
2. 消除歧义
3. 转换为关键词丰富的陈述句
4. 保持原意不变

原始问题：{question}

重写后的查询（只输出查询，不要解释）："""
    )
    
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    
    test_questions = [
        "它怎么用？",  # 缺少上下文
        "Python 和 Java 哪个更好？",  # 复合问题
        "API 限流怎么办？",  # 专业术语
    ]
    
    for question in test_questions:
        print(f"\n  原始问题：{question}")
        rewritten = rewrite_chain.invoke({"question": question})
        print(f"  重写后  ：{rewritten[:100]}")
    
    # 方法 2：多查询生成（生成多个变体，分别检索）
    print("\n\n【方法 2】多查询生成（Multi-Query）")
    
    multi_query_prompt = ChatPromptTemplate.from_template(
        """请为以下问题生成 3 个不同表述的查询版本，用于提高检索覆盖率。

原始问题：{question}

输出格式（每行一个查询）：
查询1: ...
查询2: ...
查询3: ..."""
    )
    
    multi_query_chain = multi_query_prompt | llm | StrOutputParser()
    
    question = "LangChain 如何实现记忆功能？"
    print(f"\n  原始问题：{question}")
    print(f"\n  生成的查询变体：")
    
    variants = multi_query_chain.invoke({"question": question})
    for line in variants.strip().split("\n"):
        if line.strip():
            print(f"    {line}")
    
    print("\n💡 使用建议：")
    print("  • 对每个变体分别检索，合并结果")
    print("  • 去重后再送入 LLM 生成答案")
    print("  • 适合模糊或不确定的问题")


# =============================================================================
# Part 2：混合检索（Hybrid Search）
# =============================================================================

def demo_hybrid_search():
    """
    混合检索结合多种检索策略的优势：
    
    向量检索（Dense Retrieval）：
    ✅ 语义匹配好，能理解同义词
    ❌ 可能忽略精确关键词
    
    关键词检索（Sparse Retrieval / BM25）：
    ✅ 精确匹配专有名词、代码
    ❌ 无法理解语义相似性
    
    混合策略：
    1. 并行执行两种检索
    2. 归一化分数
    3. 加权融合排序
    """
    print("\n" + "=" * 60)
    print("Part 2: Hybrid Search —— 混合检索")
    print("=" * 60)
    
    print("""
  📊 混合检索架构：
  ──────────────────────────────────────────────
  
  用户问题
      ↓
  ┌──────────────┬──────────────┐
  │ 向量检索      │ 关键词检索    │
  │ (FAISS/Chroma)│ (BM25/Elastic)│
  └──────┬───────┴──────┬───────┘
         ↓              ↓
  ┌──────────────────────────┐
  │   分数归一化 + 加权融合    │
  │   score = α*vector + β*bm25 │
  └──────────┬───────────────┘
             ↓
      Top-K 文档
             ↓
         LLM 生成
  
  ⚙️ 权重配置建议：
  ──────────────────────────────────────────────
  
  • 通用问答：α=0.7, β=0.3（偏语义）
  • 技术文档：α=0.5, β=0.5（平衡）
  • 代码搜索：α=0.3, β=0.7（偏关键词）
  
  🛠️  实现方案：
  ──────────────────────────────────────────────
  
  方案 1：LangChain EnsembleRetriever
  ─────────────────────────────────────
  from langchain.retrievers import EnsembleRetriever
  from langchain_community.retrievers import BM25Retriever
  
  vector_retriever = vectorstore.as_retriever(k=5)
  bm25_retriever = BM25Retriever.from_documents(docs, k=5)
  
  ensemble = EnsembleRetriever(
      retrievers=[vector_retriever, bm25_retriever],
      weights=[0.7, 0.3]
  )
  
  方案 2：手动融合（更灵活）
  ─────────────────────────────────────
  # 分别检索
  vector_docs = vector_retriever.invoke(query)
  keyword_docs = bm25_retriever.invoke(query)
  
  # 合并并去重
  all_docs = merge_and_deduplicate(vector_docs, keyword_docs)
  
  # 重新排序
  ranked_docs = rerank(all_docs, query)
    """)
    
    # 演示代码结构
    print("\n【代码示例】EnsembleRetriever 使用")
    print("-" * 60)
    
    example_code = '''
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 准备文档
documents = [...]  # 你的文档列表

# 2. 创建向量检索器
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. 创建 BM25 检索器
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# 4. 组合成混合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # 向量权重 0.7，BM25 权重 0.3
)

# 5. 使用
query = "如何在 LangChain 中实现缓存？"
results = ensemble_retriever.invoke(query)
print(f"检索到 {len(results)} 个文档")
    '''
    print(example_code)
    print("-" * 60)


# =============================================================================
# Part 3：重排序（Re-ranking）
# =============================================================================

def demo_reranking():
    """
    重排序：对初步检索结果进行二次排序，提升 Top-K 质量。
    
    为什么需要重排序？
    - 向量检索返回的 Top-50 可能包含噪声
    - 简单的余弦相似度不够精细
    - Cross-Encoder 能理解 query-doc 交互关系
    
    常用重排序模型：
    1. Cohere Rerank（商业 API，效果好）
    2. BGE Reranker（开源，中文支持好）
    3. ColBERT（高效，适合大规模）
    """
    print("\n" + "=" * 60)
    print("Part 3: Re-ranking —— 重排序")
    print("=" * 60)
    
    print("""
  📊 重排序流程：
  ──────────────────────────────────────────────
  
  第 1 阶段：粗排（快速召回）
  ├─ 向量检索：Top-50 候选文档
  └─ 耗时：< 100ms
  
  第 2 阶段：精排（高质量排序）
  ├─ Cross-Encoder 重排序
  └─ 耗时：100-500ms
  
  最终输出：Top-5 最相关文档
    """)
    
    # 方法 1：使用 Cohere Rerank（需要 API Key）
    print("\n【方法 1】Cohere Rerank（商业 API）")
    
    cohere_example = '''
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# 基础检索器
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# 重排序器
reranker = CohereRerank(
    model="rerank-multilingual-v3.0",
    top_n=5,  # 最终返回 5 个文档
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

# 组合
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)

# 使用
docs = compression_retriever.invoke("用户问题")
print(f"重排序后保留 {len(docs)} 个文档")
    '''
    print(cohere_example)
    
    # 方法 2：使用 BGE Reranker（开源）
    print("\n【方法 2】BGE Reranker（开源，推荐）")
    
    bge_example = '''
from FlagEmbedding import FlagReranker

# 初始化重排序模型
reranker = FlagReranker(
    'BAAI/bge-reranker-v2-m3',
    use_fp16=True  # 加速推理
)

# 计算相关性分数
pairs = [
    ("用户问题", "文档1内容"),
    ("用户问题", "文档2内容"),
    # ...
]
scores = reranker.compute_score(pairs)

# 按分数排序
ranked_docs = sorted(zip(scores, documents), reverse=True)
top_docs = [doc for _, doc in ranked_docs[:5]]
    '''
    print(bge_example)
    
    print("\n💡 性能对比：")
    print("  • 无重排序：Recall@5 = 65%")
    print("  • Cohere Rerank：Recall@5 = 82% (+17%)")
    print("  • BGE Reranker：Recall@5 = 78% (+13%)")
    print("\n⚠️  注意：重排序会增加延迟，权衡使用")


# =============================================================================
# Part 4：多跳推理（Multi-hop Retrieval）
# =============================================================================

def demo_multi_hop_retrieval():
    """
    多跳推理：将复杂问题分解为多个子问题，逐步检索。
    
    适用场景：
    - "A 公司的 CEO 之前在哪工作？"（需要先查 CEO，再查工作经历）
    - "比较 Python 和 Java 的性能差异"（需要分别检索两者）
    - "基于 X 理论的 Y 应用案例"（需要先理解理论，再找案例）
    
    实现模式：
    1. 分解问题 → 2. 逐步检索 → 3. 整合答案
    """
    print("\n" + "=" * 60)
    print("Part 4: Multi-hop Retrieval —— 多跳推理")
    print("=" * 60)
    
    # 示例：两跳推理
    print("\n【示例】两跳推理工作流")
    
    def decompose_question(question: str) -> list:
        """使用 LLM 分解复杂问题"""
        decompose_prompt = ChatPromptTemplate.from_template(
            """请将以下复杂问题分解为 2-3 个简单的子问题。

复杂问题：{question}

输出格式（每行一个问题）：
子问题1: ...
子问题2: ...
子问题3: ...（可选）"""
        )
        
        chain = decompose_prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question})
        
        # 解析子问题
        sub_questions = []
        for line in result.strip().split("\n"):
            if "子问题" in line and ":" in line:
                sub_q = line.split(":", 1)[1].strip()
                if sub_q:
                    sub_questions.append(sub_q)
        
        return sub_questions
    
    def retrieve_for_question(question: str) -> str:
        """模拟检索（实际应连接向量数据库）"""
        print(f"    [检索] {question}")
        # 这里应该调用 retriever.invoke(question)
        return f"[关于「{question}」的检索结果]"
    
    def synthesize_answer(original_q: str, sub_results: list) -> str:
        """整合所有子问题的答案"""
        synthesize_prompt = ChatPromptTemplate.from_template(
            """基于以下子问题的检索结果，回答原始问题。

原始问题：{original_q}

子问题检索结果：
{sub_results}

综合回答："""
        )
        
        chain = synthesize_prompt | llm | StrOutputParser()
        return chain.invoke({
            "original_q": original_q,
            "sub_results": "\n".join(sub_results)
        })
    
    # 测试多跳推理
    complex_question = "LangGraph 相比传统 LangChain Chain 有什么优势？"
    
    print(f"\n  原始问题：{complex_question}")
    print(f"\n  【第 1 步】分解问题")
    
    sub_questions = decompose_question(complex_question)
    for i, sq in enumerate(sub_questions, 1):
        print(f"    子问题{i}: {sq}")
    
    print(f"\n  【第 2 步】逐跳检索")
    sub_results = []
    for sq in sub_questions:
        result = retrieve_for_question(sq)
        sub_results.append(result)
    
    print(f"\n  【第 3 步】整合答案")
    final_answer = synthesize_answer(complex_question, sub_results)
    print(f"    {final_answer[:100]}...")
    
    print("\n💡 进阶技巧：")
    print("  • 自反思多跳：每跳后验证是否需要继续")
    print("  • 并行多跳：独立子问题可并行检索")
    print("  • 迭代细化：根据中间结果调整后续查询")


# =============================================================================
# Part 5：完整的高级 RAG  pipeline
# =============================================================================

def demo_advanced_rag_pipeline():
    """
    将所有高级技巧组合成完整的 RAG 系统。
    """
    print("\n" + "=" * 60)
    print("Part 5: Advanced RAG Pipeline —— 完整架构")
    print("=" * 60)
    
    architecture = """
  🏗️  生产级 RAG 系统架构：
  ──────────────────────────────────────────────
  
  用户问题
      ↓
  ┌─────────────────────┐
  │  1. 查询重写         │ ← LLM 优化问题表述
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │  2. 多查询生成       │ ← 生成 3-5 个变体
  └──────┬──────────────┘
         ↓
  ┌─────────────────────┐
  │  3. 混合检索         │ ← 向量 + BM25
  │     - 向量 Top-20    │
  │     - BM25 Top-20    │
  └──────┬──────────────┘
         ↓
  ┌─────────────────────┐
  │  4. 去重 & 合并      │ ← 去除重复文档
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │  5. 重排序           │ ← Cross-Encoder 精排
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │  6. 选择 Top-5       │ ← 最终文档
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │  7. LLM 生成答案     │ ← 带引用的回答
  └──────────┬──────────┘
             ↓
         最终答案
  
  📊 性能指标（典型值）：
  ──────────────────────────────────────────────
  
  • 端到端延迟：2-5 秒
  • 检索准确率：+15-20%（相比基础 RAG）
  • 成本增加：约 2-3 倍（主要来自 LLM 重写）
  
  ⚙️ 优化建议：
  ──────────────────────────────────────────────
  
  1. 缓存重写后的查询（相同问题无需重复重写）
  2. 异步执行多查询检索（降低延迟）
  3. 根据问题复杂度动态选择策略
     - 简单问题：直接检索
     - 复杂问题：启用多跳
  4. 监控各环节效果，调整权重
    """
    print(architecture)


def main():
    print("=" * 60)
    print("RAG 高级技巧详解")
    print("=" * 60)
    
    demo_query_rewriting()
    demo_hybrid_search()
    demo_reranking()
    demo_multi_hop_retrieval()
    demo_advanced_rag_pipeline()
    
    print("\n" + "=" * 60)
    print("学习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
