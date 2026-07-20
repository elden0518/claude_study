"""
==============================================================================
第二十课（补充）：RAG (Retrieval-Augmented Generation) 检索增强生成
==============================================================================

【为什么需要单独一课？】
现有课程没有涉及 RAG 技术，而这是 Agent 最重要的能力之一。
RAG 让 Agent 能够基于外部知识库回答问题，减少幻觉。

【学习目标】
- 理解 RAG 的原理和架构
- 掌握文档加载和分块策略
- 掌握向量化和向量检索
- 学会构建完整的 RAG Agent
- 理解高级 RAG 技术（重排序、查询改写）

【核心概念】
- Document Loading（文档加载）
- Text Splitting（文本分块）
- Embedding（向量化）
- Vector Store（向量存储）
- Retrieval（检索）
- RAG Agent（检索增强生成）

==============================================================================
"""

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# 第一部分：RAG 原理
# ============================================================================

def explain_rag_principle():
    """
    RAG 原理详解

    RAG = 检索 + 生成
    1. 检索：从知识库中找到相关文档
    2. 生成：基于检索到的文档生成回答
    """

    print("=" * 60)
    print("RAG (Retrieval-Augmented Generation) 原理")
    print("=" * 60)

    print("""
  ── 为什么需要 RAG？──

  普通 LLM 的问题：
  - 知识截止于训练时间，不知道最新信息
  - 容易产生幻觉（编造不存在的事实）
  - 无法访问私有文档（公司内部知识库）

  RAG 的解决方案：
  - 先从知识库检索相关文档
  - 然后基于检索到的文档生成回答
  - 减少幻觉，提高准确性


  ── RAG 工作流程 ──

  用户提问
     ↓
  [检索模块] 从知识库中找到相关文档
     ↓
  [增强模块] 将问题 + 相关文档组合成 Prompt
     ↓
  [LLM] 基于上下文生成回答
     ↓
  返回答案（附带引用来源）


  ── RAG vs 微调 ──

  RAG 优势：
  - 知识可以实时更新（不需要重新训练）
  - 可以追溯答案来源（可解释性强）
  - 成本低（不需要 GPU 训练）

  微调优势：
  - 推理速度快（不需要检索）
  - 可以学习特定风格/格式
  - 适合固定知识场景
    """)


# ============================================================================
# 第二部分：文档加载器
# ============================================================================

@dataclass
class Document:
    """文档模型"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __str__(self):
        return f"Document(id={self.doc_id}, len={len(self.content)})"


class DocumentLoader:
    """
    文档加载器

    【功能】
    从不同来源加载文档：
    - 文本文件
    - PDF 文件
    - 网页
    - 数据库
    """

    @staticmethod
    def from_text(text: str, metadata: Dict = None) -> List[Document]:
        """从文本创建文档"""
        return [Document(content=text, metadata=metadata or {})]

    @staticmethod
    def from_texts(texts: List[str], metadatas: List[Dict] = None) -> List[Document]:
        """从多个文本创建文档"""
        metadatas = metadatas or [{} for _ in texts]
        return [
            Document(content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]

    @staticmethod
    def from_file(file_path: str) -> List[Document]:
        """从文件加载文档"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [Document(content=content, metadata={"source": file_path})]


# ============================================================================
# 第三部分：文本分块策略
# ============================================================================

class CharacterTextSplitter:
    """
    字符分块器

    【原理】
    按字符数将文档切分成固定大小的块
    - chunk_size: 每块的最大字符数
    - chunk_overlap: 块之间的重叠字符数（保持上下文连贯）
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: List[Document]) -> List[Document]:
        """分块"""
        chunks = []
        for doc in documents:
            texts = self._split_text(doc.content)
            for i, text in enumerate(texts):
                chunks.append(Document(
                    content=text,
                    metadata={**doc.metadata, "chunk_index": i, "chunk_total": len(texts)}
                ))
        return chunks

    def _split_text(self, text: str) -> List[str]:
        """按字符分割文本"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        return chunks


class SemanticTextSplitter:
    """
    语义分块器

    【原理】
    按语义边界（段落、句子）分块
    - 保持语义完整性
    - 避免在句子中间切断
    """

    def __init__(self, separators: List[str] = None):
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", " "]

    def split(self, documents: List[Document]) -> List[Document]:
        """分块"""
        chunks = []
        for doc in documents:
            texts = self._split_by_semantic(doc.content)
            for i, text in enumerate(texts):
                if text.strip():
                    chunks.append(Document(
                        content=text.strip(),
                        metadata={**doc.metadata, "chunk_index": i}
                    ))
        return chunks

    def _split_by_semantic(self, text: str) -> List[str]:
        """按语义边界分割"""
        chunks = []
        current_chunk = []

        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                for part in parts:
                    if part.strip():
                        current_chunk.append(part)
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                return chunks

        # 如果没有找到分隔符，直接返回
        return [text] if text.strip() else []


# ============================================================================
# 第四部分：向量化与检索
# ============================================================================

class SimpleEmbedding:
    """
    简易向量化模型

    【原理】
    将文本转换为向量表示
    这里使用简化的词频向量，实际应用中应使用预训练模型
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.vocab: Dict[str, int] = {}

    def embed(self, text: str) -> List[float]:
        """将文本转换为向量"""
        # 简化的词频向量
        words = text.lower().split()
        vector = [0.0] * self.dimension

        for word in words:
            if word not in self.vocab:
                # 为新词分配一个维度
                if len(self.vocab) < self.dimension:
                    self.vocab[word] = len(self.vocab)
                else:
                    continue
            idx = self.vocab[word]
            vector[idx] += 1.0

        # 归一化
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量向量化"""
        return [self.embed(text) for text in texts]


class VectorStore:
    """
    向量存储

    【功能】
    - 存储文档向量
    - 计算相似度
    - 检索最相关的文档
    """

    def __init__(self, embedding_model: SimpleEmbedding):
        self.embedding_model = embedding_model
        self.documents: List[Document] = []
        self.vectors: List[List[float]] = []

    def add_documents(self, documents: List[Document]):
        """添加文档"""
        self.documents.extend(documents)
        vectors = self.embedding_model.embed_batch([doc.content for doc in documents])
        self.vectors.extend(vectors)
        print(f"  [VectorStore] 添加了 {len(documents)} 个文档，总计 {len(self.documents)} 个")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        检索最相关的文档

        【原理】
        1. 将查询转换为向量
        2. 计算查询向量与所有文档向量的余弦相似度
        3. 返回相似度最高的 top_k 个文档
        """
        if not self.vectors:
            return []

        # 查询向量化
        query_vector = self.embedding_model.embed(query)

        # 计算相似度
        similarities = []
        for i, doc_vector in enumerate(self.vectors):
            sim = self._cosine_similarity(query_vector, doc_vector)
            similarities.append((self.documents[i], sim))

        # 排序并返回 top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)


# ============================================================================
# 第五部分：RAG Agent 实现
# ============================================================================

class RAGAgent:
    """
    RAG Agent

    【功能】
    1. 接收用户问题
    2. 从知识库检索相关文档
    3. 构建增强的 Prompt
    4. 生成回答
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.top_k = 3

    def query(self, question: str) -> Dict[str, Any]:
        """
        处理用户查询

        【流程】
        1. 检索相关文档
        2. 构建上下文
        3. 生成回答（模拟）
        """
        print(f"\n{'='*60}")
        print(f"RAG Agent 处理查询")
        print(f"{'='*60}")
        print(f"\n  用户问题: {question}")

        # 1. 检索
        print(f"\n  [Step 1] 检索相关文档 (top_k={self.top_k})...")
        results = self.vector_store.search(question, top_k=self.top_k)

        if not results:
            return {
                "question": question,
                "answer": "抱歉，我没有找到相关信息。",
                "sources": []
            }

        # 2. 构建上下文
        print(f"\n  [Step 2] 构建上下文...")
        context_parts = []
        sources = []
        for i, (doc, score) in enumerate(results, 1):
            print(f"    - 文档 {i} (相似度: {score:.3f}): {doc.content[:50]}...")
            context_parts.append(doc.content)
            sources.append({
                "doc_id": doc.doc_id,
                "score": score,
                "content": doc.content[:100]
            })

        context = "\n\n".join(context_parts)

        # 3. 生成回答（模拟）
        print(f"\n  [Step 3] 生成回答...")
        prompt = self._build_prompt(question, context)
        answer = self._generate_answer(question, context)

        print(f"\n  回答: {answer}")
        print(f"\n  参考来源: {len(sources)} 个文档")

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "prompt": prompt
        }

    def _build_prompt(self, question: str, context: str) -> str:
        """构建增强的 Prompt"""
        return f"""基于以下上下文信息回答用户问题。

上下文：
{context}

用户问题：{question}

请基于上述上下文信息给出准确、完整的回答。如果上下文中的信息不足以回答问题，请明确说明。"""

    def _generate_answer(self, question: str, context: str) -> str:
        """生成回答（模拟 LLM 生成）"""
        # 实际应用中这里应该调用 LLM API
        return f"根据知识库中的信息，关于「{question}」的回答是：[基于 {len(context)} 字符的上下文生成]"


# ============================================================================
# 第六部分：高级 RAG 技术
# ============================================================================

class QueryRewriter:
    """
    查询改写

    【原理】
    在检索前对用户查询进行改写，提高检索效果：
    - 查询扩展：添加同义词
    - 查询分解：将复杂问题分解为多个子问题
    - HyDE：生成假设性文档嵌入
    """

    @staticmethod
    def expand(query: str) -> List[str]:
        """查询扩展"""
        # 简化的同义词扩展
        synonyms = {
            "AI": ["人工智能", "Artificial Intelligence"],
            "LLM": ["大语言模型", "Large Language Model"],
            "RAG": ["检索增强生成", "Retrieval-Augmented Generation"],
        }

        expanded = [query]
        for key, values in synonyms.items():
            if key.lower() in query.lower():
                expanded.extend(values)

        return expanded

    @staticmethod
    def decompose(query: str) -> List[str]:
        """查询分解"""
        # 简化的分解逻辑
        if "和" in query:
            parts = query.split("和")
            return [p.strip() for p in parts if p.strip()]
        return [query]

    @staticmethod
    def hyde(query: str) -> str:
        """
        HyDE (Hypothetical Document Embeddings)

        【原理】
        先生成一个假设性的答案文档
        然后用这个文档去检索（而不是用原始查询）
        """
        # 模拟生成假设文档
        return f"关于「{query}」，根据现有知识，答案可能涉及以下几个方面：首先...其次...最后..."


# ============================================================================
# 第七部分：RAG 评估
# ============================================================================

class RAGEvaluator:
    """
    RAG 评估器

    【评估维度】
    - 检索质量：Precision@K, Recall@K, MRR
    - 生成质量：答案准确性、完整性
    """

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Precision@K：前 K 个结果中有多少是相关的"""
        if k == 0:
            return 0.0
        top_k = retrieved[:k]
        relevant_count = sum(1 for doc in top_k if doc in relevant)
        return relevant_count / k

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Recall@K：前 K 个结果覆盖了多少相关文档"""
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        relevant_count = sum(1 for doc in top_k if doc in relevant)
        return relevant_count / len(relevant)

    @staticmethod
    def mrr(retrieved: List[str], relevant: List[str]) -> float:
        """MRR (Mean Reciprocal Rank)：第一个相关结果的排名倒数"""
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                return 1.0 / i
        return 0.0


# ============================================================================
# 第八部分：完整示例
# ============================================================================

def demo_rag_system():
    """演示完整的 RAG 系统"""

    print("=" * 60)
    print("RAG 系统完整演示")
    print("=" * 60)

    # 1. 准备知识库
    print("\n[Step 1] 构建知识库...")
    knowledge_base = [
        "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。它以简洁的语法和强大的功能著称。",
        "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习并改进性能，而无需进行明确的编程。",
        "深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的复杂模式。",
        "自然语言处理（NLP）是人工智能的重要领域，它致力于让计算机理解、解释和生成人类语言。",
        "向量数据库是一种专门用于存储和检索高维向量的数据库系统，广泛应用于相似性搜索场景。",
        "RAG（检索增强生成）是一种结合信息检索和文本生成的技术，它通过检索外部知识来增强 LLM 的回答能力。",
    ]

    documents = DocumentLoader.from_texts(
        knowledge_base,
        [{"source": f"doc_{i}"} for i in range(len(knowledge_base))]
    )
    print(f"  加载了 {len(documents)} 个文档")

    # 2. 文本分块
    print("\n[Step 2] 文本分块...")
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split(documents)
    print(f"  分块后得到 {len(chunks)} 个文本块")

    # 3. 向量化并存储
    print("\n[Step 3] 向量化并存储...")
    embedding_model = SimpleEmbedding(dimension=128)
    vector_store = VectorStore(embedding_model)
    vector_store.add_documents(chunks)

    # 4. 创建 RAG Agent
    print("\n[Step 4] 创建 RAG Agent...")
    rag_agent = RAGAgent(vector_store)

    # 5. 测试查询
    print("\n[Step 5] 测试查询...")
    test_queries = [
        "什么是 RAG？",
        "Python 是什么时候创建的？",
        "深度学习和机器学习有什么关系？",
    ]

    for query in test_queries:
        result = rag_agent.query(query)
        print(f"\n  问题: {result['question']}")
        print(f"  回答: {result['answer']}")
        print(f"  来源数: {len(result['sources'])}")
        print("-" * 60)

    # 6. 查询改写示例
    print("\n[Step 6] 查询改写技术...")
    print(f"  原始查询: '什么是 AI'")
    expanded = QueryRewriter.expand("什么是 AI")
    print(f"  扩展查询: {expanded}")

    decomposed = QueryRewriter.decompose("机器学习和深度学习")
    print(f"  分解查询: {decomposed}")

    hyde_doc = QueryRewriter.hyde("什么是 RAG")
    print(f"  HyDE 假设文档: {hyde_doc[:50]}...")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    explain_rag_principle()
    demo_rag_system()

    print("\n" + "=" * 60)
    print("课程总结")
    print("=" * 60)
    print("""
  本课介绍了 RAG (检索增强生成) 技术：

  核心知识点：
  1. RAG 原理：检索 + 生成
  2. 文档加载：从各种来源加载文档
  3. 文本分块：字符分块 / 语义分块
  4. 向量化：将文本转换为向量表示
  5. 向量检索：基于相似度的文档检索
  6. RAG Agent：完整的检索增强生成流程
  7. 高级技术：查询扩展、分解、HyDE
  8. 评估指标：Precision、Recall、MRR

  实际应用：
  - 企业知识库问答
  - 客服系统
  - 文档搜索
  - 知识管理
    """)
