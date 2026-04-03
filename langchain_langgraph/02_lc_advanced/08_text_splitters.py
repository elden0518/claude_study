import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
主题：Text Splitters —— 把长文档切割成适合 LLM 处理的小块

学习目标：
  1. 理解为什么要分块（LLM 上下文窗口有限，嵌入模型也有限制）
  2. 掌握 RecursiveCharacterTextSplitter（最常用）
  3. 理解 chunk_size 和 chunk_overlap 的作用
  4. 对比不同分块策略的效果
  5. 理解分块对 RAG 检索质量的影响

核心概念：
  chunk_size   = 每个块的最大字符数
  chunk_overlap = 相邻块之间重叠的字符数（防止语义被切断）

  RecursiveCharacterTextSplitter 按 ["\n\n", "\n", " ", ""] 优先级分割，
  尽量在段落、句子、词语边界切割，保持语义完整性

前置知识：已完成 07_document_loaders.py
"""

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DOCS_DIR = os.path.join(SCRIPT_DIR, "sample_docs")


# ============================================================
# Part 1: RecursiveCharacterTextSplitter 基本用法
# ============================================================

def part1_basic_splitting():
    """
    RecursiveCharacterTextSplitter 是 LangChain 中最常用的文本分割器。

    工作原理：
      1. 尝试用 "\n\n"（段落分隔符）切割
      2. 若某段仍超过 chunk_size，则用 "\n" 继续切割
      3. 依次尝试 " "（空格）和 ""（逐字符）
      4. 直到所有块都不超过 chunk_size

    chunk_overlap 使相邻块之间有重叠内容，防止关键信息被切断在两个块的边界处。
    """
    print("=" * 60)
    print("Part 1: RecursiveCharacterTextSplitter 基本用法")
    print("=" * 60)

    # 加载示例文档
    file_path = os.path.join(SAMPLE_DOCS_DIR, "langchain_intro.txt")
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    original_text = docs[0].page_content

    print(f"原始文档长度：{len(original_text)} 个字符")
    print()

    # 创建分割器
    # chunk_size=100：每个块最多 100 个字符
    # chunk_overlap=20：相邻块共享 20 个字符的重叠内容
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        # length_function 默认是 len()，按字符数计算
        length_function=len,
    )

    # split_documents() 接受 List[Document]，返回切割后的 List[Document]
    # 每个子块继承原始 Document 的 metadata，并添加位置信息
    chunks = splitter.split_documents(docs)

    print(f"分块参数：chunk_size=100, chunk_overlap=20")
    print(f"分割后块数：{len(chunks)}")
    print()

    print("【各块内容】")
    for i, chunk in enumerate(chunks):
        print(f"  块 {i + 1} (长度 {len(chunk.page_content)} 字符):")
        # 显示每块内容，截断过长内容
        preview = chunk.page_content.replace("\n", " ").strip()
        print(f"    {preview[:80]}{'...' if len(preview) > 80 else ''}")
    print()


# ============================================================
# Part 2: 对比不同 chunk_size 的效果
# ============================================================

def part2_compare_chunk_sizes():
    """
    chunk_size 的选择直接影响 RAG 系统的检索质量：

    太小（如 50）：
      - 块数多，但每块信息量少
      - 检索到的片段可能缺乏上下文
      - 嵌入向量难以捕捉完整语义

    太大（如 500）：
      - 块数少，每块信息丰富
      - 但嵌入向量包含太多信息，检索精度下降
      - 输入给 LLM 的 context 更大，成本更高

    通常推荐：chunk_size=500~1000，根据文档类型和模型能力调整。
    """
    print("=" * 60)
    print("Part 2: 对比不同 chunk_size 对分块数量的影响")
    print("=" * 60)

    file_path = os.path.join(SAMPLE_DOCS_DIR, "langchain_intro.txt")
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    text_length = len(docs[0].page_content)

    print(f"原始文档长度：{text_length} 个字符")
    print()

    # 对比三种 chunk_size
    configs = [
        {"chunk_size": 50,  "chunk_overlap": 10},
        {"chunk_size": 200, "chunk_overlap": 40},
        {"chunk_size": 500, "chunk_overlap": 100},
    ]

    print(f"{'chunk_size':>12} {'chunk_overlap':>14} {'分块数量':>10} {'平均块长':>10}")
    print("-" * 52)

    for cfg in configs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )
        chunks = splitter.split_documents(docs)
        avg_len = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0

        print(
            f"{cfg['chunk_size']:>12} "
            f"{cfg['chunk_overlap']:>14} "
            f"{len(chunks):>10} "
            f"{avg_len:>10.1f}"
        )

    print()
    print("观察：chunk_size 越小 → 分块越多；chunk_size 越大 → 分块越少")
    print()


# ============================================================
# Part 3: 演示 chunk_overlap 的作用
# ============================================================

def part3_demonstrate_overlap():
    """
    chunk_overlap 是防止语义被切断的关键参数。

    示例：假设文本是 "ABCDEFGHIJ"，chunk_size=4，chunk_overlap=2：
      块 1：ABCD
      块 2：CDEF   ← CD 是重叠部分
      块 3：EFGH
      块 4：GHIJ

    重叠确保即使关键信息跨越两个块的边界，
    至少有一个块能完整地包含它。
    """
    print("=" * 60)
    print("Part 3: 演示 chunk_overlap —— 相邻块共享内容")
    print("=" * 60)

    file_path = os.path.join(SAMPLE_DOCS_DIR, "langchain_intro.txt")
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    # 用较大的 chunk_overlap 使重叠更明显
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=50,  # 重叠 50 个字符
    )
    chunks = splitter.split_documents(docs)

    print(f"分块参数：chunk_size=150, chunk_overlap=50")
    print(f"分割后块数：{len(chunks)}")
    print()

    # 展示前 3 个相邻块，验证重叠内容
    display_count = min(3, len(chunks))
    print(f"【展示前 {display_count} 个块，观察相邻块的重叠内容】")
    print()

    for i in range(display_count):
        chunk_text = chunks[i].page_content.replace("\n", " ").strip()
        print(f"  块 {i + 1} (长度 {len(chunks[i].page_content)}):")
        print(f"    {chunk_text[:120]}{'...' if len(chunk_text) > 120 else ''}")
        print()

    # 验证相邻块之间确实存在重叠
    if len(chunks) >= 2:
        text1 = chunks[0].page_content
        text2 = chunks[1].page_content

        # 取第一块的末尾 50 字符，检查第二块是否以此开头（或包含此内容）
        tail_of_chunk1 = text1[-50:].strip()
        overlap_found = tail_of_chunk1[:20] in text2 if tail_of_chunk1 else False

        print(f"  验证：块 1 末尾内容是否出现在块 2 中？{'是 ✓' if overlap_found else '未检测到（可能在分隔符处切割）'}")
        print()


# ============================================================
# main
# ============================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     08_text_splitters.py —— Text Splitters 演示          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    part1_basic_splitting()
    part2_compare_chunk_sizes()
    part3_demonstrate_overlap()

    print("=" * 60)
    print("Text Splitters 演示完毕！")
    print()
    print("关键要点：")
    print("  • RecursiveCharacterTextSplitter 是最常用的分割器")
    print("  • chunk_size 控制每块大小，需根据任务场景调整")
    print("  • chunk_overlap 防止关键信息被切断在块的边界处")
    print("  • 分块质量直接影响 RAG 系统的检索准确性")
    print("=" * 60)


if __name__ == "__main__":
    main()
