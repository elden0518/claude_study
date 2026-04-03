import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
主题：Document Loaders —— 把各种来源的文档加载成 LangChain 文档对象

学习目标：
  1. 理解 Document 对象结构（page_content + metadata）
  2. 掌握 TextLoader 加载本地文本文件
  3. 掌握 DirectoryLoader 批量加载目录
  4. 掌握 WebBaseLoader 加载网页内容
  5. 理解 metadata 的作用（来源追踪）

核心概念：
  Document = {page_content: str, metadata: dict}
  所有 Loader 返回 List[Document]
  metadata 记录文档来源，RAG 中用于显示"答案来自哪里"

前置知识：已完成 01_lc_basics/ 全部 demo
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader, WebBaseLoader

load_dotenv()

# 获取当前脚本所在目录，用于构造 sample_docs 的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DOCS_DIR = os.path.join(SCRIPT_DIR, "sample_docs")


# ============================================================
# Part 1: TextLoader —— 加载单个文本文件
# ============================================================

def part1_text_loader():
    """
    TextLoader：最简单的 Loader，加载单个本地文本文件。

    返回 List[Document]，通常只有 1 个 Document。
    Document 的 metadata 会包含 "source" 字段，记录文件路径。
    """
    print("=" * 60)
    print("Part 1: TextLoader —— 加载单个文本文件")
    print("=" * 60)

    # 构造目标文件的绝对路径
    file_path = os.path.join(SAMPLE_DOCS_DIR, "python_intro.txt")

    # 初始化 TextLoader
    # encoding="utf-8" 确保中文字符正确读取
    loader = TextLoader(file_path, encoding="utf-8")

    # 调用 load() 返回 List[Document]
    docs = loader.load()

    print(f"加载文件：{file_path}")
    print(f"文档数量：{len(docs)}")
    print()

    # 查看第一个 Document 的结构
    doc = docs[0]
    print("【page_content 前 100 个字符】")
    print(doc.page_content[:100])
    print()

    print("【metadata】")
    print(doc.metadata)
    # metadata 示例：{'source': '/path/to/python_intro.txt'}
    # 在 RAG 中，这个 source 字段用于告诉用户答案来自哪个文件
    print()


# ============================================================
# Part 2: DirectoryLoader —— 批量加载目录下的所有文件
# ============================================================

def part2_directory_loader():
    """
    DirectoryLoader：批量加载目录下匹配 glob 模式的所有文件。

    适合 RAG 场景中需要一次性加载整个知识库的情况。
    每个文件会生成对应的 Document，metadata.source 记录各自路径。
    """
    print("=" * 60)
    print("Part 2: DirectoryLoader —— 批量加载目录")
    print("=" * 60)

    # glob="**/*.txt" 递归匹配所有子目录中的 .txt 文件
    # loader_cls=TextLoader 指定用 TextLoader 处理每个匹配文件
    loader = DirectoryLoader(
        SAMPLE_DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,  # 关闭进度条，避免输出混乱
    )

    docs = loader.load()

    print(f"加载目录：{SAMPLE_DOCS_DIR}")
    print(f"共加载文档数量：{len(docs)}")
    print()

    # 打印每个文档的 metadata，展示来源追踪能力
    print("【各文档的 metadata.source】")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        # 只显示文件名部分，路径可能很长
        filename = os.path.basename(source)
        print(f"  文档 {i + 1}: {filename}")
        print(f"    内容预览: {doc.page_content[:50].strip()}...")
    print()


# ============================================================
# Part 3: WebBaseLoader —— 加载网页内容
# ============================================================

def part3_web_loader():
    """
    WebBaseLoader：从 URL 加载网页内容。

    底层使用 requests + BeautifulSoup 解析 HTML，
    自动提取正文文本，过滤掉大部分 HTML 标签。

    注意：需要网络连接，且网页结构各异，内容提取效果因站而异。
    """
    print("=" * 60)
    print("Part 3: WebBaseLoader —— 加载网页内容")
    print("=" * 60)

    url = "https://www.python.org"
    print(f"目标 URL：{url}")
    print("（如网络不通，将跳过此部分）")
    print()

    try:
        # WebBaseLoader 接受单个 URL 字符串或 URL 列表
        loader = WebBaseLoader(url)

        # load() 会发起 HTTP 请求，解析 HTML，返回 List[Document]
        docs = loader.load()

        print(f"加载文档数量：{len(docs)}")
        print()

        doc = docs[0]
        print("【page_content 前 200 个字符】")
        # 网页内容可能包含大量空白，先做简单清理
        content_preview = " ".join(doc.page_content.split())[:200]
        print(content_preview)
        print()

        print("【metadata】")
        print(doc.metadata)
        # metadata 通常包含：source（URL）、title（页面标题）等
        print()

    except Exception as e:
        print(f"[跳过] WebBaseLoader 加载失败，原因：{e}")
        print("提示：请检查网络连接，或稍后重试。")
        print()


# ============================================================
# main
# ============================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     07_document_loaders.py —— Document Loaders 演示      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    part1_text_loader()
    part2_directory_loader()
    part3_web_loader()

    print("=" * 60)
    print("Document Loaders 演示完毕！")
    print()
    print("关键要点：")
    print("  • Document = page_content（文本）+ metadata（来源信息）")
    print("  • TextLoader      → 加载单个文本文件")
    print("  • DirectoryLoader → 批量加载目录下所有匹配文件")
    print("  • WebBaseLoader   → 加载网页（需要网络）")
    print("  • metadata.source 是 RAG 中显示答案来源的关键字段")
    print("=" * 60)


if __name__ == "__main__":
    main()
