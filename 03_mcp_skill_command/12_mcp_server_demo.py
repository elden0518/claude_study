"""
一个完整的示例 MCP Server
可以通过 `python 12_mcp_server_demo.py` 启动
在 Claude Desktop 中配置后即可使用

用法：
  直接启动：python 12_mcp_server_demo.py
  配置说明：见 12_mcp_custom_server.py 的 Part 5
"""

from fastmcp import FastMCP
from typing import Optional
import json
import os

# =============================================================================
# 创建 MCP Server 实例
# =============================================================================
# name        — Server 的显示名称，出现在 Claude 的工具列表里
# instructions — 告诉 Claude 这个 Server 的用途和使用方式

mcp = FastMCP(
    name="学习助手",
    instructions="这是一个用于学习 Claude MCP 的示例服务器，提供计算、文件读取和目录列举工具。"
)


# =============================================================================
# 工具（Tool）：Claude 可以主动调用的函数
# =============================================================================

@mcp.tool()
def calculate(expression: str) -> str:
    """安全计算数学表达式。

    支持加减乘除、括号和常用数学运算。
    例如：'2 + 3 * 4'、'(10 - 2) / 4'
    """
    # 只允许包含数字和安全字符，防止代码注入
    allowed_chars = set("0123456789+-*/()., ")
    if not all(c in allowed_chars for c in expression):
        return f"错误：表达式含有不允许的字符。仅支持数字和 +-*/() 符号。"
    try:
        result = eval(expression)  # nosec — 已做字符白名单过滤
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


@mcp.tool()
def read_file(path: str) -> str:
    """读取文本文件内容并返回。

    path: 文件的绝对路径或相对路径（相对于当前工作目录）
    """
    try:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            return f"错误：文件不存在 — {abs_path}"
        if not os.path.isfile(abs_path):
            return f"错误：路径不是文件 — {abs_path}"
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        size_kb = os.path.getsize(abs_path) / 1024
        return f"[文件：{abs_path}  大小：{size_kb:.1f} KB]\n\n{content}"
    except Exception as e:
        return f"读取失败：{e}"


@mcp.tool()
def list_directory(path: str = ".") -> str:
    """列出目录中的文件和子目录。

    path: 要列举的目录路径，默认为当前工作目录（.）
    返回 JSON 格式的文件列表，包含文件名、类型和大小。
    """
    try:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            return f"错误：目录不存在 — {abs_path}"
        if not os.path.isdir(abs_path):
            return f"错误：路径不是目录 — {abs_path}"

        entries = []
        for name in sorted(os.listdir(abs_path)):
            full = os.path.join(abs_path, name)
            if os.path.isdir(full):
                entries.append({"name": name, "type": "dir", "size": None})
            else:
                size_kb = round(os.path.getsize(full) / 1024, 1)
                entries.append({"name": name, "type": "file", "size_kb": size_kb})

        return json.dumps(
            {"path": abs_path, "count": len(entries), "entries": entries},
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        return f"列举失败：{e}"


# =============================================================================
# 资源（Resource）：Claude 可以读取的静态或动态数据
# =============================================================================
# 与 Tool 的区别：
#   Tool     — Claude 主动调用，执行操作，可能有副作用
#   Resource — 提供数据，类似「只读 API」，Claude 按需读取

# 内存中的笔记存储（演示用）
_NOTES: dict[str, str] = {
    "python": "Python 是一门简洁、易读的编程语言，适合初学者入门。",
    "mcp": "MCP（Model Context Protocol）让 AI 与外部工具以标准协议通信。",
    "fastmcp": "fastmcp 是 MCP Server 的高级封装，用极少代码即可创建工具和资源。",
}


@mcp.resource("notes://list")
def list_notes() -> str:
    """列出所有可用笔记的 ID 和摘要。"""
    summaries = [
        {"id": note_id, "preview": text[:30] + "..."}
        for note_id, text in _NOTES.items()
    ]
    return json.dumps(
        {"total": len(_NOTES), "notes": summaries},
        ensure_ascii=False,
        indent=2,
    )


@mcp.resource("notes://{note_id}")
def get_note(note_id: str) -> str:
    """获取指定 ID 的笔记完整内容。

    note_id: 笔记标识符，例如 'python'、'mcp'、'fastmcp'
    """
    if note_id not in _NOTES:
        available = ", ".join(_NOTES.keys())
        return f"笔记 '{note_id}' 不存在。可用笔记：{available}"
    return _NOTES[note_id]


# =============================================================================
# 启动入口
# =============================================================================

if __name__ == "__main__":
    # 以 stdio 模式启动 MCP Server
    # Claude Desktop / Claude Code 通过子进程管道与此 Server 通信
    mcp.run()
