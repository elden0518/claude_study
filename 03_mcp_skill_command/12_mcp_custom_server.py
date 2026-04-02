"""
主题：从零编写完整的自定义 MCP Server

学习目标：
  1. 掌握 FastMCP 的核心参数：name 和 instructions
  2. 理解 @mcp.tool() 和 @mcp.resource() 的区别和使用场景
  3. 学会直接测试工具函数（不启动服务器）
  4. 掌握在 Claude Desktop 和 Claude Code 中配置 MCP Server 的方法

前置知识：
  - 已完成 03_mcp_skill_command/11_mcp_intro.py（了解 MCP 基础架构）

本文件结构：
  Part 1 — Server 架构说明（FastMCP name / instructions）
  Part 2 — @mcp.tool() 用法详解
  Part 3 — @mcp.resource() 用法详解（与 Tool 的区别）
  Part 4 — 直接测试 12_mcp_server_demo.py 中的工具函数
  Part 5 — Claude Desktop / Claude Code 完整配置指南
"""

# ── 0. Windows 控制台编码修复 ──────────────────────────────────────────────────
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── 1. 导入 ────────────────────────────────────────────────────────────────────
import os
import json
import textwrap


# =============================================================================
# Part 1：Server 架构说明
# =============================================================================

def show_server_architecture():
    """打印 FastMCP Server 的核心结构和参数说明。"""

    print("=" * 60)
    print("Part 1：自定义 MCP Server 架构说明")
    print("=" * 60)

    print("""
一个完整的 MCP Server 由三部分组成：

  ┌─────────────────────────────────────────────────────┐
  │  from fastmcp import FastMCP                        │
  │                                                     │
  │  mcp = FastMCP(                                     │
  │      name="学习助手",          ← Server 显示名称    │
  │      instructions="..."       ← 使用说明（给 AI）   │
  │  )                                                  │
  │                                                     │
  │  @mcp.tool()                                        │
  │  def my_tool(...): ...        ← 工具：Claude 调用   │
  │                                                     │
  │  @mcp.resource("uri://...")                         │
  │  def my_resource(): ...       ← 资源：Claude 读取   │
  │                                                     │
  │  if __name__ == "__main__":                         │
  │      mcp.run()                ← 启动 stdio Server   │
  └─────────────────────────────────────────────────────┘

FastMCP() 的核心参数：

  name（可选，默认 None）：
    • Server 的显示名称
    • 出现在 Claude 的工具列表标题中
    • 建议用描述性名称，如 "文件工具箱"、"数据库助手"

  instructions（可选，默认 None）：
    • 给 Claude 的系统级使用说明
    • Claude 会在决定是否使用这个 Server 时参考此描述
    • 建议描述 Server 的功能范围和典型使用场景

  其他常用参数（进阶）：
    version    — Server 版本号
    lifespan   — 生命周期钩子（如初始化数据库连接）
""")


# =============================================================================
# Part 2：@mcp.tool() 用法
# =============================================================================

def show_tool_usage():
    """展示 @mcp.tool() 装饰器的用法和自动生成机制。"""

    print("=" * 60)
    print("Part 2：@mcp.tool() 用法详解")
    print("=" * 60)

    print("""
@mcp.tool() 会把一个普通 Python 函数注册为 MCP 工具。
fastmcp 自动从函数签名和 docstring 生成完整的工具定义。

─── 示例代码 ────────────────────────────────────────────────

@mcp.tool()
def calculate(expression: str) -> str:
    \"\"\"安全计算数学表达式。

    支持加减乘除、括号和常用数学运算。
    例如：'2 + 3 * 4'、'(10 - 2) / 4'
    \"\"\"
    allowed_chars = set("0123456789+-*/()., ")
    if not all(c in allowed_chars for c in expression):
        return "错误：表达式含有不允许的字符。"
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"

─────────────────────────────────────────────────────────────

fastmcp 自动生成的工具定义（Claude 收到的格式）：
  {
    "name": "calculate",
    "description": "安全计算数学表达式。\\n\\n支持加减乘除...",
    "input_schema": {
      "type": "object",
      "properties": {
        "expression": {"type": "string"}
      },
      "required": ["expression"]
    }
  }

自动生成规则：
  函数名         → 工具名称（name）
  docstring      → 工具描述（description）—— Claude 据此决定何时调用
  类型注解       → 参数 Schema（int/str/bool/Optional 等均支持）
  无默认值参数   → required 列表
  有默认值参数   → 可选参数（不在 required 中）

@mcp.tool() 的可选参数（进阶）：
  @mcp.tool(name="my_calc")           ← 自定义工具名（覆盖函数名）
  @mcp.tool(description="...")        ← 自定义描述（覆盖 docstring）
  @mcp.tool(tags={"math", "utils"})   ← 添加标签（便于分类过滤）
""")


# =============================================================================
# Part 3：@mcp.resource() 用法
# =============================================================================

def show_resource_usage():
    """展示 @mcp.resource() 与 @mcp.tool() 的区别。"""

    print("=" * 60)
    print("Part 3：@mcp.resource() 用法详解")
    print("=" * 60)

    print("""
MCP 中 Tool 和 Resource 的本质区别：

  Tool（工具）：
    • Claude 主动调用，执行某个操作
    • 可以有副作用（写文件、发邮件、查数据库）
    • 参数由 Claude 根据上下文决定
    • 使用 @mcp.tool() 注册

  Resource（资源）：
    • 提供数据，类似「只读 API 端点」
    • 通常无副作用，幂等（多次读取结果一致）
    • 通过 URI 寻址，Claude 按需读取
    • 使用 @mcp.resource("uri://...") 注册

─── 静态资源（固定 URI）────────────────────────────────────

@mcp.resource("notes://list")
def list_notes() -> str:
    \"\"\"列出所有可用笔记的 ID 和摘要。\"\"\"
    return json.dumps({"notes": ["python", "mcp", "fastmcp"]})

  URI：notes://list  （固定，直接映射到这个函数）

─── 动态资源（URI 模板）────────────────────────────────────

@mcp.resource("notes://{note_id}")
def get_note(note_id: str) -> str:
    \"\"\"获取指定 ID 的笔记完整内容。\"\"\"
    notes = {"python": "Python 是...", "mcp": "MCP 是..."}
    return notes.get(note_id, f"笔记 '{note_id}' 不存在")

  URI 模板：notes://{note_id}
  实际 URI：notes://python  →  调用 get_note(note_id="python")
            notes://mcp    →  调用 get_note(note_id="mcp")

─────────────────────────────────────────────────────────────

何时用 Tool，何时用 Resource？

  用 Tool：  计算、搜索、写入、调用外部 API、有参数选择的操作
  用 Resource：配置读取、数据目录、文档库、只读的结构化数据

  判断方法：「这个操作有副作用吗？参数是 Claude 决定的吗？」
            → 是  → Tool
            → 否（只是读取固定地址的数据）  → Resource
""")


# =============================================================================
# Part 4：直接测试 12_mcp_server_demo.py 中的工具函数
# =============================================================================

def demo_test_tools_directly():
    """导入并直接调用 Server 文件中的工具函数，验证逻辑正确性。

    真实的 MCP Server 启动后，工具函数运行在独立进程里。
    这里直接导入函数来测试逻辑，避免启动进程，便于调试。
    """

    print("=" * 60)
    print("Part 4：直接测试 12_mcp_server_demo.py 中的工具函数")
    print("=" * 60)

    # 确定 Server 文件路径（与本文件同目录）
    server_file_dir = os.path.dirname(os.path.abspath(__file__))

    # 由于文件名以数字开头（12_mcp_server_demo.py），不能直接用 import 语句。
    # 用 importlib 按文件路径加载模块，效果与 import 完全一致。
    # 注意：导入时 if __name__ == "__main__" 不会执行，Server 不会启动。
    import importlib.util

    server_file_path = os.path.join(server_file_dir, "12_mcp_server_demo.py")
    spec = importlib.util.spec_from_file_location("mcp_server_demo", server_file_path)
    server_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_module)

    calculate      = server_module.calculate
    read_file      = server_module.read_file
    list_directory = server_module.list_directory
    list_notes     = server_module.list_notes
    get_note       = server_module.get_note

    print("\n成功导入 12_mcp_server_demo.py 中的函数（Server 未启动）\n")

    # ── 测试 calculate ────────────────────────────────────────────────────────
    print("── 测试 calculate 工具 ──────────────────────────────")
    calc_cases = [
        "2 + 3 * 4",
        "(10 - 2) / 4",
        "100 * 0.15",
        "2 ** 10",
        "import os",       # 含非法字符，应被拦截
    ]
    for expr in calc_cases:
        result = calculate(expr)
        print(f"  calculate({repr(expr)!s:30}) → {result}")

    # ── 测试 list_directory ───────────────────────────────────────────────────
    print("\n── 测试 list_directory 工具 ────────────────────────")
    dir_result = list_directory(server_file_dir)
    dir_data = json.loads(dir_result)
    print(f"  list_directory(server_dir)")
    print(f"  路径：{dir_data['path']}")
    print(f"  文件数：{dir_data['count']}")
    print(f"  前5个条目：")
    for entry in dir_data["entries"][:5]:
        icon = "[DIR]" if entry["type"] == "dir" else "[FILE]"
        size = f" {entry.get('size_kb', '')} KB" if entry["type"] == "file" else ""
        print(f"    {icon} {entry['name']}{size}")

    # ── 测试 read_file ────────────────────────────────────────────────────────
    print("\n── 测试 read_file 工具 ─────────────────────────────")
    demo_server_path = os.path.join(server_file_dir, "12_mcp_server_demo.py")
    file_result = read_file(demo_server_path)
    # 只显示前3行（避免输出过多）
    lines = file_result.split("\n")
    print(f"  read_file('12_mcp_server_demo.py') 前4行：")
    for line in lines[:4]:
        print(f"    {line}")
    print(f"  ...（共 {len(lines)} 行）")

    # 测试不存在的文件
    missing = read_file("/不存在的/路径.txt")
    print(f"\n  read_file('/不存在的/路径.txt') → {missing}")

    # ── 测试 Resource ─────────────────────────────────────────────────────────
    print("\n── 测试 Resource 函数 ──────────────────────────────")
    notes_list = list_notes()
    notes_data = json.loads(notes_list)
    print(f"  list_notes() → 共 {notes_data['total']} 条笔记")
    for note in notes_data["notes"]:
        print(f"    id={note['id']!r}  预览：{note['preview']}")

    print()
    for note_id in ["python", "mcp", "不存在"]:
        content = get_note(note_id)
        preview = content[:40] + "..." if len(content) > 40 else content
        print(f"  get_note({note_id!r}) → {preview}")

    print("""
关键点：
  • 工具函数是普通 Python 函数，可直接导入和调用
  • @mcp.tool() / @mcp.resource() 只是「注册到 Server」的标记
  • 开发阶段直接测试函数逻辑，无需启动 Server 进程
  • 单元测试也可以用同样方式编写（参见 pytest 文档）
""")


# =============================================================================
# Part 5：Claude Desktop / Claude Code 完整配置指南
# =============================================================================

def show_configuration_guide():
    """打印在 Claude Desktop 和 Claude Code 中配置 MCP Server 的完整 JSON。"""

    print("=" * 60)
    print("Part 5：完整配置指南")
    print("=" * 60)

    # 计算 Server 文件的绝对路径（用于配置示例）
    server_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "12_mcp_server_demo.py")
    ).replace("\\", "/")

    python_exe = sys.executable.replace("\\", "/")

    # ── 配置文件路径 ──────────────────────────────────────────────────────────
    print("""
配置文件位置：

  Claude Desktop（macOS）：
    ~/Library/Application Support/Claude/claude_desktop_config.json

  Claude Desktop（Windows）：
    %APPDATA%\\Claude\\claude_desktop_config.json
    （通常是 C:\\Users\\<用户名>\\AppData\\Roaming\\Claude\\）

  Claude Code（CLI，全局）：
    ~/.claude/claude_desktop_config.json
    （与 Claude Desktop 共用同一个配置文件）
""")

    # ── 最小配置（使用系统 python）────────────────────────────────────────────
    minimal_config = {
        "mcpServers": {
            "学习助手": {
                "command": "python",
                "args": [server_file]
            }
        }
    }

    print("── 最小配置（使用系统 python）────────────────────────")
    print(json.dumps(minimal_config, ensure_ascii=False, indent=2))

    # ── 推荐配置（使用虚拟环境 python）───────────────────────────────────────
    venv_config = {
        "mcpServers": {
            "学习助手": {
                "command": python_exe,
                "args": [server_file],
                "env": {
                    "PYTHONPATH": os.path.dirname(server_file).replace("\\", "/")
                }
            }
        }
    }

    print("\n── 推荐配置（使用当前虚拟环境 python）─────────────────")
    print(json.dumps(venv_config, ensure_ascii=False, indent=2))

    # ── 配置字段说明 ───────────────────────────────────────────────────────────
    print("""
配置字段说明：

  "学习助手"   — Server 标识符（可自定义，在 Claude 中显示为工具分组名）
  "command"    — 启动 Server 的可执行程序
                  Python 项目  → "python" 或虚拟环境路径
                  Node.js 项目 → "node" 或 "npx"
                  打包工具     → "uvx"（推荐，自动管理依赖）
  "args"       — 传给 command 的参数列表
  "env"        — 环境变量（可选），覆盖系统环境变量
                  常用：PYTHONPATH、API_KEY、DATABASE_URL

── 多 Server 配置示例 ──────────────────────────────────""")

    multi_config = {
        "mcpServers": {
            "学习助手": {
                "command": python_exe,
                "args": [server_file]
            },
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            },
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-token>"
                }
            }
        }
    }

    print(json.dumps(multi_config, ensure_ascii=False, indent=2))

    print("""
── 配置生效步骤 ────────────────────────────────────────

  1. 确认 Server 文件可以独立运行（测试命令见下方）
  2. 编辑配置文件，填入上面的 JSON
  3. 完全退出并重启 Claude Desktop / Claude Code
  4. 在对话中询问 Claude："你有哪些可用的工具？"
     或点击工具图标查看已加载的 MCP Server

── 验证 Server 可独立运行 ──────────────────────────────""")

    print(f"""
  # 测试 Server 是否能正常启动（启动后 Ctrl+C 退出）：
  {python_exe} {server_file}

  # 预期输出：
  #   [fastmcp] Starting MCP server "学习助手" with transport "stdio"
  #   （等待输入，说明 Server 已就绪）

── 常见问题排查 ────────────────────────────────────────

  Q: Claude Desktop 中看不到工具
  A: 检查 JSON 格式是否正确（括号、逗号）；重启 Claude Desktop

  Q: 工具调用失败 / Server 报错
  A: 在终端手动运行 Server，复现并调试错误；检查依赖是否安装

  Q: 使用虚拟环境，但 Claude Desktop 找不到 python
  A: 在 "command" 中填写 python.exe 的完整绝对路径

  Q: 想让多台机器共用同一个 Server
  A: 把 Server 改为 HTTP/SSE 模式：mcp.run(transport="sse")
     然后在配置中改为 "url": "http://your-server:8000/sse"
""")


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("从零编写完整的自定义 MCP Server")
    print("配套 Server 文件：12_mcp_server_demo.py")
    print("=" * 60)

    show_server_architecture()
    show_tool_usage()
    show_resource_usage()
    demo_test_tools_directly()
    show_configuration_guide()

    print("=" * 60)
    print("学习完成！")
    print()
    print("下一步：")
    print("  1. 修改 12_mcp_server_demo.py，添加你自己的工具")
    print("  2. 按 Part 5 配置 Claude Desktop，体验完整的 MCP 工作流")
    print("  3. 探索更多 MCP Server：https://github.com/modelcontextprotocol/servers")
    print("=" * 60)


if __name__ == "__main__":
    main()
