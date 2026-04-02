"""
主题：MCP（Model Context Protocol）入门 —— MCP 是什么？

学习目标：
  1. 理解 MCP 的架构和它解决的问题
  2. 了解 MCP Server 的代码结构（fastmcp 用法）
  3. 理解 MCP 工具与 Claude tool_use 的关系
  4. 通过端到端示例感受完整的 MCP 工作流程
  5. 了解如何在 Claude Desktop / Claude Code 中配置 MCP

前置知识：
  - 已完成 01_basics/07_tool_use.py（理解 Claude tool_use 机制）
  - 已安装依赖：pip install anthropic fastmcp python-dotenv

课程顺序：这是 03_mcp_skill_command 模块的第一个文件。

# =============================================================================
# MCP 架构图
# =============================================================================
#
#  Claude (LLM)
#      ↕ Messages API（tool_use / tool_result）
#  MCP Client（你的应用）
#      ↕ MCP Protocol（stdio / SSE）
#  MCP Server（工具提供方）
#      ↕ 实际执行
#  外部工具（文件系统 / 数据库 / API 等）
#
# -----------------------------------------------------------------------------
# MCP vs 直接 Tool Use 的区别
# -----------------------------------------------------------------------------
#
# 直接 Tool Use（07_tool_use.py 的做法）：
#   - 工具定义：在你的代码里写 JSON Schema
#   - 工具执行：在你的代码里写 Python 函数
#   - 优点：简单直接，适合小项目
#   - 缺点：工具与应用耦合，无法跨应用复用
#
# MCP（本文件介绍的做法）：
#   - 工具定义 + 执行：在独立的 MCP Server 进程中
#   - Client 通过标准协议（stdio / SSE）与 Server 通信
#   - 优点：工具可复用（Claude Desktop、Claude Code、你自己的应用都能用同一个 Server）
#   - 缺点：需要额外管理进程，架构略复杂
#
# 简单类比：
#   直接 Tool Use  ≈  把工具代码直接复制进你的项目
#   MCP            ≈  把工具发布成一个微服务，任何人都能调用
#
# =============================================================================
"""

# ── 0. Windows 控制台编码修复 ──────────────────────────────────────────────────
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── 1. 导入所需模块 ────────────────────────────────────────────────────────────
import os
import json
import textwrap

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "ppio/pa/claude-sonnet-4-6"

client = anthropic.Anthropic()


# =============================================================================
# Part 1：MCP 架构说明
# =============================================================================

def show_mcp_architecture():
    """打印 MCP 架构说明和核心概念。"""

    print("=" * 60)
    print("Part 1：MCP 是什么？")
    print("=" * 60)

    print("""
MCP（Model Context Protocol）是 Anthropic 推出的开放标准，
让 AI 应用能够以统一的方式连接外部工具和数据源。

┌─────────────────────────────────────────────────────┐
│                   MCP 架构图                         │
│                                                     │
│   Claude (LLM)                                      │
│       ↕  Messages API（tool_use / tool_result）      │
│   MCP Client（你的应用 / Claude Desktop）            │
│       ↕  MCP Protocol（stdio / SSE）                 │
│   MCP Server（工具提供方，独立进程）                   │
│       ↕  实际执行                                     │
│   外部工具（文件系统 / 数据库 / GitHub API 等）        │
└─────────────────────────────────────────────────────┘

核心概念：
  • MCP Server  — 提供工具的独立程序，用 fastmcp / 官方 SDK 编写
  • MCP Client  — 连接 Server 的一方，如 Claude Desktop、Claude Code
  • Transport   — 通信方式：stdio（子进程管道）或 SSE（HTTP 流）
  • Tool        — Server 暴露给 Claude 调用的函数

为什么需要 MCP？
  传统做法：每个应用自己实现工具调用逻辑，重复劳动。
  MCP：一次编写 Server，Claude Desktop / Claude Code / 你的应用
       都能直接使用，彻底复用。
""")


# =============================================================================
# Part 2：MCP Server 代码结构（fastmcp 用法）
# =============================================================================

def demo_mcp_server_code():
    """展示 MCP Server 的完整代码结构。

    这段代码如果保存为独立文件（如 my_mcp_server.py），
    就能作为真正的 MCP Server 被 Claude Desktop 加载。
    """

    print("=" * 60)
    print("Part 2：MCP Server 代码结构（fastmcp）")
    print("=" * 60)

    # ── MCP Server 示例代码 ────────────────────────────────────────────────────
    # 下面是一个完整的 MCP Server，包含两个工具：add 和 greet
    # 注意：这里用字符串展示，说明其结构；实际运行见 Part 4

    server_code = textwrap.dedent("""\
        # ── my_mcp_server.py ──────────────────────────────────────────────────────
        from fastmcp import FastMCP

        # 1. 创建 MCP Server 实例，名称 "demo" 会显示在客户端工具列表里
        mcp = FastMCP("demo")


        # 2. 用 @mcp.tool() 装饰器注册工具
        #    fastmcp 会自动从函数签名和 docstring 生成 JSON Schema
        @mcp.tool()
        def add(a: int, b: int) -> int:
            \"\"\"将两个整数相加，返回它们的和。\"\"\"
            return a + b


        @mcp.tool()
        def greet(name: str) -> str:
            \"\"\"生成一条个性化问候语。\"\"\"
            return f"你好，{name}！欢迎使用 MCP 工具。"


        # 3. 启动 Server（以 stdio 模式运行，等待 Client 连接）
        if __name__ == "__main__":
            mcp.run()          # 默认 transport=stdio
        # ─────────────────────────────────────────────────────────────────────────
    """)

    print("\nMCP Server 完整代码（保存为独立 .py 文件即可使用）：\n")
    print(server_code)

    print("""
关键点解析：
  ① FastMCP("demo")     — 创建 Server，"demo" 是 Server 名称
  ② @mcp.tool()         — 自动把函数注册为 MCP 工具
                           函数签名      → 参数 Schema（自动推断）
                           docstring     → 工具描述（Claude 据此决定何时调用）
                           返回值类型    → 输出格式
  ③ mcp.run()           — 启动监听，等待 Client 通过 stdio 连接

fastmcp vs 官方 MCP SDK：
  fastmcp  — 高级封装，代码极简，推荐初学者使用
  mcp      — 官方底层 SDK，更灵活但更啰嗦
""")


# =============================================================================
# Part 3：同进程内直接测试工具逻辑
# =============================================================================

def demo_mcp_tools_directly():
    """在同一进程内直接调用工具函数，验证逻辑正确性。

    真实场景中工具运行在独立的 MCP Server 进程里，
    但逻辑完全相同 —— 这里直接调用函数来快速验证。
    """

    print("=" * 60)
    print("Part 3：直接测试工具函数（验证逻辑）")
    print("=" * 60)

    # ── 工具函数定义（与 MCP Server 中完全一致）─────────────────────────────────
    # 在真实 MCP Server 里，这些函数会被 @mcp.tool() 装饰
    # 这里去掉装饰器，直接调用，效果完全相同

    def add(a: int, b: int) -> int:
        """将两个整数相加，返回它们的和。"""
        return a + b

    def greet(name: str) -> str:
        """生成一条个性化问候语。"""
        return f"你好，{name}！欢迎使用 MCP 工具。"

    # ── 测试工具 ──────────────────────────────────────────────────────────────
    print("\n直接调用工具函数（模拟 MCP Server 执行工具）：\n")

    test_cases = [
        ("add",   {"a": 3, "b": 7}),
        ("add",   {"a": 100, "b": -25}),
        ("greet", {"name": "Alice"}),
        ("greet", {"name": "MCP 初学者"}),
    ]

    for tool_name, args in test_cases:
        if tool_name == "add":
            result = add(**args)
        else:
            result = greet(**args)

        print(f"  工具: {tool_name:<8} 参数: {json.dumps(args, ensure_ascii=False):<30} 结果: {result}")

    print("\n工具逻辑验证通过！")

    print("""
注意：
  MCP Client 调用工具时，Server 端执行的代码与这里完全一样。
  区别只是「如何传递参数和结果」——通过 stdio 管道中的 JSON 消息传递，
  而不是直接的 Python 函数调用。
""")


# =============================================================================
# Part 4：将 MCP 工具转为 Claude tool_use 格式，端到端调用 Claude
# =============================================================================

def demo_claude_with_mcp_tools():
    """把 MCP Server 暴露的工具转换为 Claude tool_use 格式，完整调用 Claude。

    这展示了 MCP Client 的核心工作：
      1. 从 MCP Server 获取工具列表（这里手动构造，等效于 server.list_tools()）
      2. 把工具列表传给 Claude
      3. 执行 Claude 请求的工具调用
      4. 把结果还给 Claude，得到最终回答
    """

    print("=" * 60)
    print("Part 4：Claude + MCP 工具端到端调用")
    print("=" * 60)

    # ── Step 1：定义工具执行函数（模拟 MCP Server 端）───────────────────────────
    # 真实场景：MCP Client 通过 stdio 向 Server 发 CallToolRequest，Server 返回结果
    # 这里：直接在本进程执行，逻辑完全等效

    def execute_tool(tool_name: str, tool_input: dict):
        """执行工具并返回结果（模拟 MCP Server 处理 CallToolRequest）。"""
        if tool_name == "add":
            a = tool_input["a"]
            b = tool_input["b"]
            return a + b
        elif tool_name == "greet":
            name = tool_input["name"]
            return f"你好，{name}！欢迎使用 MCP 工具。"
        else:
            raise ValueError(f"未知工具：{tool_name}")

    # ── Step 2：构造工具列表（模拟 MCP Client 调用 server.list_tools()）─────────
    # 真实 MCP Client 代码：
    #   async with stdio_client(server_params) as (read, write):
    #       async with ClientSession(read, write) as session:
    #           tools = await session.list_tools()
    #
    # fastmcp 会根据函数签名自动生成这个 JSON Schema，
    # 这里手动写出来，帮助你理解 MCP Server 实际暴露的内容

    mcp_tools_as_claude_format = [
        {
            "name": "add",
            "description": "将两个整数相加，返回它们的和。",
            "input_schema": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "第一个整数"},
                    "b": {"type": "integer", "description": "第二个整数"},
                },
                "required": ["a", "b"],
            },
        },
        {
            "name": "greet",
            "description": "生成一条个性化问候语。",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "要问候的人的名字"},
                },
                "required": ["name"],
            },
        },
    ]

    print(f"\n从 MCP Server 获取到 {len(mcp_tools_as_claude_format)} 个工具：")
    for tool in mcp_tools_as_claude_format:
        print(f"  • {tool['name']}: {tool['description']}")

    # ── Step 3：向 Claude 发送请求，带上 MCP 工具列表 ───────────────────────────
    user_message = "请帮我计算 42 加 58 等于多少，然后用中文问候一下'世界'。"

    print(f"\n用户提问：{user_message}")
    print("\n正在调用 Claude（携带 MCP 工具）...")

    messages = [{"role": "user", "content": user_message}]

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=mcp_tools_as_claude_format,   # MCP 工具以 Claude tool_use 格式传入
        messages=messages,
    )

    # ── Step 4：处理工具调用循环 ───────────────────────────────────────────────
    # Claude 可能连续调用多个工具，需要循环处理直到 stop_reason == "end_turn"
    tool_call_count = 0

    while response.stop_reason == "tool_use":
        tool_call_count += 1
        print(f"\n── 第 {tool_call_count} 轮工具调用 ──────────────────────────────")

        # 把 Claude 的回复加入消息历史
        messages.append({"role": "assistant", "content": response.content})

        # 收集所有工具调用结果
        tool_results = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name  = block.name
            tool_input = block.input
            tool_id    = block.id

            print(f"  Claude 调用工具: {tool_name}")
            print(f"  参数: {json.dumps(tool_input, ensure_ascii=False)}")

            # 执行工具（模拟 MCP Client 向 MCP Server 发送 CallToolRequest）
            result = execute_tool(tool_name, tool_input)

            print(f"  结果: {result}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": str(result),
            })

        # 把工具结果加入消息历史，继续对话
        messages.append({"role": "user", "content": tool_results})

        # 再次调用 Claude，带上工具执行结果
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=mcp_tools_as_claude_format,
            messages=messages,
        )

    # ── Step 5：打印最终回答 ──────────────────────────────────────────────────
    final_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            final_text += block.text

    print(f"\n── Claude 的最终回答 ─────────────────────────────────")
    print(final_text)
    print(f"\n共调用了 {tool_call_count} 次工具")
    print(f"Token 消耗：输入 {response.usage.input_tokens}，输出 {response.usage.output_tokens}")

    # ── 小结：MCP Client 的核心职责 ───────────────────────────────────────────
    print("""
MCP Client 的核心工作（本函数完整演示）：
  1. list_tools()      — 从 Server 获取工具列表，转换为 Claude 格式
  2. 调用 Claude       — 携带工具列表，发送用户消息
  3. call_tool()       — 收到 tool_use 时，向 Server 发送工具调用请求
  4. 返回结果给 Claude — 用 tool_result 继续对话
  5. 循环直到 end_turn — 得到最终自然语言回答

真实 MCP Client（使用 mcp 官方库）的写法与此结构完全一致，
只是第 1、3 步改为通过 stdio/SSE 与 Server 进程通信。
""")


# =============================================================================
# Part 5：Claude Desktop / Claude Code 配置说明
# =============================================================================

def show_desktop_config():
    """展示在 Claude Desktop 和 Claude Code 中配置 MCP Server 的方法。"""

    print("=" * 60)
    print("Part 5：在 Claude Desktop / Claude Code 中使用 MCP")
    print("=" * 60)

    config_example = textwrap.dedent("""\
        // 文件路径：~/.claude/claude_desktop_config.json
        // （Windows：%APPDATA%\\Claude\\claude_desktop_config.json）
        {
          "mcpServers": {
            "demo": {
              "command": "python",
              "args": ["D:/claude_project/claude_study/my_mcp_server.py"]
            }
          }
        }
    """)

    print("""
配置方式（以 Claude Desktop 为例）：

  1. 把 MCP Server 代码保存为独立 .py 文件，例如 my_mcp_server.py
  2. 修改配置文件，添加 Server 条目：
""")
    print(config_example)

    print("""
配置字段说明：
  "demo"      — Server 名称（自定义，显示在 Claude 工具列表中）
  "command"   — 启动 Server 的可执行程序（python / node / uvx 等）
  "args"      — 传给可执行程序的参数列表

配置后重启 Claude Desktop，即可在对话框看到 Server 提供的工具。

Claude Code（CLI）配置方式相同：
  ~/.claude/claude_desktop_config.json（同一个配置文件）
  或在项目目录下创建 .claude/mcp.json

常见 MCP Server 生态（可直接使用，无需自己编写）：
  • @modelcontextprotocol/server-filesystem  — 本地文件操作
  • @modelcontextprotocol/server-github      — GitHub 仓库操作
  • @modelcontextprotocol/server-sqlite      — SQLite 数据库查询
  • mcp-server-fetch                         — 网页抓取
  更多：https://github.com/modelcontextprotocol/servers
""")


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("MCP（Model Context Protocol）入门")
    print("=" * 60)

    # Part 1：MCP 架构和概念介绍
    show_mcp_architecture()

    # Part 2：MCP Server 代码结构（fastmcp 用法）
    demo_mcp_server_code()

    # Part 3：同进程测试工具函数
    demo_mcp_tools_directly()

    # Part 4：Claude + MCP 工具端到端调用（核心演示）
    demo_claude_with_mcp_tools()

    # Part 5：Claude Desktop / Claude Code 配置说明
    show_desktop_config()

    print("=" * 60)
    print("MCP 入门学习完成！")
    print("下一步：03_mcp_skill_command/12_mcp_custom_server.py")
    print("  → 学习如何编写并独立运行一个完整的 MCP Server")
    print("=" * 60)


if __name__ == "__main__":
    main()
