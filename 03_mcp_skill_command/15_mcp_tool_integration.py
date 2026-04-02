"""
MCP 工具集成完整链路演示
========================
本文件是 MCP 层的最后一个 demo，展示 MCP 工具与 Claude tool_use 的完整集成链路，
并对比 Task 7（02_advanced/07_tool_use.py）中直接 Tool Use 的区别。

架构对比
--------
直接 Tool Use（Task 7）:
    用户代码 → Claude API → tool_use → 用户代码执行函数 → tool_result → Claude API

MCP Tool Integration（本文件）:
    用户代码 → MCP Server（发现工具）→ Claude API（with tools）→ tool_use
    → 用户代码（调用 MCP Server 执行工具）→ tool_result → Claude API

关键区别：工具在 MCP Server 中定义，通过协议通信，解耦更彻底。
  - 直接 Tool Use：工具实现与调用代码耦合在同一进程
  - MCP 集成：工具实现位于独立 Server，客户端通过标准协议（stdio / SSE）发现并调用
  - MCP 工具可被多个不同客户端/模型复用，无需每次重新定义

本文件采用「模拟 MCP Client」策略：
  - MCPClientSimulator 类复现真实 MCP Client 的接口（list_tools / call_tool）
  - 工具实现复用 12_mcp_server_demo.py 中的函数（importlib 动态导入）
  - Claude API 调用和 agentic loop 完全真实
"""

import sys
import os
import json
import importlib.util

# 解决 Windows 中文乱码问题
sys.stdout.reconfigure(encoding="utf-8")

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

CLIENT = Anthropic()
MODEL = "ppio/pa/claude-sonnet-4-6"

# =============================================================================
# 第一部分：动态加载 12_mcp_server_demo.py 中的工具函数
# =============================================================================
# 在真实 MCP 场景中，工具实现运行在独立的 Server 进程里；
# 这里用 importlib 在同进程内加载，以便聚焦在集成链路本身。

_SERVER_PATH = os.path.join(os.path.dirname(__file__), "12_mcp_server_demo.py")

def _load_server_module():
    """动态加载 12_mcp_server_demo.py，返回其模块对象。"""
    spec = importlib.util.spec_from_file_location("mcp_server_demo", _SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    # 避免 FastMCP 在导入时试图启动 Server
    # spec.loader.exec_module 会执行模块顶层代码，但 mcp.run() 在 __main__ 块中，安全
    spec.loader.exec_module(mod)
    return mod

try:
    _server_mod = _load_server_module()
    _TOOL_FUNCS = {
        "calculate": _server_mod.calculate,
        "read_file": _server_mod.read_file,
        "list_directory": _server_mod.list_directory,
    }
    print("[加载] 成功从 12_mcp_server_demo.py 导入工具函数")
except Exception as e:
    print(f"[警告] 无法加载 12_mcp_server_demo.py（{e}），将使用内置备用实现")
    import math

    def _fallback_calculate(expression: str) -> str:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "错误：表达式含有不允许的字符。"
        try:
            return f"{expression} = {eval(expression)}"  # nosec
        except Exception as ex:
            return f"计算失败：{ex}"

    def _fallback_list_directory(path: str = ".") -> str:
        abs_path = os.path.abspath(path)
        entries = []
        for name in sorted(os.listdir(abs_path)):
            full = os.path.join(abs_path, name)
            if os.path.isdir(full):
                entries.append({"name": name, "type": "dir"})
            else:
                entries.append({"name": name, "type": "file",
                                 "size_kb": round(os.path.getsize(full) / 1024, 1)})
        return json.dumps({"path": abs_path, "count": len(entries), "entries": entries},
                          ensure_ascii=False, indent=2)

    _TOOL_FUNCS = {
        "calculate": _fallback_calculate,
        "list_directory": _fallback_list_directory,
    }


# =============================================================================
# 第二部分：MCPClientSimulator —— 模拟 MCP Client 接口
# =============================================================================

class MCPClientSimulator:
    """
    模拟 MCP Client 的接口。

    在真实场景中，这个类会通过 stdio / SSE 与 MCP Server 进程通信：
      - list_tools()  对应 MCP 协议的 tools/list  请求
      - call_tool()   对应 MCP 协议的 tools/call  请求

    MCP 工具格式（inputSchema，camelCase）与 Claude tools 格式（input_schema，snake_case）
    不同，to_claude_tools() 负责转换。
    """

    # ------------------------------------------------------------------
    # MCP 工具定义（inputSchema，符合 MCP 协议规范）
    # ------------------------------------------------------------------
    _MCP_TOOLS: list[dict] = [
        {
            "name": "calculate",
            "description": (
                "安全计算数学表达式，支持加减乘除、括号和常用数学运算。"
                "例如：'2 + 3 * 4'、'(10 - 2) / 4'、'2**8'"
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式字符串"
                    }
                },
                "required": ["expression"]
            }
        },
        {
            "name": "list_directory",
            "description": (
                "列出目录中的文件和子目录，返回 JSON 格式的条目列表（含文件名、类型、大小）。"
                "path 参数默认为当前工作目录（.）。"
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要列举的目录路径，默认为 '.'",
                        "default": "."
                    }
                },
                "required": []
            }
        },
    ]

    # ------------------------------------------------------------------
    # MCP 协议接口
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict]:
        """列出所有可用工具（模拟 MCP tools/list 响应）。"""
        return self._MCP_TOOLS

    def call_tool(self, name: str, arguments: dict) -> str:
        """
        调用工具（模拟 MCP tools/call 请求）。

        真实 MCP Client 会将请求序列化为 JSON 并通过 stdio / SSE 发送给 Server，
        Server 执行后返回结果字符串；这里直接调用本地函数模拟该过程。
        """
        if name not in _TOOL_FUNCS:
            return f"错误：未知工具 '{name}'，可用工具：{list(_TOOL_FUNCS.keys())}"
        print(f"  [MCP call_tool] 工具={name}  参数={arguments}")
        result = _TOOL_FUNCS[name](**arguments)
        print(f"  [MCP call_tool] 返回结果（前 120 字）：{str(result)[:120]}")
        return result

    def to_claude_tools(self) -> list[dict]:
        """
        将 MCP 工具格式转换为 Claude tool_use 格式。

        格式差异：
          MCP   → inputSchema  （camelCase，符合 JSON Schema 惯例）
          Claude → input_schema （snake_case，Anthropic API 规范）
        """
        claude_tools = []
        for tool in self._MCP_TOOLS:
            claude_tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["inputSchema"],   # ← 关键转换
            })
        return claude_tools


# =============================================================================
# 第三部分：完整的 MCP 集成 Agentic Loop
# =============================================================================

def run_with_mcp(user_input: str, mcp_client: MCPClientSimulator) -> str:
    """
    完整集成链路：
    1. 从 MCP Client 获取可用工具列表（list_tools）
    2. 将工具转换为 Claude 格式后附加到 API 请求
    3. 进入 agentic loop：
       a. 调用 Claude API
       b. 若 stop_reason == 'tool_use'，通过 MCP Client 执行工具（call_tool）
       c. 将 tool_result 回传给 Claude，继续循环
       d. 直到 stop_reason == 'end_turn'
    """
    # 步骤 1：通过 MCP 协议发现工具
    print("\n[MCP list_tools] 发现工具列表...")
    mcp_tools = mcp_client.list_tools()
    for t in mcp_tools:
        print(f"  - {t['name']}: {t['description'][:60]}...")

    # 步骤 2：格式转换
    claude_tools = mcp_client.to_claude_tools()

    messages = [{"role": "user", "content": user_input}]

    print(f"\n[Claude API] 发送请求，工具数量={len(claude_tools)}")

    # 步骤 3：Agentic Loop
    for iteration in range(5):  # 最多循环 5 次防止死循环
        response = CLIENT.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=claude_tools,
            messages=messages,
        )

        print(f"  [循环 {iteration + 1}] stop_reason={response.stop_reason}")

        if response.stop_reason == "end_turn":
            # 提取最终文本回答
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "(无文本回答)"

        if response.stop_reason == "tool_use":
            # 将 Claude 的 assistant 消息加入历史
            messages.append({"role": "assistant", "content": response.content})

            # 处理所有工具调用请求
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n[tool_use 请求] id={block.id}  name={block.name}")
                    # 通过 MCP Client 执行工具
                    result_text = mcp_client.call_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

            # 将工具结果回传给 Claude
            messages.append({"role": "user", "content": tool_results})
            continue

        # 其他 stop_reason（如 max_tokens）
        break

    return "(达到最大循环次数，未获得完整回答)"


def run_without_tools(user_input: str) -> str:
    """直接调用 Claude，不附带任何工具（降级策略使用）。"""
    response = CLIENT.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": user_input}],
    )
    return response.content[0].text


# =============================================================================
# 第四部分：降级策略
# =============================================================================

def run_with_fallback(user_input: str, mcp_client: MCPClientSimulator) -> str:
    """
    带降级的执行入口。

    若 MCP 工具调用链路出现异常，自动降级为直接回答（不使用工具）。
    这在生产环境中保障了服务可用性。
    """
    try:
        return run_with_mcp(user_input, mcp_client)
    except Exception as e:
        print(f"\n[降级] MCP 工具调用失败：{e}，降级为直接回答")
        return run_without_tools(user_input)


# =============================================================================
# 第五部分：主程序 —— 运行测试用例并打印对比总结
# =============================================================================

def print_separator(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    mcp_client = MCPClientSimulator()

    # -------------------------------------------------------
    # 测试用例 1：数学计算（触发 calculate 工具）
    # -------------------------------------------------------
    print_separator("测试用例 1：数学计算")
    query1 = "帮我计算 sqrt(144) + 2^8 的结果"
    print(f"用户输入：{query1}")

    # 注意：12_mcp_server_demo.py 的 calculate 仅允许有限字符集，
    # Claude 会将 sqrt / ^ 转换为合法的 Python 表达式再调用工具。
    answer1 = run_with_fallback(query1, mcp_client)
    print(f"\n最终回答：\n{answer1}")

    # -------------------------------------------------------
    # 测试用例 2：目录列举（触发 list_directory 工具）
    # -------------------------------------------------------
    print_separator("测试用例 2：列举当前目录文件")
    query2 = "列出当前目录有哪些文件"
    print(f"用户输入：{query2}")
    answer2 = run_with_fallback(query2, mcp_client)
    print(f"\n最终回答：\n{answer2}")

    # -------------------------------------------------------
    # 对比总结
    # -------------------------------------------------------
    print_separator("与 Task 7（直接 Tool Use）的对比总结")
    print("""
┌─────────────────────┬──────────────────────────┬──────────────────────────┐
│ 维度                │ 直接 Tool Use（Task 7）  │ MCP Tool Integration     │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ 工具定义位置        │ 与调用代码同一文件/进程  │ 独立 MCP Server 进程     │
│ 发现工具方式        │ 硬编码 TOOLS 列表        │ list_tools() 协议请求    │
│ 工具执行方式        │ 直接调用本地函数          │ call_tool() 协议请求     │
│ 工具格式字段        │ input_schema（snake）    │ inputSchema（camel）     │
│ 解耦程度            │ 紧耦合                   │ 松耦合，可跨进程/语言    │
│ 复用性              │ 仅本项目可用             │ 多客户端/模型共享        │
│ 复杂度              │ 简单，适合单一应用        │ 稍复杂，适合平台化场景   │
│ 降级能力            │ 无协议层，天然无需降级    │ 可在协议失败时降级       │
└─────────────────────┴──────────────────────────┴──────────────────────────┘
""")
    print("关键结论：")
    print("  1. Agentic Loop 逻辑完全相同：tool_use → 执行 → tool_result → 继续")
    print("  2. MCP 增加了「工具发现」步骤，使工具定义与调用代码解耦")
    print("  3. 格式转换（inputSchema → input_schema）是集成的关键细节")
    print("  4. 降级策略保障了 MCP 链路异常时的服务连续性")


if __name__ == "__main__":
    main()
