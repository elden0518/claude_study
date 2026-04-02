"""
04_project/cli_assistant.py — CLI 智能助手（综合项目）
=====================================================

本文件整合了课程前 15 个 Demo 的所有核心知识点：

  知识点整合列表：
  ┌─────────────────────────────────────────────────────────────────┐
  │ Demo 04  04_streaming.py      → 流式输出（client.messages.stream）  │
  │ Demo 05  05_error_handling.py → API 错误捕获与优雅退出               │
  │ Demo 07  07_tool_use.py       → Tool Use Agentic Loop              │
  │ Demo 09  09_conversation.py   → ConversationManager 多轮对话管理    │
  │ Demo 14  14_slash_command.py  → Slash Command 解析与处理            │
  └─────────────────────────────────────────────────────────────────┘

功能清单：
  - 多轮对话（滑动窗口，最多保留 20 条消息）
  - 流式输出（逐 token 打印 Claude 回复）
  - Tool Use Agentic Loop（calculator / get_weather / read_file / write_file）
  - 内置 Slash Commands（/help /clear /save /load /tools /exit）
  - ANSI 彩色终端输出（青色=Claude, 黄色=工具, 绿色=系统, 红色=错误）
  - 完善的错误处理与 KeyboardInterrupt 优雅退出

运行方式：
  # 演示模式（非交互）：
  .venv/Scripts/python.exe 04_project/cli_assistant.py

  # 交互模式（需在源码末尾改为 main()）：
  .venv/Scripts/python.exe 04_project/cli_assistant.py
"""

# ── 0. Windows 控制台编码修复（来自 04_streaming.py 的实践）──────────────────
import sys
import os
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── 1. 标准库导入 ──────────────────────────────────────────────────────────────
import json
import math
from pathlib import Path

# ── 2. 第三方库导入 ────────────────────────────────────────────────────────────
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── 3. 全局配置 ────────────────────────────────────────────────────────────────
MODEL = "ppio/pa/claude-sonnet-4-6"

# System prompt：控制 Claude 的角色与行为风格
SYSTEM_PROMPT = """你是一个实用的 AI 助手，具备以下能力：
1. 回答问题和进行对话
2. 执行数学计算（使用 calculator 工具）
3. 查询天气信息（使用 get_weather 工具，数据为模拟）
4. 读写本地文件（使用 read_file/write_file 工具）

请用中文回复，保持简洁友好的风格。"""


# ============================================================
# 第一部分：彩色输出辅助函数
# （知识点：ANSI 转义码，不依赖第三方库）
# ============================================================

def cyan(text: str) -> str:
    """Claude 回复 — 青色"""
    return f"\033[36m{text}\033[0m"

def yellow(text: str) -> str:
    """工具调用 — 黄色"""
    return f"\033[33m{text}\033[0m"

def green(text: str) -> str:
    """系统消息 — 绿色"""
    return f"\033[32m{text}\033[0m"

def red(text: str) -> str:
    """错误消息 — 红色"""
    return f"\033[31m{text}\033[0m"


# ============================================================
# 第二部分：ConversationManager（来自 09_conversation.py）
# 核心改造：messages 列表存储对话历史，支持工具调用消息格式
# ============================================================

class ConversationManager:
    """
    封装多轮对话历史的管理类。（整合自 09_conversation.py）

    与原版的差异：
    - add_assistant_content() 支持 list 类型（用于 tool_use / tool_result 消息）
    - save/load 保持 JSON 格式兼容
    """

    def __init__(self, system_prompt: str = "", max_messages: int = 20):
        """
        参数：
            system_prompt  — 发给 Claude 的 system 字段
            max_messages   — 滑动窗口最大消息条数（user + assistant 合计）
        """
        self.messages: list[dict] = []
        self.system_prompt = system_prompt
        self.max_messages = max_messages

    # ----------------------------------------------------------
    # 基础操作
    # ----------------------------------------------------------

    def add_user_message(self, content):
        """追加用户消息（content 可以是 str 或 list）"""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        """追加 assistant 消息（content 可以是 str 或 list，支持 tool_use block）"""
        self.messages.append({"role": "assistant", "content": content})

    def clear(self):
        """清空全部对话历史"""
        self.messages.clear()
        print(green("  [对话历史已清空]"))

    def _trim_history(self):
        """
        滑动窗口截断：超过 max_messages 时从头部删除旧消息。
        （整合自 09_conversation.py 的 _trim_history 方法）
        """
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)
        # 确保第一条消息是 user（Claude API 要求 messages 以 user 开头）
        while self.messages and self.messages[0]["role"] != "user":
            self.messages.pop(0)

    # ----------------------------------------------------------
    # 持久化（来自 09_conversation.py 的 save / load）
    # ----------------------------------------------------------

    def save(self, filepath: str):
        """保存对话历史到 JSON 文件"""
        # tool_use / tool_result 消息中可能有非序列化对象，需要转换
        def serialize_content(content):
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                result = []
                for block in content:
                    if hasattr(block, "__dict__"):
                        # SDK 对象（ToolUseBlock 等），提取关键字段
                        d = {"type": getattr(block, "type", "unknown")}
                        for attr in ("id", "name", "input", "text"):
                            if hasattr(block, attr):
                                d[attr] = getattr(block, attr)
                        result.append(d)
                    else:
                        result.append(block)
                return result
            return content

        serializable = []
        for msg in self.messages:
            serializable.append({
                "role": msg["role"],
                "content": serialize_content(msg["content"])
            })

        data = {
            "system_prompt": self.system_prompt,
            "max_messages": self.max_messages,
            "messages": serializable,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(green(f"  [已保存] {filepath}（{len(self.messages)} 条消息）"))

    def load(self, filepath: str):
        """从 JSON 文件加载对话历史（覆盖当前历史）"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.system_prompt = data.get("system_prompt", "")
        self.max_messages = data.get("max_messages", 20)
        self.messages = data.get("messages", [])
        print(green(f"  [已加载] {filepath}（{len(self.messages)} 条消息）"))

    def message_count(self) -> int:
        return len(self.messages)


# ============================================================
# 第三部分：内置工具定义（来自 07_tool_use.py，扩展为 4 个工具）
# ============================================================

# 工具 JSON Schema 列表（告知 Claude 可用工具及其参数）
TOOLS = [
    {
        "name": "calculator",
        "description": (
            "计算数学表达式，支持加减乘除、幂运算、括号、常用 math 函数等。"
            "输入合法的 Python 数学表达式字符串，返回计算结果。"
            "例如：'2 ** 10 + 100'、'sqrt(16) * pi'。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，如 '(2**10 + 100) / 3'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": (
            "查询指定城市的当前天气信息（模拟数据，非真实天气）。"
            "返回温度、天气状况、湿度等信息。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如 '北京'、'上海'、'London'"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "read_file",
        "description": (
            "读取本地文件的内容，返回文件文本。"
            "仅支持读取文本文件（UTF-8 编码）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径（绝对路径或相对于工作目录的路径）"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": (
            "将内容写入本地文件（覆盖写入）。"
            "如果文件不存在会自动创建，父目录必须已存在。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径"
                },
                "content": {
                    "type": "string",
                    "description": "要写入文件的文本内容"
                }
            },
            "required": ["path", "content"]
        }
    }
]


# ============================================================
# 第四部分：工具实现函数（来自 07_tool_use.py，扩展了 2 个文件工具）
# ============================================================

def tool_calculator(expression: str) -> str:
    """
    安全计算数学表达式。（整合自 07_tool_use.py 的 calculate 函数）
    使用白名单 eval，仅允许 math 模块的常用函数和运算符。
    """
    try:
        safe_globals = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow,
        }
        for name in ["sqrt", "log", "log2", "log10", "sin", "cos", "tan",
                     "pi", "e", "ceil", "floor", "factorial"]:
            safe_globals[name] = getattr(math, name)

        result = eval(expression, safe_globals)
        return f"计算结果：{expression} = {result}"
    except ZeroDivisionError:
        return f"计算错误：除以零"
    except Exception as e:
        return f"计算出错：{str(e)}"


def tool_get_weather(city: str) -> str:
    """
    返回模拟天气数据（非真实天气）。
    模拟不同城市的天气信息，使演示更真实。
    """
    # 预设几个城市的模拟天气数据
    weather_db = {
        "北京": {"temp": 18, "condition": "晴天", "humidity": 45, "wind": "北风 3级"},
        "上海": {"temp": 22, "condition": "多云", "humidity": 65, "wind": "东南风 2级"},
        "广州": {"temp": 28, "condition": "阵雨", "humidity": 80, "wind": "南风 2级"},
        "深圳": {"temp": 27, "condition": "多云转晴", "humidity": 75, "wind": "东风 2级"},
        "成都": {"temp": 16, "condition": "阴天", "humidity": 70, "wind": "微风"},
        "london": {"temp": 12, "condition": "cloudy", "humidity": 75, "wind": "W 3mph"},
        "new york": {"temp": 15, "condition": "partly cloudy", "humidity": 55, "wind": "NE 5mph"},
        "tokyo": {"temp": 20, "condition": "sunny", "humidity": 60, "wind": "N 2m/s"},
    }

    city_lower = city.lower().strip()
    # 精确匹配（中文城市）
    for key, data in weather_db.items():
        if city == key or city_lower == key:
            return (
                f"【{city} 天气（模拟数据）】\n"
                f"  温度：{data['temp']}°C\n"
                f"  天气：{data['condition']}\n"
                f"  湿度：{data['humidity']}%\n"
                f"  风向：{data['wind']}"
            )

    # 未找到城市，返回通用模拟数据
    import random
    temp = random.randint(10, 35)
    conditions = ["晴天", "多云", "阴天", "小雨"]
    return (
        f"【{city} 天气（模拟数据）】\n"
        f"  温度：{temp}°C\n"
        f"  天气：{random.choice(conditions)}\n"
        f"  湿度：{random.randint(40, 90)}%\n"
        f"  注意：该城市为随机模拟，仅供演示"
    )


def tool_read_file(path: str) -> str:
    """
    读取本地文件内容。
    错误时返回描述性错误消息而非抛出异常（工具执行不应崩溃主程序）。
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return f"错误：文件不存在 → {path}"
        if not file_path.is_file():
            return f"错误：路径不是文件 → {path}"

        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        preview = "\n".join(lines[:50])  # 最多返回前 50 行
        suffix = f"\n... (共 {len(lines)} 行，只显示前 50 行)" if len(lines) > 50 else ""
        return f"文件内容（{path}）：\n{preview}{suffix}"
    except PermissionError:
        return f"错误：没有权限读取文件 → {path}"
    except UnicodeDecodeError:
        return f"错误：文件编码非 UTF-8，无法以文本方式读取 → {path}"
    except Exception as e:
        return f"读取文件出错：{str(e)}"


def tool_write_file(path: str, content: str) -> str:
    """
    写入内容到本地文件（覆盖写入）。
    父目录不存在时提示用户，而非自动创建。
    """
    try:
        file_path = Path(path)
        if not file_path.parent.exists():
            return f"错误：父目录不存在 → {file_path.parent}，请先创建目录"

        file_path.write_text(content, encoding="utf-8")
        return f"写入成功：{path}（{len(content)} 字符，{len(content.splitlines())} 行）"
    except PermissionError:
        return f"错误：没有权限写入文件 → {path}"
    except Exception as e:
        return f"写入文件出错：{str(e)}"


def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """
    工具分发函数：根据工具名路由到对应的 Python 函数。
    （整合自 07_tool_use.py 的 dispatch_tool 模式）
    """
    try:
        if tool_name == "calculator":
            return tool_calculator(**tool_input)
        elif tool_name == "get_weather":
            return tool_get_weather(**tool_input)
        elif tool_name == "read_file":
            return tool_read_file(**tool_input)
        elif tool_name == "write_file":
            return tool_write_file(**tool_input)
        else:
            return f"未知工具：{tool_name}，可用工具：calculator / get_weather / read_file / write_file"
    except Exception as e:
        # 工具执行异常不崩溃主程序（来自 05_error_handling.py 的优雅降级实践）
        return f"工具 {tool_name} 执行出错：{str(e)}"


# ============================================================
# 第五部分：Slash Command 处理器（来自 14_slash_command.py）
# 本版本将内置命令直接硬编码，而非从文件加载模板
# ============================================================

class SlashCommandProcessor:
    """
    处理内置 Slash Commands。（整合自 14_slash_command.py 的 SlashCommandProcessor）

    与原版的差异：
    - 命令直接在代码中定义（内置命令），无需读取 .md 文件
    - 增加了 /save、/load、/tools、/clear 等 CLI 特有命令
    """

    COMMANDS = {
        "help":  "显示所有可用命令",
        "clear": "清空当前对话历史",
        "save":  "保存对话历史到 JSON 文件，用法：/save [文件名]",
        "load":  "从 JSON 文件加载对话历史，用法：/load [文件名]",
        "tools": "列出所有可用的工具及其说明",
        "exit":  "退出程序（等同于输入 exit）",
    }

    def parse(self, user_input: str) -> tuple[str, str]:
        """
        解析 '/command [args]' 格式的用户输入。
        返回 (command_name, arguments)。
        （整合自 14_slash_command.py 的 parse_input 方法）
        """
        user_input = user_input.strip()
        if not user_input.startswith("/"):
            return "", user_input

        without_slash = user_input[1:]
        parts = without_slash.split(" ", maxsplit=1)
        command_name = parts[0].lower()
        arguments = parts[1].strip() if len(parts) > 1 else ""
        return command_name, arguments

    def is_known(self, command_name: str) -> bool:
        return command_name in self.COMMANDS


def handle_command(user_input: str,
                   manager: ConversationManager,
                   processor: SlashCommandProcessor) -> bool:
    """
    处理 slash command 或 exit 命令。
    返回 True 表示应退出程序，否则返回 False。

    支持命令：/help /clear /save /load /tools /exit（以及裸 exit）
    （整合自 14_slash_command.py 的 execute 逻辑）
    """
    inp = user_input.strip()

    # 处理裸 exit
    if inp.lower() == "exit":
        print(green("再见！感谢使用 CLI 智能助手。"))
        return True

    command, args = processor.parse(inp)

    # /exit
    if command == "exit":
        print(green("再见！感谢使用 CLI 智能助手。"))
        return True

    # /help
    elif command == "help":
        print(green("\n=== 可用命令 ==="))
        for cmd, desc in SlashCommandProcessor.COMMANDS.items():
            print(green(f"  /{cmd:<8} — {desc}"))
        print(green(f"\n  直接输入文字即可与 Claude 对话。"))
        print(green(f"  当前对话历史：{manager.message_count()} 条消息\n"))

    # /clear
    elif command == "clear":
        manager.clear()

    # /save [filename]
    elif command == "save":
        filename = args if args else "conversation.json"
        if not filename.endswith(".json"):
            filename += ".json"
        try:
            manager.save(filename)
        except Exception as e:
            print(red(f"  保存失败：{e}"))

    # /load [filename]
    elif command == "load":
        filename = args if args else "conversation.json"
        if not filename.endswith(".json"):
            filename += ".json"
        try:
            manager.load(filename)
        except FileNotFoundError:
            print(red(f"  文件不存在：{filename}"))
        except Exception as e:
            print(red(f"  加载失败：{e}"))

    # /tools
    elif command == "tools":
        print(yellow("\n=== 可用工具 ==="))
        for tool in TOOLS:
            print(yellow(f"  {tool['name']:<15} — {tool['description'][:60]}..."))
        print()

    else:
        print(red(f"  未知命令：/{command}，输入 /help 查看所有命令"))

    return False


# ============================================================
# 第六部分：核心对话函数 — 流式 + Tool Use 整合
# 策略：普通对话用流式，遇到工具调用时切换非流式（简化实现）
# ============================================================

def chat_with_tools_stream(user_input: str, manager: ConversationManager):
    """
    发送用户消息，处理 Claude 回复（流式输出 + Tool Use Agentic Loop）。

    整合策略（来自 04_streaming.py + 07_tool_use.py）：
    ─────────────────────────────────────────────────
    1. 将用户消息追加到对话历史
    2. 第一次调用使用流式（client.messages.stream），实时打印文字
    3. 若 stop_reason == "end_turn" → 流式结束，正常完成
    4. 若 stop_reason == "tool_use" → 切换为非流式 Agentic Loop：
       a. 执行所有工具，打印工具名称 / 结果（黄色）
       b. 追加 tool_result，调用 API 获取最终回复（非流式）
       c. 重复直到 stop_reason == "end_turn"
    5. 将最终回复追加到对话历史，触发滑动窗口截断
    """
    client = anthropic.Anthropic()

    # 追加用户消息到历史（不调用 API）
    manager.add_user_message(user_input)

    # ── 第一次调用：尝试流式 ─────────────────────────────────────────────────
    print(cyan("Claude: "), end="", flush=True)

    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=1024,
            system=manager.system_prompt,
            tools=TOOLS,
            messages=manager.messages,
        ) as stream:
            # 流式打印文字部分
            text_so_far = []
            for chunk in stream.text_stream:
                print(cyan(chunk), end="", flush=True)
                text_so_far.append(chunk)

            # 流结束后获取完整消息对象（含 stop_reason 和 content blocks）
            final_msg = stream.get_final_message()

        print()  # 换行

        stop_reason = final_msg.stop_reason

        # ── 情况 A：正常结束，无工具调用 ────────────────────────────────────
        if stop_reason == "end_turn":
            # 提取完整文本（可能包含多个 TextBlock）
            full_text = "".join(
                block.text for block in final_msg.content if hasattr(block, "text")
            )
            manager.add_assistant_message(full_text)
            manager._trim_history()
            return

        # ── 情况 B：有工具调用，切换到非流式 Agentic Loop ───────────────────
        if stop_reason == "tool_use":
            # 将 assistant 的回复（含 tool_use blocks）加入历史
            manager.add_assistant_message(final_msg.content)

            # 进入 Agentic Loop（来自 07_tool_use.py 的 run_agent 逻辑）
            _agentic_loop(client, manager)
            return

        # ── 情况 C：未预期的 stop_reason ─────────────────────────────────
        print(red(f"[警告] 未预期的 stop_reason: {stop_reason}"))

    except anthropic.APIError as e:
        # API 错误处理（来自 05_error_handling.py 的错误捕获实践）
        print()  # 确保换行
        print(red(f"API 错误：{e}"))
        # 回滚：移除刚才追加的用户消息，避免历史污染
        if manager.messages and manager.messages[-1]["role"] == "user":
            manager.messages.pop()


def _agentic_loop(client: anthropic.Anthropic, manager: ConversationManager):
    """
    工具调用的 Agentic Loop（非流式）。
    （整合自 07_tool_use.py 的 run_agent 核心逻辑）

    循环直到 stop_reason == "end_turn" 为止：
      1. 执行所有 tool_use blocks（打印黄色工具信息）
      2. 将 tool_result 追加为 user 消息
      3. 调用 API，获取下一步回复
      4. 若继续有工具调用则重复，否则打印最终回复并退出
    """
    iteration = 0
    max_iterations = 10  # 防止无限循环

    while iteration < max_iterations:
        iteration += 1

        # 获取最后一条 assistant 消息（包含 tool_use blocks）
        last_assistant_msg = None
        for msg in reversed(manager.messages):
            if msg["role"] == "assistant":
                last_assistant_msg = msg
                break

        if last_assistant_msg is None:
            break

        # 遍历所有 tool_use blocks，逐一执行
        content = last_assistant_msg["content"]
        tool_results = []
        has_tool_use = False

        for block in content:
            # block 可能是 SDK 对象（ToolUseBlock）或字典
            if hasattr(block, "type"):
                block_type = block.type
                tool_id = getattr(block, "id", None)
                tool_name = getattr(block, "name", None)
                tool_input = getattr(block, "input", {})
            else:
                block_type = block.get("type", "")
                tool_id = block.get("id")
                tool_name = block.get("name")
                tool_input = block.get("input", {})

            if block_type != "tool_use":
                continue

            has_tool_use = True
            print(yellow(f"\n[工具调用] {tool_name}({json.dumps(tool_input, ensure_ascii=False)})"))

            # 执行工具（来自 dispatch_tool）
            result = dispatch_tool(tool_name, tool_input)
            print(yellow(f"[工具结果] {result[:200]}{'...' if len(result) > 200 else ''}"))

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result
            })

        if not has_tool_use:
            break

        # 将 tool_result 追加为 user 消息
        manager.add_user_message(tool_results)

        # 非流式调用 API，获取下一步回复
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=manager.system_prompt,
                tools=TOOLS,
                messages=manager.messages,
            )
        except anthropic.APIError as e:
            print(red(f"工具循环中 API 错误：{e}"))
            break

        # 打印回复文本（青色）
        reply_text = ""
        for block in response.content:
            if hasattr(block, "text") and block.text:
                print(cyan(f"Claude: {block.text}"))
                reply_text += block.text

        # 将 assistant 回复追加到历史
        manager.add_assistant_message(response.content)

        # 检查是否还有工具调用
        if response.stop_reason == "end_turn":
            # 最终回复已打印，整理历史（将复杂对象替换为纯文本）
            manager._trim_history()
            break
        elif response.stop_reason != "tool_use":
            print(red(f"[警告] Agentic Loop 遇到未预期 stop_reason: {response.stop_reason}"))
            break
        # 否则继续循环（stop_reason == "tool_use"）

    if iteration >= max_iterations:
        print(red(f"[警告] 工具调用循环超过最大轮次（{max_iterations}），强制退出"))


# ============================================================
# 第七部分：欢迎横幅与主程序
# ============================================================

def print_banner():
    """打印欢迎横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║           Claude CLI 智能助手  v1.0                       ║
║                                                           ║
║  模型：claude-sonnet-4-6                                  ║
║  整合：多轮对话 + 流式输出 + Tool Use + Slash Commands     ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(green(banner))


def main():
    """
    交互式主循环。
    （整合所有知识点的主程序入口）
    """
    print_banner()
    manager = ConversationManager(system_prompt=SYSTEM_PROMPT, max_messages=20)
    processor = SlashCommandProcessor()

    print(green("输入 /help 查看命令，输入 exit 退出"))
    print()

    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue

            # 处理 slash commands 和 exit（来自 14_slash_command.py）
            if user_input.startswith("/") or user_input.lower() == "exit":
                should_exit = handle_command(user_input, manager, processor)
                if should_exit:
                    break
                continue

            # 普通对话：流式输出 + Tool Use（整合 04 + 07）
            chat_with_tools_stream(user_input, manager)

        except KeyboardInterrupt:
            # 优雅退出（来自 05_error_handling.py 的 KeyboardInterrupt 处理）
            print("\n" + green("收到中断信号，再见！"))
            break
        except anthropic.APIError as e:
            # API 层级错误兜底（来自 05_error_handling.py）
            print(red(f"\nAPI 错误：{e}"))


# ============================================================
# 第八部分：非交互式自动演示（用于自动化测试，不需要用户输入）
# ============================================================

def demo_non_interactive():
    """
    自动演示函数，不需要用户输入。
    测试 4 个核心场景：普通对话、计算器工具、天气工具、Slash Command。
    """
    print_banner()
    print(green("=== 自动演示模式（demo_non_interactive）===\n"))

    client = anthropic.Anthropic()
    manager = ConversationManager(system_prompt=SYSTEM_PROMPT, max_messages=20)
    processor = SlashCommandProcessor()

    # ── 场景 1：/help 命令 ─────────────────────────────────────────────────────
    print(green("─" * 60))
    print(green("【场景 1】Slash Command: /help"))
    print(green("─" * 60))
    handle_command("/help", manager, processor)

    # ── 场景 2：普通对话（流式） ──────────────────────────────────────────────
    print(green("─" * 60))
    print(green("【场景 2】普通对话（流式输出）"))
    print(green("─" * 60))
    print(f"> 你好！请简单介绍一下你有哪些能力？")
    chat_with_tools_stream("你好！请简单介绍一下你有哪些能力？", manager)
    print()

    # ── 场景 3：工具调用 — 计算器 ─────────────────────────────────────────────
    print(green("─" * 60))
    print(green("【场景 3】工具调用 — calculator"))
    print(green("─" * 60))
    print(f"> 帮我计算 (2**10 + 100) / 3 的结果")
    chat_with_tools_stream("帮我计算 (2**10 + 100) / 3 的结果", manager)
    print()

    # ── 场景 4：工具调用 — 天气查询 ───────────────────────────────────────────
    print(green("─" * 60))
    print(green("【场景 4】工具调用 — get_weather"))
    print(green("─" * 60))
    print(f"> 北京现在天气怎么样？")
    chat_with_tools_stream("北京现在天气怎么样？", manager)
    print()

    # ── 场景 5：/tools 命令 ─────────────────────────────────────────────────
    print(green("─" * 60))
    print(green("【场景 5】Slash Command: /tools"))
    print(green("─" * 60))
    handle_command("/tools", manager, processor)

    # ── 结束报告 ──────────────────────────────────────────────────────────────
    print(green("─" * 60))
    print(green(f"演示完成！对话历史共 {manager.message_count()} 条消息"))
    print(green("─" * 60))


# ============================================================
# 入口：默认运行自动演示
# ============================================================

if __name__ == "__main__":
    # 自动演示（不需要用户输入）
    demo_non_interactive()

    # 若需要交互式 CLI，注释上面一行并取消以下注释：
    # main()
