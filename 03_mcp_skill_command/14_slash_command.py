"""
14_slash_command.py — Claude Code Slash Command 演示

【Slash Command 原理】
Claude Code 在启动时会扫描以下目录中的 .md 文件作为自定义命令：
  - 项目级：.claude/commands/*.md   （只在当前项目生效）
  - 全局级：~/.claude/commands/*.md （对所有项目生效）

当用户输入 /command_name args 时，Claude Code 会：
  1. 找到对应的 .md 文件（文件名去掉 .md 后缀即命令名）
  2. 读取文件中的 Markdown 内容作为 prompt 模板
  3. 将模板中所有的 $ARGUMENTS 占位符替换为用户实际输入的参数
  4. 把替换后的完整 prompt 发送给 Claude 模型执行

这样做的好处：
  - 无需编写代码，只用 Markdown 就能扩展 Claude Code 的功能
  - 可以定制团队共用的标准化 prompt 工作流
  - $ARGUMENTS 让同一个命令可以处理不同的输入内容
"""

import os
import glob
from pathlib import Path


# ============================================================
# 第一部分：展示 review.md 命令文件的内容
# ============================================================

def show_command_file_content():
    """读取并展示 review.md 文件内容，说明命令文件的结构"""
    commands_dir = Path(__file__).parent / "commands"
    review_file = commands_dir / "review.md"

    print("=" * 60)
    print("【命令文件示例】review.md 的内容：")
    print("=" * 60)

    if review_file.exists():
        print(review_file.read_text(encoding="utf-8"))
    else:
        print("(文件不存在，请先创建 commands/review.md)")

    print()
    print("说明：")
    print("  - 文件名 review.md → 命令名 /review")
    print("  - $ARGUMENTS 是占位符，执行时被用户输入的实际参数替换")
    print()


# ============================================================
# 第二部分：命令解析器（Python 实现）
# ============================================================

class SlashCommandProcessor:
    """
    模拟 Claude Code 的 Slash Command 处理逻辑。

    真实的 Claude Code 用 TypeScript 实现，这里用 Python 演示核心思路：
      1. 扫描命令目录，把 .md 文件加载到内存
      2. 解析用户输入，拆分出命令名和参数
      3. 将 $ARGUMENTS 替换为实际参数，生成最终 prompt
    """

    def __init__(self, commands_dir: str):
        self.commands_dir = commands_dir
        self.commands: dict[str, str] = {}  # {命令名: prompt模板}

    def load_commands(self):
        """扫描并加载所有 .md 命令文件"""
        md_files = glob.glob(os.path.join(self.commands_dir, "*.md"))

        if not md_files:
            print(f"[警告] 在 {self.commands_dir} 中未找到任何 .md 命令文件")
            return

        for filepath in md_files:
            # 文件名（不含 .md）即命令名，例如 review.md → review
            command_name = Path(filepath).stem
            with open(filepath, encoding="utf-8") as f:
                template = f.read()
            self.commands[command_name] = template
            print(f"[加载] /{command_name}  ← {filepath}")

        print(f"\n共加载 {len(self.commands)} 个命令：{list(self.commands.keys())}\n")

    def parse_input(self, user_input: str) -> tuple[str, str]:
        """
        解析 '/command args' 格式的用户输入。

        规则：
          - 必须以 '/' 开头
          - 第一个空格之前是命令名（去掉 '/'）
          - 第一个空格之后的所有内容是参数（$ARGUMENTS）
          - 如果没有空格，参数为空字符串

        返回：(command_name, arguments)
        """
        user_input = user_input.strip()

        if not user_input.startswith("/"):
            return "", user_input  # 不是 slash command，原样返回

        # 去掉开头的 '/'，然后按第一个空格分割
        without_slash = user_input[1:]
        parts = without_slash.split(" ", maxsplit=1)

        command_name = parts[0]
        arguments = parts[1] if len(parts) > 1 else ""

        return command_name, arguments

    def execute(self, command_name: str, arguments: str) -> str:
        """
        将命令模板中的 $ARGUMENTS 替换为实际参数，返回最终 prompt。

        如果命令不存在，返回错误提示。
        如果参数为空，$ARGUMENTS 被替换为空字符串（命令文件中通常有相应的说明）。
        """
        if command_name not in self.commands:
            available = ", ".join(f"/{name}" for name in self.commands)
            return f"[错误] 未知命令 /{command_name}。可用命令：{available}"

        template = self.commands[command_name]
        # 核心替换：$ARGUMENTS → 用户输入的实际参数
        final_prompt = template.replace("$ARGUMENTS", arguments)
        return final_prompt


# ============================================================
# 第三部分：实际执行演示
# ============================================================

def demo_slash_commands():
    """模拟用户在 Claude Code 中使用 slash command 的完整流程"""

    # 初始化处理器，指向本文件同级的 commands/ 目录
    commands_dir = str(Path(__file__).parent / "commands")
    processor = SlashCommandProcessor(commands_dir)

    print("=" * 60)
    print("【步骤 1】加载命令文件")
    print("=" * 60)
    processor.load_commands()

    # 模拟用户输入的 slash command
    inputs = [
        "/review def add(a,b): return a+b  # 这个函数有什么问题？",
        "/summarize Python是一种高级编程语言，以简洁的语法和强大的库生态系统著称...",
        "/review",                          # 无参数的情况
        "/unknown some text",               # 未知命令
        "普通文本，不是 slash command",      # 非命令输入
    ]

    print("=" * 60)
    print("【步骤 2】解析并执行用户输入")
    print("=" * 60)

    for i, user_input in enumerate(inputs, 1):
        print(f"\n--- 输入 {i}: {user_input!r} ---")

        command_name, arguments = processor.parse_input(user_input)

        if not command_name:
            print(f"  → 识别为普通文本，不触发 slash command")
            print(f"  → 内容：{arguments}")
            continue

        print(f"  → 命令名：/{command_name}")
        print(f"  → 参数：{arguments!r}")

        final_prompt = processor.execute(command_name, arguments)

        print(f"\n  → 最终发送给 Claude 的 prompt（前 200 字符）：")
        preview = final_prompt[:200].replace("\n", "\n     ")
        print(f"     {preview}")
        if len(final_prompt) > 200:
            print(f"     ... (共 {len(final_prompt)} 字符)")


# ============================================================
# 第四部分：如何部署到 Claude Code
# ============================================================

def show_deployment_guide():
    """说明如何将自定义命令部署到真实的 Claude Code 环境"""

    print()
    print("=" * 60)
    print("【如何部署到 Claude Code】")
    print("=" * 60)
    print("""
方式一：项目级命令（推荐团队共享）
  1. 在项目根目录创建 .claude/commands/ 文件夹
  2. 将 .md 文件放入该目录，例如：
       .claude/commands/review.md
       .claude/commands/summarize.md
  3. 重启 Claude Code 或刷新命令列表
  4. 在 Claude Code 中输入 /review <代码> 即可使用

方式二：全局命令（对所有项目生效）
  - Windows: C:/Users/<用户名>/.claude/commands/*.md
  - macOS/Linux: ~/.claude/commands/*.md

验证命令是否加载成功：
  - 在 Claude Code 中输入 / 触发自动补全
  - 应该能看到自定义命令出现在列表中

本 demo 的命令文件位于：
  03_mcp_skill_command/commands/
  ├── review.md     →  /review
  └── summarize.md  →  /summarize

将这两个文件复制到 .claude/commands/ 即可在 Claude Code 中使用。
""")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    show_command_file_content()
    demo_slash_commands()
    show_deployment_guide()
