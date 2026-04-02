"""
主题：Claude Code Skill 系统 —— Skill 是什么？如何在 Python API 中模拟 Skill 注入？

学习目标：
  1. 理解 Skill 的概念和工作原理
  2. 了解 Skill 文件的目录结构和 Markdown 格式
  3. 用 Python 实现 SkillLoader，模拟 Skill 注入到 system prompt
  4. 创建并使用示例 Skill 文件（code-review.md）
  5. 实战：用 Skill 增强 system prompt，让 Claude 进行结构化代码审查

前置知识：
  - 已完成 01_basics/02_system_prompt.py（理解 system prompt 作用）
  - 已完成 03_mcp_skill_command/11_mcp_intro.py（了解 MCP 概念）

课程顺序：这是 03_mcp_skill_command 模块的第三个文件（13_skill_usage.py）。

# =============================================================================
# Skill 系统架构图
# =============================================================================
#
#  开发者编写 Skill 文件（Markdown）
#       ↓  保存到
#  ~/.claude/skills/skill-name.md         ← 全局 Skill
#  <project>/.claude/skills/skill-name.md ← 项目级 Skill
#       ↓  Claude Code 触发某类任务时
#  Claude Code 读取对应 Skill 文件
#       ↓  将内容
#  注入到 system prompt（作为额外指令）
#       ↓  Claude 基于增强后的 system prompt
#  执行任务（代码审查 / 文档生成 / 测试编写 等）
#
# -----------------------------------------------------------------------------
# Skill 的本质
# -----------------------------------------------------------------------------
#
#  Skill ≠ 代码插件，Skill = 提示词模板（Markdown 文件）
#
#  Claude Code 在执行任务前，会把对应 Skill 的 Markdown 内容
#  注入到 system prompt，相当于动态加载"专家知识"或"工作规范"。
#
#  类比：
#    Skill  ≈  提前写好的"操作手册"，Claude 在干活前先读一遍
#    system prompt  ≈  Claude 的"全局行为准则"
#    Skill 注入  ≈  把操作手册追加到行为准则里
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
import re
import textwrap

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "ppio/pa/claude-sonnet-4-6"

client = anthropic.Anthropic()


# =============================================================================
# Part 1：Skill 概念和目录结构说明
# =============================================================================

def show_skill_concepts():
    """打印 Skill 系统的核心概念和目录结构。"""

    print("=" * 60)
    print("Part 1：Skill 是什么？")
    print("=" * 60)

    print("""
Skill（技能文件）是 Claude Code 的可复用工作流模板。
本质是一个 Markdown 文件，包含结构化的提示词指令。

┌─────────────────────────────────────────────────────┐
│                  Skill 工作原理                       │
│                                                     │
│  1. 开发者编写 Skill（Markdown 文件）                  │
│  2. Claude Code 执行特定任务时，自动加载对应 Skill     │
│  3. Skill 内容被注入到 system prompt                  │
│  4. Claude 基于增强后的 system prompt 执行任务         │
└─────────────────────────────────────────────────────┘

Skill 文件的目录位置：

  全局 Skill（所有项目可用）：
    ~/.claude/skills/skill-name.md

  项目级 Skill（仅当前项目可用）：
    <project>/.claude/skills/skill-name.md

  优先级：项目级 Skill > 全局 Skill

Skill 文件格式（Markdown + YAML frontmatter）：

  ---
  name: skill-name          ← 技能唯一标识符
  description: 技能描述     ← 简短说明（Claude Code 用于匹配任务类型）
  ---

  # 技能标题

  ## 步骤 1：...
  ## 步骤 2：...

  frontmatter（--- 之间的部分）是元数据，不会注入到 system prompt。
  frontmatter 以下的 Markdown 内容才是真正注入的指令。

Skill vs MCP vs Tool Use 的区别：

  Tool Use  — Claude 调用外部函数/API，获取数据或执行操作
  MCP       — 工具提供方（独立进程），工具可跨应用复用
  Skill     — 提示词模板，告诉 Claude"如何做某类任务"，不执行代码
""")


# =============================================================================
# Part 2：SkillLoader —— 读取 Skill 文件，注入到 system prompt
# =============================================================================

class SkillLoader:
    """加载和管理 Skill 文件，模拟 Claude Code 的 Skill 注入机制。

    Claude Code 内部也做了类似的事情：
      1. 扫描 ~/.claude/skills/ 和项目 .claude/skills/ 目录
      2. 读取 .md 文件，解析 frontmatter（name/description）
      3. 将 frontmatter 以下的 Markdown 内容注入到 system prompt

    这个类用纯 Python 复现了这一流程，帮助你理解 Skill 的工作方式。
    """

    def __init__(self, skills_dir: str):
        """
        初始化 SkillLoader。

        Args:
            skills_dir: 存放 Skill .md 文件的目录路径。
        """
        self.skills_dir = skills_dir

    # ── 核心方法 1：加载 Skill 内容（去掉 frontmatter）─────────────────────────
    def load_skill(self, skill_name: str) -> str:
        """加载指定 Skill 的 Markdown 内容（仅保留 frontmatter 以下的部分）。

        Skill 文件结构：
          ---
          name: xxx         ← frontmatter（元数据，不注入）
          description: xxx
          ---

          # 实际指令内容  ← 这部分才注入到 system prompt
          ...

        Args:
            skill_name: Skill 名称（文件名不含 .md 后缀，如 "code-review"）。

        Returns:
            去掉 frontmatter 后的 Markdown 内容字符串。

        Raises:
            FileNotFoundError: 如果 Skill 文件不存在。
        """
        skill_path = os.path.join(self.skills_dir, f"{skill_name}.md")

        if not os.path.exists(skill_path):
            raise FileNotFoundError(
                f"Skill 文件不存在：{skill_path}\n"
                f"可用 Skill：{self.list_skills()}"
            )

        with open(skill_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # ── 解析：去掉 YAML frontmatter（--- 到 --- 之间的部分）──────────────
        # frontmatter 格式：文件开头 "---\n...\n---\n" 包裹的 YAML 块
        # 正则说明：
        #   ^---\s*\n   → 文件开头的 "---"
        #   .*?         → 任意内容（非贪婪）
        #   ^---\s*\n   → 结束的 "---"
        #   re.DOTALL   → 让 "." 匹配换行符
        #   re.MULTILINE→ 让 "^" 匹配每行开头
        frontmatter_pattern = re.compile(
            r'^---\s*\n.*?^---\s*\n',
            re.DOTALL | re.MULTILINE
        )
        # 去掉 frontmatter，保留剩余内容，并去除首尾空白
        skill_content = frontmatter_pattern.sub("", raw_content).strip()

        return skill_content

    # ── 核心方法 2：将 Skill 内容注入到 system prompt ────────────────────────
    def inject_into_system_prompt(self, base_prompt: str, skill_name: str) -> str:
        """将 Skill 内容追加到 base_prompt，返回增强后的 system prompt。

        这模拟了 Claude Code 在执行任务前，把 Skill 注入到 system prompt 的行为。

        注入格式（Claude Code 的惯例）：
          <base_prompt>

          ---
          # Skill: <skill_name>
          <skill_content>

        Args:
            base_prompt:  原始 system prompt（不含 Skill）。
            skill_name:   要注入的 Skill 名称。

        Returns:
            注入了 Skill 内容的完整 system prompt 字符串。
        """
        skill_content = self.load_skill(skill_name)

        # 用分隔符清晰区分 base prompt 和 Skill 注入部分
        injected_prompt = (
            f"{base_prompt}\n\n"
            f"---\n"
            f"# Skill: {skill_name}\n\n"
            f"{skill_content}"
        )
        return injected_prompt

    # ── 核心方法 3：列出可用的 Skill ─────────────────────────────────────────
    def list_skills(self) -> list:
        """扫描 skills_dir，返回所有可用 Skill 的名称列表。

        Returns:
            Skill 名称列表（文件名不含 .md 后缀）。
        """
        if not os.path.exists(self.skills_dir):
            return []

        skills = []
        for filename in sorted(os.listdir(self.skills_dir)):
            if filename.endswith(".md"):
                skill_name = filename[:-3]   # 去掉 ".md" 后缀
                skills.append(skill_name)

        return skills

    # ── 辅助方法：读取 frontmatter 元数据（用于展示）────────────────────────
    def get_skill_meta(self, skill_name: str) -> dict:
        """解析 Skill 文件的 frontmatter，返回 name/description 等元数据。

        Args:
            skill_name: Skill 名称。

        Returns:
            包含 name、description 等字段的字典。
        """
        skill_path = os.path.join(self.skills_dir, f"{skill_name}.md")

        if not os.path.exists(skill_path):
            return {}

        with open(skill_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        meta = {}
        # 提取 frontmatter 块
        frontmatter_match = re.match(
            r'^---\s*\n(.*?)^---\s*\n',
            raw_content,
            re.DOTALL | re.MULTILINE
        )
        if frontmatter_match:
            frontmatter_text = frontmatter_match.group(1)
            # 逐行解析简单的 key: value 格式（不依赖 PyYAML）
            for line in frontmatter_text.splitlines():
                line = line.strip()
                if ":" in line:
                    key, _, value = line.partition(":")
                    meta[key.strip()] = value.strip()

        return meta


# =============================================================================
# Part 3：展示 Skill 文件和 SkillLoader 的使用
# =============================================================================

def demo_skill_loader():
    """演示 SkillLoader 的基本用法：列出、加载、查看元数据。"""

    print("=" * 60)
    print("Part 3：SkillLoader 基本用法")
    print("=" * 60)

    # Skill 文件存放在本模块目录下的 skills/ 子目录
    # 路径：03_mcp_skill_command/skills/
    skills_dir = os.path.join(os.path.dirname(__file__), "skills")
    loader = SkillLoader(skills_dir)

    # ── 3.1 列出可用 Skill ────────────────────────────────────────────────────
    available_skills = loader.list_skills()
    print(f"\n扫描目录：{skills_dir}")
    print(f"发现 {len(available_skills)} 个 Skill：{available_skills}")

    if not available_skills:
        print("  （暂无 Skill 文件，请先创建 skills/code-review.md）")
        return

    # ── 3.2 查看元数据（frontmatter）────────────────────────────────────────
    print("\n── Skill 元数据（frontmatter）──────────────────────────────")
    for skill_name in available_skills:
        meta = loader.get_skill_meta(skill_name)
        print(f"  Skill: {skill_name}")
        for k, v in meta.items():
            print(f"    {k}: {v}")

    # ── 3.3 加载 Skill 内容（去掉 frontmatter）───────────────────────────────
    skill_name = "code-review"
    print(f"\n── 加载 Skill 内容（{skill_name}.md，去掉 frontmatter 后）────────")
    skill_content = loader.load_skill(skill_name)
    print(skill_content)

    # ── 3.4 展示注入后的完整 system prompt ───────────────────────────────────
    base_prompt = "你是一个 Python 编程助手，帮助用户解决代码问题。"
    enhanced_prompt = loader.inject_into_system_prompt(base_prompt, skill_name)

    print(f"\n── 注入 Skill 后的完整 system prompt ──────────────────────")
    print(f"[原始 system prompt]\n{base_prompt}\n")
    print(f"[注入后的 system prompt（共 {len(enhanced_prompt)} 字符）]")
    print(enhanced_prompt)

    print("""
SkillLoader 核心流程小结：
  1. list_skills()              → 扫描目录，发现可用 Skill
  2. load_skill(name)           → 读取 .md，剥离 frontmatter，返回纯指令内容
  3. inject_into_system_prompt  → 把 Skill 内容追加到 base_prompt
  → 最终 Claude 接收到的是"增强版 system prompt"，包含 Skill 的专业指令
""")


# =============================================================================
# Part 4：实战 —— 用 Skill 增强 system prompt，Claude 进行代码审查
# =============================================================================

def demo_skill_enhanced_code_review():
    """实战演示：加载 code-review Skill，让 Claude 审查一段有问题的 Python 代码。

    展示 Skill 注入对 Claude 输出质量的影响：
      - 不带 Skill：Claude 给出普通回答
      - 带 Skill：Claude 按 Skill 定义的结构化格式审查代码
    """

    print("=" * 60)
    print("Part 4：实战 —— Skill 增强代码审查")
    print("=" * 60)

    # ── 待审查的代码（故意埋入多个问题）───────────────────────────────────────
    # 问题清单（供对照）：
    #   1. 硬编码密码（安全性：高危）
    #   2. SQL 拼接字符串（安全性：SQL 注入风险）
    #   3. 函数名不符合命名规范（代码质量）
    #   4. 循环内重复执行 len()（性能）
    #   5. 函数职责不单一（代码质量）

    buggy_code = textwrap.dedent("""\
        import sqlite3

        PASSWORD = "admin123"           # 问题1：密码硬编码
        DB_PATH = "users.db"

        def GetUserData(username):      # 问题3：函数名应为 get_user_data（snake_case）
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # 问题2：SQL 字符串拼接，存在注入风险
            query = "SELECT * FROM users WHERE username = '" + username + "'"
            cursor.execute(query)

            results = cursor.fetchall()
            conn.close()

            # 问题4：循环内重复调用 len(results)（应在循环外缓存）
            processed = []
            for i in range(len(results)):   # 可改为 for row in results
                if i < len(results):        # 多余判断，且重复调用 len()
                    processed.append(results[i])

            # 问题5：函数职责不单一——既查数据库，又打印日志，又返回数据
            print(f"查询到 {len(processed)} 条记录")
            return processed
    """)

    print(f"\n待审查的 Python 代码：\n")
    print(buggy_code)

    # ── 初始化 SkillLoader ────────────────────────────────────────────────────
    skills_dir = os.path.join(os.path.dirname(__file__), "skills")
    loader = SkillLoader(skills_dir)

    # ── 场景 A：不带 Skill 的普通审查 ────────────────────────────────────────
    print("─" * 60)
    print("场景 A：不带 Skill 的普通代码审查（无结构化要求）")
    print("─" * 60)

    base_system_prompt = "你是一个 Python 编程助手，帮助用户进行代码审查。"

    print(f"\n[system prompt]\n{base_system_prompt}\n")
    print("正在调用 Claude（不带 Skill）...\n")

    response_a = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=base_system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"请审查以下 Python 代码，指出存在的问题：\n\n```python\n{buggy_code}\n```"
            }
        ],
    )

    answer_a = "".join(
        block.text for block in response_a.content if hasattr(block, "text")
    )
    print(f"[Claude 回答（普通模式）]\n{answer_a}")
    print(f"\nToken 消耗：输入 {response_a.usage.input_tokens}，输出 {response_a.usage.output_tokens}")

    # ── 场景 B：注入 code-review Skill 后的结构化审查 ─────────────────────────
    print("\n" + "─" * 60)
    print("场景 B：注入 code-review Skill 后的结构化审查")
    print("─" * 60)

    # 用 SkillLoader 注入 Skill，获得增强版 system prompt
    enhanced_system_prompt = loader.inject_into_system_prompt(
        base_system_prompt,
        "code-review"
    )

    print(f"\n[system prompt（注入 Skill 后，共 {len(enhanced_system_prompt)} 字符）]")
    # 只打印前 200 字，避免输出过长
    preview = enhanced_system_prompt[:200].replace("\n", "\\n")
    print(f"  {preview}...")
    print(f"  （完整内容已包含 code-review Skill 的审查指南）\n")

    print("正在调用 Claude（注入 code-review Skill）...\n")

    response_b = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=enhanced_system_prompt,     # 注入 Skill 后的增强版 system prompt
        messages=[
            {
                "role": "user",
                "content": f"请审查以下 Python 代码，指出存在的问题：\n\n```python\n{buggy_code}\n```"
            }
        ],
    )

    answer_b = "".join(
        block.text for block in response_b.content if hasattr(block, "text")
    )
    print(f"[Claude 回答（Skill 增强模式）]\n{answer_b}")
    print(f"\nToken 消耗：输入 {response_b.usage.input_tokens}，输出 {response_b.usage.output_tokens}")

    # ── 对比总结 ──────────────────────────────────────────────────────────────
    print("""
─ 对比总结 ──────────────────────────────────────────────────
  场景 A（无 Skill）：Claude 给出通用审查，格式自由，可能遗漏某些维度。
  场景 B（有 Skill）：Claude 按 Skill 定义的安全性/性能/代码质量三个维度
                      输出结构化报告，每个问题包含严重程度/位置/修复建议。

  核心结论：
  Skill 的作用 = 注入"专家知识 + 输出规范"到 system prompt
  → Claude 在执行任务时有更明确的工作准则
  → 输出结果更一致、更专业、更易复用
""")


# =============================================================================
# Part 5：Skill 系统在 Claude Code 中的完整使用方式
# =============================================================================

def show_claude_code_skill_usage():
    """展示在 Claude Code（CLI）中如何创建和使用 Skill。"""

    print("=" * 60)
    print("Part 5：在 Claude Code 中使用 Skill")
    print("=" * 60)

    skill_create_example = textwrap.dedent("""\
        # 步骤 1：创建 Skill 文件（全局 Skill）
        mkdir -p ~/.claude/skills

        cat > ~/.claude/skills/code-review.md << 'EOF'
        ---
        name: code-review
        description: Python 代码审查工作流
        ---

        # Python 代码审查指南

        请按以下维度审查代码：
        ## 1. 安全性
        ## 2. 性能
        ## 3. 代码质量
        EOF

        # 步骤 2：在 Claude Code 中触发 Skill
        # 方式一：slash command（如果 Skill 注册了 command）
        /code-review

        # 方式二：让 Claude Code 自动根据任务类型加载对应 Skill
        # （Claude Code 会根据 description 字段匹配当前任务）
        > 帮我审查这段代码
    """)

    print("\nClaude Code 中创建和使用 Skill 的完整流程：\n")
    print(skill_create_example)

    project_structure = textwrap.dedent("""\
        my_project/
        ├── .claude/
        │   └── skills/
        │       ├── code-review.md    ← 代码审查 Skill
        │       ├── doc-gen.md        ← 文档生成 Skill
        │       └── test-writer.md    ← 测试编写 Skill
        ├── src/
        │   └── main.py
        └── README.md
    """)

    print("推荐的项目级 Skill 目录结构：\n")
    print(project_structure)

    print("""
Skill 设计最佳实践：
  1. 一个 Skill 只做一件事（职责单一）
  2. description 字段清晰描述适用场景（Claude Code 据此自动匹配）
  3. 在 Skill 内明确"输出格式"要求，确保结果可预期
  4. 项目级 Skill 优先于全局 Skill（便于项目定制化）
  5. 将 Skill 文件纳入版本控制，团队共享工作规范

本文件创建的 Skill 文件位置：
  03_mcp_skill_command/skills/code-review.md
  （类比真实场景中的 <project>/.claude/skills/code-review.md）
""")


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("Claude Code Skill 系统：概念与 Python 实战")
    print("=" * 60)

    # Part 1：Skill 概念和目录结构说明
    show_skill_concepts()

    # Part 2 + 3：SkillLoader 定义在类中，这里演示其用法
    demo_skill_loader()

    # Part 4：实战 —— Skill 增强代码审查（调用 Claude API）
    demo_skill_enhanced_code_review()

    # Part 5：在 Claude Code 中使用 Skill 的完整方式
    show_claude_code_skill_usage()

    print("=" * 60)
    print("Skill 系统学习完成！")
    print("下一步：尝试创建自己的 Skill 文件，")
    print("  放入 ~/.claude/skills/ 或 <project>/.claude/skills/")
    print("  让 Claude Code 在对应任务中自动加载你的专属工作流。")
    print("=" * 60)


if __name__ == "__main__":
    main()
