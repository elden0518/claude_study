"""
Prompt Engineering 技巧演示
============================
本文件演示 5 种核心 prompt 工程技巧：
1. Zero-shot vs Few-shot：示例数量对输出质量的影响
2. Chain-of-Thought (CoT)：逐步推理提升准确率
3. XML 标签结构化复杂 prompt
4. 角色扮演 system prompt
5. 负向提示：明确告诉模型不要做什么

认知科学背景：
- 大模型本质上是"概率续写"引擎，提供的上下文越丰富、越结构化，
  模型就能更准确地"猜测"你期望的输出分布。
- Few-shot 示例相当于给模型"打样"，CoT 让模型激活推理路径，
  XML 结构帮助模型区分信息层级，角色扮演激活特定知识域，
  负向提示则缩小了输出空间，减少无效内容。
"""

import sys
import os

# 解决 Windows 中文乱码问题
sys.stdout.reconfigure(encoding='utf-8')

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    base_url=os.environ.get("ANTHROPIC_BASE_URL"),
)

MODEL = "ppio/pa/claude-sonnet-4-6"

# 工具函数：统一调用 API 并打印结果
def call_claude(system_prompt: str, user_message: str, label: str = "") -> str:
    """调用 Claude API，打印 prompt 和回复，返回回复文本。"""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    if system_prompt:
        print(f"\n[System Prompt]\n{system_prompt}")

    print(f"\n[User Message]\n{user_message}")

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    reply = response.content[0].text
    print(f"\n[Claude 回复]\n{reply}")
    return reply


# ─────────────────────────────────────────────
# Demo 1: Zero-shot vs Few-shot
# ─────────────────────────────────────────────
def demo_zero_shot_vs_few_shot():
    """
    Zero-shot：不提供任何示例，直接让模型完成任务。
    Few-shot：提供 2-3 个高质量示例，让模型学习"格式+逻辑"。

    为什么 Few-shot 更好？
    ▸ 示例相当于运行时"微调"——无需训练，只靠上下文窗口。
    ▸ 模型看到示例后，会对输出格式、分析粒度形成一致预期。
    ▸ 特别适合需要固定输出结构（JSON / 评级 / 标签）的场景。
    """
    task = "分析以下句子的情感（积极/消极/中性），并给出置信度（高/中/低）。\n\n句子：今天的会议开得莫名其妙，浪费了三个小时。"

    # ── 1A: Zero-shot ──────────────────────────────────
    call_claude(
        system_prompt="你是一个情感分析专家。",
        user_message=task,
        label="Demo 1A：Zero-shot 情感分析（无示例）",
    )

    # ── 1B: Few-shot ──────────────────────────────────
    # 通过 2 个精心挑选的示例，告诉模型：
    #   ① 输出格式（情感 + 置信度 + 关键词）
    #   ② 分析粒度（为什么这样判断）
    few_shot_prompt = """你是一个情感分析专家。请按照以下示例格式分析句子情感。

示例 1：
句子：终于拿到了梦想公司的 offer！
分析：
- 情感：积极
- 置信度：高
- 关键词：终于、梦想、offer
- 理由：感叹号 + "终于""梦想"等词强烈表达了兴奋与满足感。

示例 2：
句子：快递还没到，也没有物流更新。
分析：
- 情感：中性（轻微消极）
- 置信度：中
- 关键词：没到、没有更新
- 理由：陈述事实，带轻微不满但无强烈情绪词。

现在请分析用户给出的句子，严格遵循上述格式。"""

    call_claude(
        system_prompt=few_shot_prompt,
        user_message=task,
        label="Demo 1B：Few-shot 情感分析（含 2 个示例）",
    )


# ─────────────────────────────────────────────
# Demo 2: Chain-of-Thought (CoT)
# ─────────────────────────────────────────────
def demo_chain_of_thought():
    """
    Chain-of-Thought（思维链）：让模型在给出答案前，
    先把推理过程逐步写出来。

    为什么 CoT 有效？
    ▸ 对于需要多步推理的题目，模型"写下中间步骤"
      相当于为后续 token 创造了更准确的上下文。
    ▸ 就像人类打草稿：先算中间结果，再得出答案，
      出错概率远低于"拍脑袋直接说答案"。
    ▸ 研究表明，仅凭"Let's think step by step"这句话，
      模型在算术推理上的准确率可提升 40%+（Kojima et al., 2022）。
    """
    problem = (
        "一个水箱，每小时注水 30 升，同时以每小时 12 升的速度漏水。"
        "水箱现在有 50 升水，最多能装 200 升。"
        "请问多少小时后水箱会装满？"
    )

    # ── 2A: 直接回答（无 CoT）──────────────────────────
    call_claude(
        system_prompt="你是一个数学助手，直接给出答案，不需要解释过程。",
        user_message=problem,
        label="Demo 2A：直接回答（无 Chain-of-Thought）",
    )

    # ── 2B: Chain-of-Thought ──────────────────────────
    cot_system = (
        "你是一个数学助手。在回答之前，请先一步一步地分析问题，"
        "列出每一个推理步骤，最后再给出最终答案。"
    )
    cot_user = problem + "\n\n让我们一步一步思考："

    call_claude(
        system_prompt=cot_system,
        user_message=cot_user,
        label="Demo 2B：Chain-of-Thought（逐步推理）",
    )


# ─────────────────────────────────────────────
# Demo 3: XML 标签结构化 prompt
# ─────────────────────────────────────────────
def demo_xml_structure():
    """
    用 XML 标签把 prompt 的不同部分（背景、指令、示例、输出格式）
    清晰地分隔开来。

    为什么 XML 结构有效？
    ▸ Claude 在训练时大量接触了结构化文档（HTML/XML/Markdown），
      能够高效解析标签边界，准确区分"背景信息"与"执行指令"。
    ▸ 标签让长 prompt 层次分明，减少模型混淆不同部分的概率。
    ▸ 特别适合复杂任务：RAG、多轮对话摘要、代码审查报告等。
    """
    xml_prompt = """<context>
你正在帮助一家初创公司的产品经理分析用户反馈。
公司产品是一款 AI 写作助手，目前处于 Beta 阶段。
</context>

<instructions>
请对下方用户反馈进行结构化分析：
1. 提取主要问题点（最多 3 条）
2. 识别用户情感（积极/消极/混合）
3. 给出优先级最高的改进建议（1 条，具体可执行）
</instructions>

<examples>
输入反馈："速度很快，但有时候生成的内容会跑题，希望能加个重新生成按钮。"
输出：
- 问题点：①内容跑题 ②缺少重新生成功能
- 情感：混合（速度满意，功能有缺失）
- 改进建议：在生成结果旁添加"重新生成"按钮，优先级 P1
</examples>

<output_format>
严格按照以下格式输出，不要添加额外说明：
- 问题点：①... ②... ③...（如不足 3 条则列出实际数量）
- 情感：[积极/消极/混合] + 一句话说明原因
- 改进建议：[具体行动] + [预期效果]
</output_format>

<user_feedback>
这个工具写邮件挺顺手的，但每次打开都要重新登录，太烦了。
而且生成的中文有时候语气很奇怪，像机器翻译。整体还行吧，凑合用。
</user_feedback>"""

    call_claude(
        system_prompt="",
        user_message=xml_prompt,
        label="Demo 3：XML 标签结构化复杂 prompt",
    )


# ─────────────────────────────────────────────
# Demo 4: 角色扮演 system prompt
# ─────────────────────────────────────────────
def demo_role_playing():
    """
    通过 system prompt 为模型赋予专业角色身份。

    为什么角色扮演有效？
    ▸ 角色定义激活了模型中与该角色强相关的知识子空间。
      "Python 专家"会自然联想到最佳实践、PEP 8、性能优化等概念。
    ▸ 角色还隐含了"受众假设"：专家对专家说话 vs 专家对新手说话，
      输出的深度和术语密度截然不同。
    ▸ 角色约束可以减少模型"和稀泥"的倾向——专家会直接指出问题。
    """
    code_to_review = """
def get_user_data(user_id):
    import sqlite3
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    cursor.execute(query)
    data = cursor.fetchall()
    return data
"""

    # ── 4A: 无角色定义 ──────────────────────────────────
    call_claude(
        system_prompt="",
        user_message=f"帮我看看这段代码：\n```python{code_to_review}```",
        label="Demo 4A：无角色定义的代码审查",
    )

    # ── 4B: 专业角色定义 ──────────────────────────────────
    expert_system = """你是一位资深 Python 安全工程师，拥有 10 年以上后端开发经验。
你的代码审查风格：
- 直接、精准，不废话
- 优先指出安全漏洞，再指出性能和可读性问题
- 每个问题都提供修复后的代码示例
- 最后给出一个总体安全评分（1-10 分）"""

    call_claude(
        system_prompt=expert_system,
        user_message=f"请审查以下 Python 代码，按你的专业标准给出反馈：\n```python{code_to_review}```",
        label="Demo 4B：Python 安全工程师角色扮演",
    )


# ─────────────────────────────────────────────
# Demo 5: 负向提示
# ─────────────────────────────────────────────
def demo_negative_prompting():
    """
    负向提示：通过明确列出"禁止事项"来约束输出。

    为什么负向提示有效？
    ▸ 模型的输出空间是巨大的。正向指令定义了"目标区域"，
      负向指令则切除了不想要的子空间，让模型在更窄的范围内搜索。
    ▸ 类比：告诉厨师"做一道菜"（正向）+ "不放辣椒、不用猪肉"（负向），
      比单独给正向指令更能得到你想要的结果。
    ▸ 常见用途：禁止废话、限制技术路线、防止输出特定内容。
    """
    task = "写一个 Python 函数，找出列表中所有重复的元素，并返回去重后的重复元素列表。"

    # ── 5A: 无负向提示 ──────────────────────────────────
    call_claude(
        system_prompt="你是一个 Python 编程助手。",
        user_message=task,
        label="Demo 5A：无负向提示（可能包含解释和多种方案）",
    )

    # ── 5B: 带负向提示 ──────────────────────────────────
    negative_system = """你是一个 Python 编程助手。
严格遵守以下限制：
- 不要解释代码，只输出代码本身
- 不要使用 for 循环（使用列表推导式或集合操作）
- 不要导入任何第三方库
- 不要提供多种实现方案，只给一种最优解
- 代码中只允许有必要的行内注释（# 开头），不要写文档字符串"""

    call_claude(
        system_prompt=negative_system,
        user_message=task,
        label="Demo 5B：负向提示（禁止解释 + 禁止 for 循环）",
    )


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────
def main():
    print("\n" + "█" * 60)
    print("  Prompt Engineering 核心技巧演示")
    print("  模型：" + MODEL)
    print("█" * 60)

    print("\n\n【技巧 1】Zero-shot vs Few-shot")
    print("─" * 40)
    demo_zero_shot_vs_few_shot()

    print("\n\n【技巧 2】Chain-of-Thought (CoT)")
    print("─" * 40)
    demo_chain_of_thought()

    print("\n\n【技巧 3】XML 标签结构化 Prompt")
    print("─" * 40)
    demo_xml_structure()

    print("\n\n【技巧 4】角色扮演 System Prompt")
    print("─" * 40)
    demo_role_playing()

    print("\n\n【技巧 5】负向提示")
    print("─" * 40)
    demo_negative_prompting()

    print("\n\n" + "█" * 60)
    print("  所有演示完成！")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
