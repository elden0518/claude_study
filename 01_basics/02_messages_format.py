"""
主题：Messages 格式详解 —— 理解对话结构的核心

学习目标：
  1. 掌握 system prompt 的作用与写法（定义 AI 角色和行为约束）
  2. 理解 messages 列表结构：role（user/assistant）与 content 的对应关系
  3. 学会传入多条历史消息，模拟多轮对话上下文
  4. 掌握 content 的两种写法：字符串形式 vs 列表（块）形式
  5. 学会使用 assistant pre-fill 技巧引导模型按指定格式输出

前置知识：
  - 已完成 01_hello_claude.py，了解基本 API 调用流程
  - 知道 messages.create() 的三个核心参数：model / max_tokens / messages

课程顺序：这是 01_basics 模块的第二个文件，建议按序学习。
"""

# ── 0. Windows 控制台编码修复 ──────────────────────────────────────────────────
# Windows 默认控制台编码（GBK）无法显示部分 Unicode 字符
# 强制设为 UTF-8，确保中文能正常打印
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── 1. 导入所需模块 ────────────────────────────────────────────────────────────
import os
import anthropic
from dotenv import load_dotenv

# ── 2. 加载环境变量并初始化客户端 ─────────────────────────────────────────────
load_dotenv()
client = anthropic.Anthropic()

# 统一使用的模型名称（通过代理前缀 ppio/pa/）
MODEL = "ppio/pa/claude-sonnet-4-6"


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 1：System Prompt
# ═══════════════════════════════════════════════════════════════════════════════

def demo_system_prompt():
    """
    演示 system prompt 的作用

    【是什么】
    system prompt 是一段特殊的"幕后指令"，在对话开始前传给模型。
    它不属于对话消息列表（messages），而是通过独立的 system 参数传入。

    【为什么】
    用于定义 AI 的角色、语气、输出格式、行为约束等全局规则。
    例如：让 Claude 扮演客服、只用英文回答、始终输出 JSON 等。

    【怎么用】
    client.messages.create(
        system="你是...",   # <-- 独立参数，不放进 messages 列表
        messages=[...]
    )
    """
    print("\n" + "=" * 60)
    print("Demo 1：System Prompt 的作用")
    print("=" * 60)

    # ── 对比：不使用 system prompt ───────────────────────────────────────────
    # 没有 system prompt 时，Claude 使用默认行为
    print("\n[无 system prompt] 询问 Claude 的身份：")
    response_no_sys = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[
            {"role": "user", "content": "你是谁？请用一句话回答。"}
        ]
    )
    print(f"  回复：{response_no_sys.content[0].text}")

    # ── 对比：使用 system prompt 定义角色 ────────────────────────────────────
    # system prompt 让 Claude 扮演一个特定角色，约束其回答风格
    print("\n[有 system prompt] 相同问题，使用角色扮演：")
    response_with_sys = client.messages.create(
        model=MODEL,
        max_tokens=256,
        # system 参数：独立于 messages 之外，优先级最高
        # 最佳实践：清晰描述角色、职责、输出约束
        system="你是一位严格的数学老师，名叫李老师。无论被问什么，你总是先说'同学们注意了！'，然后用简洁的语言作答。",
        messages=[
            {"role": "user", "content": "你是谁？请用一句话回答。"}
        ]
    )
    print(f"  回复：{response_with_sys.content[0].text}")

    print("\n结论：system prompt 可以显著改变 Claude 的行为和角色定位。")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 2：Messages 列表结构
# ═══════════════════════════════════════════════════════════════════════════════

def demo_messages_structure():
    """
    演示基本 messages 列表结构

    【是什么】
    messages 是一个列表，每条消息是一个字典，包含两个字段：
      - role：消息发送者的角色，只能是 "user" 或 "assistant"
      - content：消息的实际内容

    【为什么】
    Claude API 是无状态的——每次调用都需要完整传入对话历史。
    通过 role 字段区分"谁说了什么"，模型才能理解对话上下文。

    【怎么用】
    messages = [
        {"role": "user",      "content": "用户说的话"},
        {"role": "assistant", "content": "Claude 上一轮的回复"},
        {"role": "user",      "content": "用户继续说的话"},
    ]
    注意：messages 列表必须以 "user" 消息开头，且 role 必须交替出现。
    """
    print("\n" + "=" * 60)
    print("Demo 2：Messages 列表结构（role 与 content）")
    print("=" * 60)

    # ── 最简单的单条消息结构 ─────────────────────────────────────────────────
    # 只有一条 user 消息：这是最基本的调用方式
    print("\n[单条 user 消息] 最基本结构：")
    messages_single = [
        {
            "role": "user",           # 角色：用户
            "content": "1 + 1 等于几？"  # 内容：用户的提问
        }
    ]
    print(f"  messages 结构：{messages_single}")

    response = client.messages.create(
        model=MODEL,
        max_tokens=64,
        messages=messages_single
    )
    # 响应中 role 固定为 "assistant"，表示这是 Claude 的回答
    print(f"  Claude（assistant）回复：{response.content[0].text.strip()}")

    # ── role 的两种取值说明 ──────────────────────────────────────────────────
    print("\n[role 说明]")
    print("  'user'      → 代表用户/人类发出的消息")
    print("  'assistant' → 代表 Claude 发出的消息（用于传入历史回复或 pre-fill）")
    print("  规则：messages 列表必须以 user 开头，且 user/assistant 交替出现")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 3：多轮对话历史
# ═══════════════════════════════════════════════════════════════════════════════

def demo_conversation_history():
    """
    演示多轮历史消息传入

    【是什么】
    Claude API 本身是无状态的——它不会自动记住之前的对话。
    要实现"多轮对话"，需要在每次请求时，把完整的历史消息列表一起传入。

    【为什么】
    这种设计让开发者完全掌控上下文：
      - 可以裁剪过长的历史（节省 token）
      - 可以修改历史消息（纠正错误）
      - 可以在不同会话之间复用历史

    【怎么用】
    将历史的 user/assistant 消息按顺序放入列表，最后一条必须是 user 消息。
    """
    print("\n" + "=" * 60)
    print("Demo 3：多轮对话历史传入")
    print("=" * 60)

    # ── 模拟已发生过的对话历史 ───────────────────────────────────────────────
    # 假设用户和 Claude 已经聊过两轮，现在第三轮需要带入上下文
    # 这些历史消息通常来自你自己的数据库或内存存储
    conversation_history = [
        # 第一轮：用户介绍自己
        {
            "role": "user",
            "content": "你好！我叫小明，我是一名 Python 初学者。"
        },
        # 第一轮：Claude 的回复（这是之前 API 返回的内容）
        {
            "role": "assistant",
            "content": "你好，小明！很高兴认识你。Python 是一门非常适合初学者的语言，加油！"
        },
        # 第二轮：用户提问
        {
            "role": "user",
            "content": "我最近在学列表和字典，感觉字典有点难理解。"
        },
        # 第二轮：Claude 的回复
        {
            "role": "assistant",
            "content": "字典是 Python 中的键值对容器，就像真实的字典一样：通过'词（key）'查找'解释（value）'。举个例子：student = {'name': '小明', 'age': 18}。"
        },
        # 第三轮：当前用户的新消息（继续追问）
        # Claude 需要结合上下文（知道用户是小明、在学 Python）来回答
        {
            "role": "user",
            "content": "明白了！那我之前说我叫什么名字？另外字典怎么添加新键值对？"
        }
    ]

    print(f"\n传入历史消息共 {len(conversation_history)} 条（{(len(conversation_history)+1)//2} 轮对话）")
    print("最新问题：", conversation_history[-1]["content"])

    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=conversation_history  # 完整历史一起传入
    )

    print(f"\nClaude 的回复（能记住上下文）：")
    print(f"  {response.content[0].text.strip()}")
    print("\n注意：Claude 正确记住了'小明'这个名字，证明历史上下文生效了。")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 4：content 的两种形式
# ═══════════════════════════════════════════════════════════════════════════════

def demo_content_formats():
    """
    演示 content 字符串 vs 列表两种写法

    【是什么】
    messages 中的 content 字段支持两种格式：
      1. 字符串（String）：最简单，直接写文本内容
         content = "你好"
      2. 列表（List of Blocks）：更灵活，每个块有 type 和对应字段
         content = [{"type": "text", "text": "你好"}]

    【为什么】
    - 字符串格式：简单文本消息，日常使用最常见
    - 列表格式：支持混合内容（文本 + 图片 + 工具结果等），是更底层的表示
      当消息中需要包含多种类型内容（如图文混排）时，必须用列表格式

    【怎么用】
    两种格式效果完全相同（纯文本时），API 内部会统一处理。
    推荐：纯文本用字符串，有图片/工具结果时用列表。
    """
    print("\n" + "=" * 60)
    print("Demo 4：content 的两种格式（字符串 vs 列表）")
    print("=" * 60)

    question = "用一句话解释什么是变量？"

    # ── 格式 1：字符串形式（最常用）────────────────────────────────────────
    # 直接将文本内容赋给 content，简洁直观
    print(f"\n[格式 1：字符串] content = \"{question}\"")
    response_str = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[
            {
                "role": "user",
                "content": question   # 字符串形式
            }
        ]
    )
    print(f"  回复：{response_str.content[0].text.strip()}")

    # ── 格式 2：列表（块）形式 ────────────────────────────────────────────
    # content 是一个列表，每个元素是一个"内容块"（Content Block）
    # 对于纯文本，块的 type 为 "text"，内容放在 "text" 字段
    # 其他 type 还有："image"（图片）、"tool_result"（工具结果）等
    print(f"\n[格式 2：列表块] content = [{{\"type\": \"text\", \"text\": \"...\"}}]")
    response_list = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[
            {
                "role": "user",
                "content": [           # 列表形式
                    {
                        "type": "text",    # 块类型：纯文本
                        "text": question   # 实际文本内容
                    }
                ]
            }
        ]
    )
    print(f"  回复：{response_list.content[0].text.strip()}")

    # ── 列表形式的真正优势：多块内容 ─────────────────────────────────────
    # 列表形式可以在一条消息中包含多个文本块（实际场景中常用于拼接动态内容）
    print("\n[列表形式多块] 一条消息中包含多个文本块：")
    response_multi = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "背景信息：我正在学 Python。\n"},
                    {"type": "text", "text": "问题：什么是函数？"}
                    # 扩展：如果有图片，可以在这里加 {"type": "image", ...}
                ]
            }
        ]
    )
    print(f"  回复：{response_multi.content[0].text.strip()}")
    print("\n两种格式在纯文本时效果相同；列表格式在多媒体场景下不可替代。")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 5：Assistant Pre-fill（引导输出）
# ═══════════════════════════════════════════════════════════════════════════════

def demo_assistant_prefill():
    """
    演示 assistant pre-fill 引导输出格式

    【是什么】
    Assistant pre-fill 是指在 messages 列表的末尾添加一条 "role": "assistant"
    的消息，但内容只写"开头部分"，让 Claude 从这个开头继续补全。

    【为什么】
    这是一种强力的输出格式控制技巧：
      - 强制 Claude 直接以指定格式/词语开始输出，跳过前置废话
      - 常用于：强制输出 JSON、让回答以特定词开头、控制语气等
      - 比 system prompt 中的格式要求更精确可靠

    【怎么用】
    在 messages 末尾添加 assistant 消息（不是最后一条 user 消息之后，而是作为最后一条）：
    messages = [
        {"role": "user",      "content": "你的问题"},
        {"role": "assistant", "content": "{"}   # <-- pre-fill：告诉 Claude 从 { 开始
    ]
    注意：pre-fill 只能是 messages 的最后一条消息，且不能是空字符串。
    """
    print("\n" + "=" * 60)
    print("Demo 5：Assistant Pre-fill（引导输出格式）")
    print("=" * 60)

    # ── 对比：不使用 pre-fill ─────────────────────────────────────────────
    # 不使用 pre-fill 时，Claude 可能会在 JSON 前加解释性文字
    print("\n[无 pre-fill] 要求输出 JSON，Claude 可能会附带解释：")
    response_no_prefill = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": "把以下信息转成 JSON 格式：姓名张三，年龄25，城市北京。"
            }
        ]
    )
    print(f"  回复：{response_no_prefill.content[0].text.strip()[:200]}")

    # ── 使用 pre-fill 强制从 { 开始 ──────────────────────────────────────
    # 在 messages 末尾加一条 assistant 消息，内容为 "{"
    # Claude 会把它当成自己已经开始输出，然后直接续写后面的 JSON 内容
    print("\n[有 pre-fill] 用 '{' 开头，强制直接输出 JSON：")
    response_with_prefill = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": "把以下信息转成 JSON 格式：姓名张三，年龄25，城市北京。"
            },
            # pre-fill：assistant 消息作为最后一条，以 { 开头
            # Claude 会把这条消息理解为"我已经开始输出了"，然后续写
            {
                "role": "assistant",
                "content": "{"   # 告诉 Claude：你的输出从 { 这里开始
            }
        ]
    )
    # 注意：API 返回的是 Claude 续写的部分，不含我们传入的 pre-fill "{" 本身
    # 所以需要手动拼接完整结果
    prefill_start = "{"
    continuation = response_with_prefill.content[0].text.strip()
    full_json = prefill_start + continuation
    print(f"  完整输出（pre-fill + 续写）：{full_json}")

    # ── 另一个用例：强制以特定词语开头 ───────────────────────────────────
    print("\n[pre-fill 用例 2] 强制回答以'答：'开头，跳过废话：")
    response_direct = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[
            {
                "role": "user",
                "content": "Python 和 Java 哪个更适合初学者？"
            },
            {
                "role": "assistant",
                "content": "答："   # 强制以"答："开头，直接给出结论
            }
        ]
    )
    print(f"  回复：答：{response_direct.content[0].text.strip()}")
    print("\npre-fill 技巧让你精确控制 Claude 的输出起点，是格式化输出的利器。")


# ═══════════════════════════════════════════════════════════════════════════════
# Main：依次运行所有 Demo
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("02_messages_format.py —— Messages 格式详解")
    print("=" * 60)
    print("本文件将依次演示 5 个知识点，每个都会真实调用 Claude API。")

    demo_system_prompt()          # Demo 1：system prompt
    demo_messages_structure()     # Demo 2：messages 列表结构
    demo_conversation_history()   # Demo 3：多轮对话历史
    demo_content_formats()        # Demo 4：content 两种格式
    demo_assistant_prefill()      # Demo 5：assistant pre-fill

    print("\n" + "=" * 60)
    print("全部 Demo 运行完毕！")
    print("下一步：学习 03_streaming.py 了解流式输出。")
    print("=" * 60)


if __name__ == "__main__":
    main()
