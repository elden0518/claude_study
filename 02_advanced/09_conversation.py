"""
多轮对话管理完整演示
====================
本文件演示如何用 ConversationManager 封装多轮对话，涵盖：

核心知识点：
1. ConversationManager 类 — 封装对话历史，提供统一的 chat 接口
2. 上下文窗口管理 — 滑动窗口 + token 粗略估算两种策略
3. 对话历史持久化 — JSON 文件读写（save / load）
4. 模拟多轮对话 — 验证 Claude 能跨轮次记住上下文信息
"""

import sys
import os
import json

# 解决 Windows 中文乱码问题
sys.stdout.reconfigure(encoding='utf-8')

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
MODEL = "ppio/pa/claude-sonnet-4-6"

# ============================================================
# 核心类：ConversationManager
# ============================================================

class ConversationManager:
    """
    封装多轮对话的管理类。

    维护一个 messages 列表，每次 chat() 调用都会：
    1. 把用户消息追加进历史
    2. 发送完整历史给 Claude
    3. 把 Claude 的回复追加进历史
    4. 触发滑动窗口截断（_trim_history）
    """

    def __init__(self, system_prompt: str = "", max_messages: int = 20):
        """
        参数：
            system_prompt  — 系统提示词，控制 Claude 的角色和行为
            max_messages   — 滑动窗口最大消息条数（user + assistant 合计）
        """
        self.messages: list[dict] = []
        self.system_prompt = system_prompt
        self.max_messages = max_messages

    # ----------------------------------------------------------
    # 基础操作
    # ----------------------------------------------------------

    def add_user_message(self, text: str):
        """手动追加用户消息（不调用 API）"""
        self.messages.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str):
        """手动追加 assistant 消息（不调用 API）"""
        self.messages.append({"role": "assistant", "content": text})

    def clear(self):
        """清空全部对话历史"""
        self.messages.clear()
        print("  [历史已清空]")

    # ----------------------------------------------------------
    # 核心：发送消息，获取回复
    # ----------------------------------------------------------

    def chat(self, user_input: str) -> str:
        """
        发送用户消息，返回 Claude 的回复文本。

        流程：
          1. 追加用户消息到历史
          2. 调用 API（携带全部历史）
          3. 追加 Claude 回复到历史
          4. 滑动窗口截断旧消息
        """
        self.add_user_message(user_input)

        kwargs = {
            "model": MODEL,
            "max_tokens": 1024,
            "messages": self.messages,
        }
        if self.system_prompt:
            kwargs["system"] = self.system_prompt

        response = client.messages.create(**kwargs)
        reply = response.content[0].text

        self.add_assistant_message(reply)
        self._trim_history()

        return reply

    # ----------------------------------------------------------
    # 策略 1：滑动窗口（保留最近 N 条消息）
    # ----------------------------------------------------------

    def _trim_history(self):
        """
        滑动窗口截断：当消息总数超过 max_messages 时，
        从头部删除最旧的消息（保持 user/assistant 交替结构，
        始终以 user 消息开头）。
        """
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)
        # 确保第一条消息是 user（Claude 要求 messages 以 user 开头）
        while self.messages and self.messages[0]["role"] != "user":
            self.messages.pop(0)

    # ----------------------------------------------------------
    # 策略 2：token 粗略估算（字符数 / 4）
    # ----------------------------------------------------------

    def _estimate_tokens(self) -> int:
        """粗略估算当前历史的 token 数（用字符数 / 4 近似）"""
        total_chars = sum(len(m["content"]) for m in self.messages)
        return total_chars // 4

    def trim_by_token_limit(self, max_tokens: int = 2000):
        """
        token 估算截断策略：当估算 token 超过 max_tokens 时，
        逐对（user + assistant）从头部删除旧消息。

        与 _trim_history 不同，这里以 token 为单位而非消息条数。
        """
        while self._estimate_tokens() > max_tokens and len(self.messages) >= 2:
            # 删除最旧的一对消息（user + assistant）
            self.messages.pop(0)
            if self.messages and self.messages[0]["role"] == "assistant":
                self.messages.pop(0)
        print(f"  [token 截断后] 剩余约 {self._estimate_tokens()} tokens，"
              f"{len(self.messages)} 条消息")

    # ----------------------------------------------------------
    # 持久化：保存 / 加载 JSON
    # ----------------------------------------------------------

    def save(self, filepath: str):
        """
        保存对话历史到 JSON 文件。

        JSON 结构：
          {
            "system_prompt": "...",
            "max_messages": 20,
            "messages": [{"role": "user", "content": "..."}, ...]
          }
        """
        data = {
            "system_prompt": self.system_prompt,
            "max_messages": self.max_messages,
            "messages": self.messages,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  [已保存] {filepath}（{len(self.messages)} 条消息）")

    def load(self, filepath: str):
        """
        从 JSON 文件加载对话历史（会覆盖当前历史）。
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.system_prompt = data.get("system_prompt", "")
        self.max_messages = data.get("max_messages", 20)
        self.messages = data.get("messages", [])
        print(f"  [已加载] {filepath}（{len(self.messages)} 条消息）")

    # ----------------------------------------------------------
    # 辅助：打印当前历史摘要
    # ----------------------------------------------------------

    def print_history_summary(self):
        print(f"\n  --- 当前历史：{len(self.messages)} 条消息，"
              f"约 {self._estimate_tokens()} tokens ---")
        for i, m in enumerate(self.messages):
            role = "👤 用户" if m["role"] == "user" else "🤖 Claude"
            snippet = m["content"][:60].replace("\n", " ")
            ellipsis = "…" if len(m["content"]) > 60 else ""
            print(f"  [{i+1}] {role}: {snippet}{ellipsis}")


# ============================================================
# 演示 1：上下文记忆（Claude 能记住之前说的内容）
# ============================================================

def demo_context_memory():
    print("=" * 60)
    print("演示 1：多轮对话上下文记忆")
    print("=" * 60)

    manager = ConversationManager(
        system_prompt="你是一个友好的助手，请用简洁的中文回答。",
        max_messages=20,
    )

    # 第 1 轮：自我介绍
    print("\n[第 1 轮] 用户：我的名字叫小明，我喜欢 Python")
    reply1 = manager.chat("我的名字叫小明，我喜欢 Python")
    print(f"Claude：{reply1}\n")

    # 第 2 轮：测试是否记得名字
    print("[第 2 轮] 用户：我刚才说我叫什么？")
    reply2 = manager.chat("我刚才说我叫什么？")
    print(f"Claude：{reply2}\n")

    # 第 3 轮：测试是否记得偏好
    print("[第 3 轮] 用户：我喜欢什么编程语言？")
    reply3 = manager.chat("我喜欢什么编程语言？")
    print(f"Claude：{reply3}\n")

    # 第 4 轮：追问（验证更长的上下文链）
    print("[第 4 轮] 用户：能帮我推荐一个适合我的 Python 学习项目吗？")
    reply4 = manager.chat("能帮我推荐一个适合我的 Python 学习项目吗？")
    print(f"Claude：{reply4}\n")

    manager.print_history_summary()
    return manager


# ============================================================
# 演示 2：滑动窗口截断（max_messages=4）
# ============================================================

def demo_sliding_window():
    print("\n" + "=" * 60)
    print("演示 2：滑动窗口 — 只保留最近 4 条消息")
    print("=" * 60)

    manager = ConversationManager(
        system_prompt="你是一个简洁的助手，回答尽量简短。",
        max_messages=4,  # 只保留最近 4 条（2 轮）
    )

    questions = [
        "请记住：数字 A = 120",
        "请记住：数字 B = 250",
        "请记住：数字 C = 300",
        "请记住：数字 D = 400",
        "A、B、C、D 分别是多少？（你还记得吗）",
    ]

    for q in questions:
        print(f"\n[用户] {q}")
        reply = manager.chat(q)
        print(f"[Claude] {reply}")
        print(f"  → 当前历史条数：{len(manager.messages)}")

    print("\n说明：由于窗口只保留 4 条消息（最近 2 轮），")
    print("      Claude 可能已忘记最早设置的数字 A，这是预期行为。")


# ============================================================
# 演示 3：token 估算截断
# ============================================================

def demo_token_trim():
    print("\n" + "=" * 60)
    print("演示 3：token 估算截断（超出 500 tokens 时删旧消息）")
    print("=" * 60)

    manager = ConversationManager(
        system_prompt="请用50字以内简短回答。",
        max_messages=50,  # 不走滑动窗口，用 token 策略控制
    )

    # 先手动塞入一些"旧"历史，模拟历史很长的情况
    manager.add_user_message("很久以前的问题 1：Python 是什么？")
    manager.add_assistant_message("Python 是一种高级编程语言，以简洁著称。" * 10)
    manager.add_user_message("很久以前的问题 2：JavaScript 是什么？")
    manager.add_assistant_message("JavaScript 是网页开发的核心脚本语言。" * 10)

    print(f"  截断前：约 {manager._estimate_tokens()} tokens，"
          f"{len(manager.messages)} 条消息")

    # 触发 token 截断
    manager.trim_by_token_limit(max_tokens=100)

    # 继续对话（使用截断后的历史）
    print("\n[用户] 现在请推荐一个入门级编程语言")
    reply = manager.chat("现在请推荐一个入门级编程语言")
    print(f"[Claude] {reply}")


# ============================================================
# 演示 4：持久化（保存 + 加载 JSON）
# ============================================================

HISTORY_FILE = "D:/claude_project/claude_study/02_advanced/conversation_history.json"

def demo_persistence():
    print("\n" + "=" * 60)
    print("演示 4：对话历史持久化（JSON 保存与加载）")
    print("=" * 60)

    # --- 第一阶段：建立对话并保存 ---
    print("\n[阶段 A] 建立对话，保存历史")
    manager_a = ConversationManager(
        system_prompt="你是一位耐心的编程老师。",
        max_messages=20,
    )
    manager_a.chat("我是一名 Python 新手，今天开始学习")
    manager_a.chat("请问 list 和 tuple 有什么区别？")
    manager_a.print_history_summary()
    manager_a.save(HISTORY_FILE)

    # --- 第二阶段：新建 manager，加载历史，继续对话 ---
    print("\n[阶段 B] 新建 manager，加载历史，验证上下文延续")
    manager_b = ConversationManager()
    manager_b.load(HISTORY_FILE)
    manager_b.print_history_summary()

    print("\n[用户（延续之前对话）] 那我什么时候应该用 tuple？")
    reply = manager_b.chat("那我什么时候应该用 tuple？")
    print(f"[Claude] {reply}")

    print("\n持久化演示完成：Claude 成功在新 session 中延续了上下文。")

    # 清理临时文件
    # if os.path.exists(HISTORY_FILE):
    #     os.remove(HISTORY_FILE)
    #     print(f"  [清理] 已删除临时文件 {HISTORY_FILE}")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("Claude 多轮对话管理演示")
    print("模型：", MODEL)
    print()

    # 演示 1：上下文记忆（最核心的演示）
    demo_context_memory()

    # 演示 2：滑动窗口
    demo_sliding_window()

    # 演示 3：token 估算截断
    demo_token_trim()

    # 演示 4：持久化
    demo_persistence()

    print("\n" + "=" * 60)
    print("全部演示完成！")
    print("=" * 60)
