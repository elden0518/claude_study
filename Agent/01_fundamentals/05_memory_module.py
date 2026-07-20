"""
==============================================================================
第五课：Memory 模块 - 让 Agent 记住上下文
==============================================================================

【学习目标】
- 理解 Agent 记忆系统的设计原理
- 掌握短期记忆（对话上下文窗口）
- 掌握长期记忆（跨会话持久化）
- 掌握向量记忆（语义检索）
- 学会记忆压缩和摘要策略

【核心概念】
- 短期记忆 vs 长期记忆
- 上下文窗口管理
- 向量相似度检索
- 记忆压缩与摘要

【前置知识】
- 第四课：Schema 模块（Message、AgentState）

==============================================================================
"""

import json
import math
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Any

from pydantic import BaseModel, Field

# 复用第四课定义的 Message（这里简化定义以便独立运行）
from enum import Enum


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_call_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=Role.SYSTEM, content=content)


# ============================================================================
# 第一部分：为什么 Agent 需要记忆？
# ============================================================================

def why_memory_matters():
    """
    解释为什么 Agent 需要记忆系统

    LLM 本身是无状态的：每次调用都是独立的，不记得之前的对话。
    记忆系统让 Agent 能够：
    1. 记住当前对话的上下文（短期记忆）
    2. 记住跨会话的重要信息（长期记忆）
    3. 通过语义检索找到相关记忆（向量记忆）
    """

    # ── 无记忆的问题 ─────────────────────────────────────────────
    no_memory_example = """
    用户: 我叫张三
    AI:   你好张三！

    用户: 我叫什么名字？
    AI:   抱歉，我不知道你叫什么名字。  ← 忘记了！

    原因：LLM 每次调用都是独立的，没有"记住"之前的对话。
    """

    # ── 有记忆的效果 ─────────────────────────────────────────────
    with_memory_example = """
    用户: 我叫张三
    AI:   你好张三！
    [记忆系统保存: 用户名字=张三]

    用户: 我叫什么名字？
    AI:   你叫张三。  ← 从记忆中检索到了！

    用户: （第二天）我还叫张三吗？
    AI:   是的，你叫张三。  ← 从长期记忆中检索到了！
    """

    print("=" * 60)
    print("为什么 Agent 需要记忆")
    print("=" * 60)
    print(no_memory_example)
    print(with_memory_example)


# ============================================================================
# 第二部分：记忆系统架构
# ============================================================================

def show_memory_architecture():
    """
    展示记忆系统的三层架构

    ┌─────────────────────────────────────────────────────────────┐
    │                    记忆系统架构                               │
    │                                                             │
    │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │
    │  │  短期记忆    │  │  长期记忆    │  │   向量记忆        │   │
    │  │ Short-Term  │  │ Long-Term   │  │  Vector Memory   │   │
    │  │             │  │             │  │                  │   │
    │  │ 当前对话     │  │ 跨会话持久   │  │ 语义检索         │   │
    │  │ 上下文窗口   │  │ 化存储       │  │ 相似度匹配       │   │
    │  │             │  │             │  │                  │   │
    │  │ 容量: 有限   │  │ 容量: 大    │  │ 容量: 大         │   │
    │  │ 速度: 快     │  │ 速度: 中    │  │ 速度: 中         │   │
    │  │ 存储: 内存   │  │ 存储: 文件   │  │ 存储: 向量数据库  │   │
    │  ─────────────┘  └─────────────┘  └──────────────────┘   │
    └─────────────────────────────────────────────────────────────┘
    """
    print("=" * 60)
    print("记忆系统三层架构")
    print("=" * 60)
    print(show_memory_architecture.__doc__)
    print()


# ============================================================================
# 第三部分：记忆基类（base.py）
# ============================================================================
#
# 定义记忆的抽象接口，所有具体记忆实现都必须遵循这个接口。
# 这样上层代码不依赖具体的记忆实现，可以轻松切换。


class BaseMemory(ABC):
    """
    记忆系统抽象基类

    定义了所有记忆实现必须提供的方法。
    遵循接口隔离原则：只定义必要的方法。
    """

    @abstractmethod
    def add(self, message: Message) -> None:
        """添加一条记忆"""
        pass

    @abstractmethod
    def get_all(self) -> list[Message]:
        """获取所有记忆"""
        pass

    @abstractmethod
    def get_recent(self, n: int = 10) -> list[Message]:
        """获取最近 n 条记忆"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空所有记忆"""
        pass

    @abstractmethod
    def size(self) -> int:
        """获取记忆数量"""
        pass


# ============================================================================
# 第四部分：短期记忆 - 对话上下文窗口（short_term.py）
# ============================================================================
#
# 短期记忆保存当前对话的最近消息。
# 由于 LLM 的上下文窗口有限，我们需要管理窗口大小。
#
# 策略：
# 1. 固定窗口：只保留最近 N 条消息
# 2. 滑动窗口：保留最近 N 条，但始终包含系统消息
# 3. Token 窗口：根据 Token 数量限制（更精确）


class ShortTermMemory(BaseMemory):
    """
    短期记忆实现

    使用滑动窗口策略管理对话上下文：
    - 始终保留系统消息（第一条）
    - 保留最近 N 条对话消息
    - 超出窗口时自动丢弃最旧的消息
    """

    def __init__(
        self,
        window_size: int = 20,
        always_include_system: bool = True,
    ):
        """
        Args:
            window_size: 窗口大小（最大消息数）
            always_include_system: 是否始终包含系统消息
        """
        self._messages: list[Message] = []
        self.window_size = window_size
        self.always_include_system = always_include_system

    def add(self, message: Message) -> None:
        """添加消息，自动管理窗口大小"""
        self._messages.append(message)
        self._trim_to_window()

    def get_all(self) -> list[Message]:
        """获取窗口内的所有消息"""
        return list(self._messages)

    def get_recent(self, n: int = 10) -> list[Message]:
        """获取最近 n 条消息"""
        return self._messages[-n:]

    def clear(self) -> None:
        """清空记忆"""
        self._messages.clear()

    def size(self) -> int:
        """获取消息数量"""
        return len(self._messages)

    def _trim_to_window(self):
        """
        裁剪到窗口大小

        策略：
        1. 如果设置了 always_include_system，保留第一条系统消息
        2. 保留最近 (window_size - 1) 条非系统消息
        3. 总消息数不超过 window_size
        """
        if len(self._messages) <= self.window_size:
            return

        if self.always_include_system and self._messages:
            # 保留系统消息 + 最近的消息
            system_msg = self._messages[0]
            recent_msgs = self._messages[-(self.window_size - 1):]
            self._messages = [system_msg] + recent_msgs
        else:
            # 直接保留最近的消息
            self._messages = self._messages[-self.window_size:]

    def to_llm_messages(self) -> list[dict]:
        """
        转换为 LLM API 需要的消息格式

        这是短期记忆最常用的方法：将记忆转换为可以发送给 LLM 的格式。
        """
        return [msg.to_dict() if hasattr(msg, 'to_dict')
                else {"role": msg.role.value, "content": msg.content}
                for msg in self._messages]

    def get_summary(self) -> dict:
        """获取记忆摘要"""
        return {
            "type": "short_term",
            "size": self.size(),
            "window_size": self.window_size,
            "roles": {
                role.value: sum(1 for m in self._messages if m.role == role)
                for role in Role
            },
        }


# ============================================================================
# 第五部分：长期记忆 - 跨会话持久化（long_term.py）
# ============================================================================
#
# 长期记忆保存跨会话的重要信息。
# 实现方式：
# 1. 文件存储（JSON）- 简单，适合学习
# 2. 数据库存储 - 适合生产
# 3. 键值存储（Redis）- 适合分布式


class MemoryEntry(BaseModel):
    """记忆条目"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    key: str = Field(description="记忆键（如 'user_name', 'preference'）")
    value: str = Field(description="记忆值")
    category: str = Field(default="general", description="记忆分类")
    importance: float = Field(default=0.5, description="重要性评分 (0-1)")
    created_at: datetime = Field(default_factory=datetime.now)
    access_count: int = Field(default=0, description="被访问次数")
    last_accessed: Optional[datetime] = Field(default=None)


class LongTermMemory(BaseMemory):
    """
    长期记忆实现

    使用 JSON 文件持久化存储重要信息。
    支持按类别、重要性检索。
    """

    def __init__(self, storage_file: str = "long_term_memory.json"):
        self._entries: dict[str, MemoryEntry] = {}
        self.storage_file = storage_file
        self._load_from_file()

    def add(self, message: Message) -> None:
        """
        从消息中提取并存储记忆

        实际应用中，这里可以用 LLM 从对话中提取关键信息。
        这里简化为直接存储。
        """
        entry = MemoryEntry(
            key=f"msg_{message.id}",
            value=message.content,
            category=message.role.value,
        )
        self._entries[entry.id] = entry
        self._save_to_file()

    def add_entry(self, entry: MemoryEntry) -> None:
        """直接添加记忆条目"""
        self._entries[entry.id] = entry
        self._save_to_file()

    def get_all(self) -> list[Message]:
        """获取所有记忆（转换为 Message 格式）"""
        return [
            Message(
                role=Role.USER,
                content=f"[{e.key}] {e.value}",
            )
            for e in self._entries.values()
        ]

    def get_recent(self, n: int = 10) -> list[Message]:
        """获取最近添加的 n 条记忆"""
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.created_at,
            reverse=True,
        )
        return [
            Message(role=Role.USER, content=f"[{e.key}] {e.value}")
            for e in sorted_entries[:n]
        ]

    def get_by_key(self, key: str) -> Optional[MemoryEntry]:
        """按键查找记忆"""
        for entry in self._entries.values():
            if entry.key == key:
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                return entry
        return None

    def get_by_category(self, category: str) -> list[MemoryEntry]:
        """按类别查找记忆"""
        return [
            e for e in self._entries.values()
            if e.category == category
        ]

    def get_important(self, threshold: float = 0.7) -> list[MemoryEntry]:
        """获取高重要性记忆"""
        return [
            e for e in self._entries.values()
            if e.importance >= threshold
        ]

    def search(self, query: str) -> list[MemoryEntry]:
        """
        简单文本搜索（关键词匹配）

        实际生产中应该用向量检索（见向量记忆部分）。
        """
        query_lower = query.lower()
        results = []
        for entry in self._entries.values():
            if query_lower in entry.key.lower() or query_lower in entry.value.lower():
                results.append(entry)
        return results

    def clear(self) -> None:
        """清空所有记忆"""
        self._entries.clear()
        self._save_to_file()

    def size(self) -> int:
        """获取记忆数量"""
        return len(self._entries)

    def forget(self, entry_id: str) -> bool:
        """
        遗忘指定记忆

        模拟人类的遗忘机制：可以主动删除不重要的记忆。
        """
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._save_to_file()
            return True
        return False

    def _save_to_file(self):
        """持久化到文件"""
        try:
            data = {
                eid: entry.model_dump()
                for eid, entry in self._entries.items()
            }
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"  ⚠️ 保存记忆失败: {e}")

    def _load_from_file(self):
        """从文件加载"""
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for eid, entry_data in data.items():
                self._entries[eid] = MemoryEntry(**entry_data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # 文件不存在或格式错误，使用空记忆


# ============================================================================
# 第六部分：向量记忆 - 语义检索（vector_memory.py）
# ============================================================================
#
# 向量记忆使用 Embedding 将文本转换为向量，
# 然后通过向量相似度检索相关记忆。
#
# 这是最强大的记忆方式，可以：
# 1. 找到语义相关但不完全匹配的记忆
# 2. 处理大量记忆时保持高效
# 3. 支持模糊查询


class VectorMemoryEntry(BaseModel):
    """向量记忆条目"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str
    embedding: Optional[list[float]] = Field(default=None)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class SimpleVectorMemory(BaseMemory):
    """
    简化的向量记忆实现

    使用简单的词频向量（TF）代替真正的 Embedding，
    便于理解向量检索的原理。

    生产环境中应该使用：
    - OpenAI Embedding / Claude Embedding
    - FAISS / ChromaDB 向量数据库
    """

    def __init__(self, top_k: int = 5):
        """
        Args:
            top_k: 检索时返回的最相似条目数
        """
        self._entries: list[VectorMemoryEntry] = []
        self.top_k = top_k
        self._vocabulary: dict[str, int] = {}  # 词表

    def add(self, message: Message) -> None:
        """添加记忆并计算向量"""
        entry = VectorMemoryEntry(content=message.content)
        entry.embedding = self._text_to_vector(message.content)
        self._entries.append(entry)

    def get_all(self) -> list[Message]:
        return [
            Message(role=Role.USER, content=e.content)
            for e in self._entries
        ]

    def get_recent(self, n: int = 10) -> list[Message]:
        return [
            Message(role=Role.USER, content=e.content)
            for e in self._entries[-n:]
        ]

    def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        语义检索：找到与查询最相似的记忆

        步骤：
        1. 将查询文本转换为向量
        2. 计算与每个记忆条目的余弦相似度
        3. 返回最相似的 top_k 个结果
        """
        k = top_k or self.top_k
        query_vector = self._text_to_vector(query)

        results = []
        for entry in self._entries:
            if entry.embedding:
                similarity = self._cosine_similarity(
                    query_vector, entry.embedding
                )
                results.append({
                    "entry": entry,
                    "similarity": similarity,
                })

        # 按相似度排序
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]

    def clear(self) -> None:
        self._entries.clear()
        self._vocabulary.clear()

    def size(self) -> int:
        return len(self._entries)

    # ── 向量计算工具方法 ─────────────────────────────────────────

    def _text_to_vector(self, text: str) -> list[float]:
        """
        简化的文本向量化（词频向量）

        将文本转换为词频向量，用于演示向量检索原理。
        生产环境应使用真正的 Embedding 模型。
        """
        # 简单的中文分词（按字符）
        words = list(text.lower())

        # 更新词表
        for word in words:
            if word not in self._vocabulary:
                self._vocabulary[word] = len(self._vocabulary)

        # 创建词频向量
        vector = [0.0] * len(self._vocabulary)
        for word in words:
            vector[self._vocabulary[word]] += 1.0

        # 归一化
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        计算余弦相似度

        余弦相似度衡量两个向量的方向相似程度：
        - 1.0: 完全相同
        - 0.0: 完全无关
        - -1.0: 完全相反
        """
        if len(a) != len(b):
            # 补齐较短的向量
            max_len = max(len(a), len(b))
            a = a + [0.0] * (max_len - len(a))
            b = b + [0.0] * (max_len - len(b))

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


# ============================================================================
# 第七部分：记忆压缩与摘要策略
# ============================================================================

class MemoryCompressor:
    """
    记忆压缩器

    当记忆过多时，需要压缩以节省上下文空间。
    压缩策略：
    1. 摘要压缩：将多条消息压缩为一条摘要
    2. 重要性过滤：只保留重要记忆
    3. 时间衰减：旧记忆权重降低
    """

    @staticmethod
    def summarize_messages(
        messages: list[Message],
        max_messages: int = 5,
    ) -> list[Message]:
        """
        摘要压缩：将多条消息压缩为摘要

        实际应用中，这里会调用 LLM 生成摘要。
        这里用简化方式演示。
        """
        if len(messages) <= max_messages:
            return messages

        # 保留最新的消息
        recent = messages[-max_messages:]

        # 创建摘要消息（模拟 LLM 生成的摘要）
        summary_content = (
            f"[对话摘要：共 {len(messages)} 条消息，"
            f"已压缩为最近 {max_messages} 条]"
        )
        summary_msg = Message.system(summary_content)

        return [summary_msg] + recent

    @staticmethod
    def filter_by_importance(
        entries: list[MemoryEntry],
        threshold: float = 0.5,
    ) -> list[MemoryEntry]:
        """重要性过滤"""
        return [e for e in entries if e.importance >= threshold]

    @staticmethod
    def apply_time_decay(
        entries: list[MemoryEntry],
        half_life_days: float = 7.0,
    ) -> list[tuple[MemoryEntry, float]]:
        """
        时间衰减：计算每条记忆的当前权重

        权重随时间指数衰减：
        weight = importance * (0.5 ^ (days_since_created / half_life))
        """
        now = datetime.now()
        results = []
        for entry in entries:
            days_old = (now - entry.created_at).total_seconds() / 86400
            decay = 0.5 ** (days_old / half_life_days)
            current_weight = entry.importance * decay
            results.append((entry, current_weight))

        # 按权重排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# ============================================================================
# 第八部分：记忆系统综合演示
# ============================================================================

def demonstrate_short_term_memory():
    """演示短期记忆"""
    print("=" * 60)
    print("短期记忆演示")
    print("=" * 60)

    memory = ShortTermMemory(window_size=5)

    # 添加系统消息
    memory.add(Message.system("你是一个有帮助的AI助手。"))

    # 添加对话
    conversations = [
        ("你好", "你好！有什么可以帮助你的？"),
        ("我叫张三", "很高兴认识你，张三！"),
        ("今天天气不错", "是的，今天天气很好。"),
        ("我喜欢编程", "编程是一项很棒的技能！"),
        ("我最近在学习Python", "Python 是很流行的编程语言。"),
        ("你觉得AI未来会怎样", "AI 的发展前景非常广阔。"),
    ]

    for user_msg, ai_msg in conversations:
        memory.add(Message.user(user_msg))
        memory.add(Message.assistant(ai_msg))

    # 查看记忆状态
    print(f"  记忆摘要: {memory.get_summary()}")
    print(f"\n  窗口内消息 ({memory.size()} 条):")
    for i, msg in enumerate(memory.get_all()):
        print(f"    [{i}] {msg.role.value}: {msg.content[:30]}")

    print(f"\n  最近 3 条:")
    for msg in memory.get_recent(3):
        print(f"    {msg.role.value}: {msg.content[:30]}")
    print()


def demonstrate_long_term_memory():
    """演示长期记忆"""
    print("=" * 60)
    print("长期记忆演示")
    print("=" * 60)

    memory = LongTermMemory(storage_file="/tmp/demo_long_term_memory.json")
    memory.clear()  # 清空之前的数据

    # 添加记忆条目
    entries = [
        MemoryEntry(key="user_name", value="张三", category="user_info",
                    importance=0.9),
        MemoryEntry(key="user_preference", value="喜欢简洁的回答",
                    category="preference", importance=0.7),
        MemoryEntry(key="user_goal", value="学习 Agent 开发",
                    category="goal", importance=0.8),
        MemoryEntry(key="temp_info", value="今天日期是周一",
                    category="temp", importance=0.2),
    ]

    for entry in entries:
        memory.add_entry(entry)
        print(f"  ✅ 添加记忆: [{entry.key}] = {entry.value}")

    # 检索记忆
    print(f"\n  按键查找 'user_name':")
    result = memory.get_by_key("user_name")
    if result:
        print(f"    找到: {result.key} = {result.value}")

    print(f"\n  按类别查找 'preference':")
    results = memory.get_by_category("preference")
    for r in results:
        print(f"    {r.key} = {r.value}")

    print(f"\n  高重要性记忆 (>=0.7):")
    important = memory.get_important(0.7)
    for e in important:
        print(f"    [{e.key}] {e.value} (重要性: {e.importance})")

    print(f"\n  搜索 '学习':")
    results = memory.search("学习")
    for r in results:
        print(f"    [{r.key}] {r.value}")

    # 清理临时文件
    import os
    try:
        os.remove("/tmp/demo_long_term_memory.json")
    except Exception:
        pass

    print()


def demonstrate_vector_memory():
    """演示向量记忆"""
    print("=" * 60)
    print("向量记忆演示")
    print("=" * 60)

    memory = SimpleVectorMemory(top_k=3)

    # 添加记忆
    memories = [
        "Python 是一种流行的编程语言",
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络",
        "自然语言处理让计算机理解文本",
        "向量数据库用于存储和检索高维向量",
        "余弦相似度衡量向量之间的方向相似性",
    ]

    for mem in memories:
        memory.add(Message.user(mem))
        print(f"  ✅ 添加: {mem}")

    # 语义检索
    queries = [
        "编程语言",
        "人工智能技术",
        "数据库检索",
    ]

    for query in queries:
        print(f"\n  查询: 「{query}」")
        results = memory.search(query)
        for i, result in enumerate(results):
            entry = result["entry"]
            sim = result["similarity"]
            print(f"    [{i+1}] 相似度: {sim:.3f} - {entry.content}")
    print()


def demonstrate_memory_compression():
    """演示记忆压缩"""
    print("=" * 60)
    print("记忆压缩演示")
    print("=" * 60)

    # 创建大量消息
    messages = []
    for i in range(10):
        messages.append(Message.user(f"用户消息 {i+1}"))
        messages.append(Message.assistant(f"AI 回复 {i+1}"))

    print(f"  原始消息数: {len(messages)}")

    # 摘要压缩
    compressed = MemoryCompressor.summarize_messages(messages, max_messages=4)
    print(f"  压缩后消息数: {len(compressed)}")
    for msg in compressed:
        print(f"    {msg.role.value}: {msg.content[:40]}")

    # 时间衰减
    print(f"\n  时间衰减演示:")
    entries = [
        MemoryEntry(key="old", value="旧记忆", importance=0.8,
                    created_at=datetime.now() - timedelta(days=14)),
        MemoryEntry(key="medium", value="中等记忆", importance=0.6,
                    created_at=datetime.now() - timedelta(days=7)),
        MemoryEntry(key="new", value="新记忆", importance=0.5,
                    created_at=datetime.now() - timedelta(days=1)),
    ]

    results = MemoryCompressor.apply_time_decay(entries)
    for entry, weight in results:
        print(f"    [{entry.key}] 原始重要性: {entry.importance}, "
              f"当前权重: {weight:.3f}")
    print()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🧠" * 30)
    print("第五课：Memory 模块 - 让 Agent 记住上下文")
    print("" * 30 + "\n")

    # 1. 为什么需要记忆
    why_memory_matters()

    # 2. 记忆架构
    show_memory_architecture()

    # 3. 短期记忆演示
    demonstrate_short_term_memory()

    # 4. 长期记忆演示
    demonstrate_long_term_memory()

    # 5. 向量记忆演示
    demonstrate_vector_memory()

    # 6. 记忆压缩演示
    demonstrate_memory_compression()

    print("=" * 60)
    print("✅ 第五课学习完成！")
    print("=" * 60)
    print("""
 本课要点总结：
1. 短期记忆：滑动窗口管理对话上下文，始终保留系统消息
2. 长期记忆：持久化存储重要信息，支持分类和搜索
3. 向量记忆：语义检索，找到相关但不完全匹配的记忆
4. 记忆压缩：摘要、重要性过滤、时间衰减
5. 记忆系统通过 BaseMemory 接口统一，可灵活切换实现

 下一课：Session 模块 - 我们将实现多用户会话管理
    """)
