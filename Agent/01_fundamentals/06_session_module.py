"""
==============================================================================
第六课：Session 模块 - 多用户会话管理
==============================================================================

【学习目标】
- 理解会话管理的核心概念
- 掌握会话生命周期管理
- 实现多用户会话隔离
- 掌握会话持久化策略
- 学会会话超时和清理机制

【核心概念】
- 会话（Session）：一个用户与 Agent 的完整交互过程
- 会话隔离：不同用户的会话互不干扰
- 会话生命周期：创建 → 活跃 → 超时 → 销毁
- 会话存储：内存存储 vs 持久化存储

【前置知识】
- 第五课：Memory 模块（短期记忆、长期记忆）

==============================================================================
"""

import uuid
import time
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field

# 复用之前定义的 Message
from enum import Enum as PyEnum


class Role(PyEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: Role
    content: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role=Role.ASSISTANT, content=content)


# ============================================================================
# 第一部分：为什么需要 Session 模块？
# ============================================================================

def why_session_matters():
    """
    解释为什么需要会话管理

    当多个用户同时与 Agent 交互时：
    1. 每个用户需要独立的对话上下文（不能串话）
    2. 需要管理会话的生命周期（创建、超时、销毁）
    3. 需要跟踪会话状态（活跃、空闲、已结束）
    4. 需要支持会话恢复（用户断开后重新连接）
    """

    scenarios = [
        {
            "scenario": "多用户聊天机器人",
            "problem": "用户A和用户B同时聊天，不能把A的对话发给B",
            "solution": "每个用户一个独立的 Session",
        },
        {
            "scenario": "客服系统",
            "problem": "客服需要知道用户之前的咨询记录",
            "solution": "Session 关联历史对话和记忆",
        },
        {
            "scenario": "长时间任务",
            "problem": "用户中途离开，回来后需要恢复上下文",
            "solution": "Session 持久化，支持恢复",
        },
        {
            "scenario": "资源管理",
            "problem": "大量空闲会话占用内存",
            "solution": "会话超时自动清理",
        },
    ]

    print("=" * 60)
    print("为什么需要 Session 模块")
    print("=" * 60)
    for s in scenarios:
        print(f"\n  场景: {s['scenario']}")
        print(f"    问题: {s['problem']}")
        print(f"    方案: {s['solution']}")
    print()


# ============================================================================
# 第二部分：会话数据模型
# ============================================================================

class SessionStatus(Enum):
    """会话状态"""
    CREATED = "created"       # 刚创建
    ACTIVE = "active"         # 活跃中
    IDLE = "idle"             # 空闲（超时但未销毁）
    EXPIRED = "expired"       # 已过期
    CLOSED = "closed"         # 已关闭


class SessionConfig(BaseModel):
    """
    会话配置

    定义会话的行为参数。
    """
    max_messages: int = Field(default=100, description="最大消息数")
    idle_timeout_seconds: int = Field(
        default=1800, description="空闲超时时间（秒），默认30分钟"
    )
    max_lifetime_seconds: int = Field(
        default=86400, description="最大生命周期（秒），默认24小时"
    )
    auto_cleanup: bool = Field(
        default=True, description="是否自动清理过期会话"
    )


class Session(BaseModel):
    """
    会话模型

    一个 Session 代表一个用户与 Agent 的完整交互过程。
    包含：
    - 会话 ID（唯一标识）
    - 用户信息
    - 对话历史
    - 会话状态
    - 时间信息
    """

    # 基本信息
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="会话唯一 ID",
    )
    user_id: str = Field(description="用户 ID")

    # 对话内容
    messages: list[Message] = Field(
        default_factory=list,
        description="对话历史",
    )

    # 状态
    status: SessionStatus = Field(
        default=SessionStatus.CREATED,
        description="会话状态",
    )

    # 配置
    config: SessionConfig = Field(
        default_factory=SessionConfig,
        description="会话配置",
    )

    # 时间
    created_at: datetime = Field(default_factory=datetime.now)
    last_active_at: datetime = Field(default_factory=datetime.now)
    closed_at: Optional[datetime] = Field(default=None)

    # 元数据
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ── 会话操作方法 ─────────────────────────────────────────────

    def add_message(self, message: Message) -> None:
        """添加消息并更新活跃时间"""
        self.messages.append(message)
        self.last_active_at = datetime.now()
        self.status = SessionStatus.ACTIVE

    def get_recent_messages(self, n: int = 10) -> list[Message]:
        """获取最近 n 条消息"""
        return self.messages[-n:]

    def is_expired(self) -> bool:
        """检查是否过期"""
        now = datetime.now()

        # 检查空闲超时
        idle_time = (now - self.last_active_at).total_seconds()
        if idle_time > self.config.idle_timeout_seconds:
            return True

        # 检查最大生命周期
        lifetime = (now - self.created_at).total_seconds()
        if lifetime > self.config.max_lifetime_seconds:
            return True

        return False

    def close(self) -> None:
        """关闭会话"""
        self.status = SessionStatus.CLOSED
        self.closed_at = datetime.now()

    def get_summary(self) -> dict:
        """获取会话摘要"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "messages": len(self.messages),
            "created": self.created_at.isoformat(),
            "last_active": self.last_active_at.isoformat(),
            "idle_seconds": (
                datetime.now() - self.last_active_at
            ).total_seconds(),
        }


# ============================================================================
# 第三部分：会话存储（store.py）
# ============================================================================
#
# 会话存储负责保存和检索会话数据。
# 支持两种实现：
# 1. 内存存储（快速，但进程重启后丢失）
# 2. Redis 存储（持久化，支持分布式）


class BaseSessionStore(ABC):
    """会话存储抽象基类"""

    @abstractmethod
    def save(self, session: Session) -> None:
        """保存会话"""
        pass

    @abstractmethod
    def get(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        pass

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """删除会话"""
        pass

    @abstractmethod
    def get_all(self) -> list[Session]:
        """获取所有会话"""
        pass

    @abstractmethod
    def get_by_user(self, user_id: str) -> list[Session]:
        """获取用户的所有会话"""
        pass


class InMemorySessionStore(BaseSessionStore):
    """
    内存会话存储

    使用 Python 字典存储会话，速度快但进程重启后丢失。
    适合开发和测试环境。
    """

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def save(self, session: Session) -> None:
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def get_all(self) -> list[Session]:
        return list(self._sessions.values())

    def get_by_user(self, user_id: str) -> list[Session]:
        return [
            s for s in self._sessions.values()
            if s.user_id == user_id
        ]

    def size(self) -> int:
        return len(self._sessions)


# ============================================================================
# 第四部分：会话管理器（manager.py）
# ============================================================================
#
# 会话管理器是 Session 模块的核心。
# 它负责：
# 1. 创建新会话
# 2. 查找现有会话
# 3. 管理会话生命周期
# 4. 清理过期会话


class SessionManager:
    """
    会话管理器

    提供统一的会话管理接口，封装存储和生命周期管理逻辑。
    """

    def __init__(
        self,
        store: Optional[BaseSessionStore] = None,
        default_config: Optional[SessionConfig] = None,
    ):
        """
        Args:
            store: 会话存储实现（默认使用内存存储）
            default_config: 默认会话配置
        """
        self.store = store or InMemorySessionStore()
        self.default_config = default_config or SessionConfig()

    def create_session(
        self,
        user_id: str,
        config: Optional[SessionConfig] = None,
        metadata: Optional[dict] = None,
    ) -> Session:
        """
        创建新会话

        Args:
            user_id: 用户 ID
            config: 会话配置（可选，使用默认配置）
            metadata: 元数据（可选）

        Returns:
            新创建的 Session 对象
        """
        session = Session(
            user_id=user_id,
            config=config or self.default_config,
            metadata=metadata or {},
        )
        self.store.save(session)
        print(f"  ✅ 创建会话: {session.session_id[:8]}... (用户: {user_id})")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        获取会话

        如果会话已过期，自动标记为过期状态。
        """
        session = self.store.get(session_id)
        if session and session.is_expired():
            session.status = SessionStatus.EXPIRED
            self.store.save(session)
        return session

    def get_or_create_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> Session:
        """
        获取或创建会话

        如果提供了 session_id 且会话存在且有效，返回该会话。
        否则创建新会话。
        """
        if session_id:
            session = self.get_session(session_id)
            if session and session.status != SessionStatus.EXPIRED:
                return session

        # 创建新会话
        return self.create_session(user_id)

    def close_session(self, session_id: str) -> bool:
        """关闭会话"""
        session = self.store.get(session_id)
        if session:
            session.close()
            self.store.save(session)
            print(f"  ✅ 关闭会话: {session_id[:8]}...")
            return True
        return False

    def cleanup_expired(self) -> int:
        """
        清理所有过期会话

        返回清理的会话数量。
        """
        expired_count = 0
        for session in self.store.get_all():
            if session.is_expired() or session.status == SessionStatus.CLOSED:
                self.store.delete(session.session_id)
                expired_count += 1

        if expired_count > 0:
            print(f"  清理了 {expired_count} 个过期/已关闭会话")
        return expired_count

    def get_active_sessions(self) -> list[Session]:
        """获取所有活跃会话"""
        return [
            s for s in self.store.get_all()
            if s.status == SessionStatus.ACTIVE
        ]

    def get_stats(self) -> dict:
        """获取会话统计信息"""
        all_sessions = self.store.get_all()
        status_counts = {}
        for s in all_sessions:
            status = s.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total": len(all_sessions),
            "by_status": status_counts,
            "total_messages": sum(len(s.messages) for s in all_sessions),
        }


# ============================================================================
# 第五部分：会话管理演示
# ============================================================================

def demonstrate_session_lifecycle():
    """演示会话生命周期"""
    print("=" * 60)
    print("会话生命周期演示")
    print("=" * 60)

    manager = SessionManager()

    # 1. 创建会话
    print("\n── 1. 创建会话 ──")
    session = manager.create_session(
        user_id="user_001",
        metadata={"platform": "web", "language": "zh"},
    )
    print(f"  会话 ID: {session.session_id}")
    print(f"  状态: {session.status.value}")

    # 2. 添加消息
    print("\n── 2. 对话交互 ──")
    conversations = [
        ("你好", "你好！有什么可以帮助你的？"),
        ("我想学习 Agent 开发", "Agent 开发是一个很有趣的领域！"),
        ("从哪里开始？", "建议从基础的 LLM 调用开始学习。"),
    ]

    for user_msg, ai_msg in conversations:
        session.add_message(Message.user(user_msg))
        session.add_message(Message.assistant(ai_msg))
        print(f"  用户: {user_msg}")
        print(f"  AI:   {ai_msg}")

    # 3. 查看会话状态
    print(f"\n── 3. 会话状态 ──")
    summary = session.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # 4. 获取最近消息
    print(f"\n── 4. 最近消息 ──")
    recent = session.get_recent_messages(2)
    for msg in recent:
        print(f"  {msg.role.value}: {msg.content}")

    # 5. 关闭会话
    print(f"\n── 5. 关闭会话 ──")
    manager.close_session(session.session_id)
    print(f"  状态: {session.status.value}")
    print()


def demonstrate_multi_user():
    """演示多用户会话隔离"""
    print("=" * 60)
    print("多用户会话隔离演示")
    print("=" * 60)

    manager = SessionManager()

    # 创建多个用户的会话
    users = ["alice", "bob", "charlie"]
    sessions = {}

    print("\n── 创建用户会话 ──")
    for user in users:
        session = manager.create_session(user_id=user)
        sessions[user] = session

        # 每个用户有不同的对话
        session.add_message(Message.user(f"我是{user}"))
        session.add_message(Message.assistant(f"你好，{user}！"))

    # 验证会话隔离
    print("\n── 验证会话隔离 ──")
    for user, session in sessions.items():
        messages = session.get_recent_messages(1)
        print(f"  {user} 的最近消息: {messages[0].content}")

    # 统计信息
    print(f"\n── 会话统计 ──")
    stats = manager.get_stats()
    print(f"  总会话数: {stats['total']}")
    print(f"  按状态: {stats['by_status']}")
    print(f"  总消息数: {stats['total_messages']}")

    # 按用户查询
    print(f"\n── 按用户查询 ──")
    alice_sessions = manager.store.get_by_user("alice")
    print(f"  Alice 的会话数: {len(alice_sessions)}")
    print()


def demonstrate_session_timeout():
    """演示会话超时机制"""
    print("=" * 60)
    print("会话超时机制演示")
    print("=" * 60)

    # 创建短超时的会话（用于演示）
    config = SessionConfig(
        idle_timeout_seconds=2,  # 2秒超时（演示用）
        max_lifetime_seconds=10,
    )
    manager = SessionManager(default_config=config)

    session = manager.create_session(user_id="timeout_user")
    session.add_message(Message.user("测试消息"))

    print(f"  初始状态: {session.status.value}")
    print(f"  超时设置: {config.idle_timeout_seconds} 秒")

    # 模拟时间流逝
    print(f"\n  等待 {config.idle_timeout_seconds + 1} 秒...")
    time.sleep(config.idle_timeout_seconds + 1)

    # 检查是否过期
    is_expired = session.is_expired()
    print(f"  是否过期: {is_expired}")

    if is_expired:
        session.status = SessionStatus.EXPIRED
        print(f"  状态已更新为: {session.status.value}")

    # 清理
    cleaned = manager.cleanup_expired()
    print(f"  清理会话数: {cleaned}")
    print()


def demonstrate_session_recovery():
    """演示会话恢复"""
    print("=" * 60)
    print("会话恢复演示")
    print("=" * 60)

    manager = SessionManager()

    # 用户创建会话并对话
    print("\n── 第一轮对话 ──")
    session = manager.create_session(user_id="returning_user")
    session.add_message(Message.user("我叫张三"))
    session.add_message(Message.assistant("你好张三！"))
    session.add_message(Message.user("我喜欢编程"))
    session.add_message(Message.assistant("编程很棒！"))

    session_id = session.session_id
    print(f"  会话 ID: {session_id[:8]}...")
    print(f"  消息数: {len(session.messages)}")

    # 模拟用户离开后回来（使用同一个 session_id）
    print(f"\n── 用户回来（恢复会话）──")
    recovered = manager.get_or_create_session(
        user_id="returning_user",
        session_id=session_id,
    )

    if recovered:
        print(f"  ✅ 会话恢复成功！")
        print(f"  历史消息数: {len(recovered.messages)}")

        # 继续对话
        recovered.add_message(Message.user("我之前说了什么？"))
        recovered.add_message(
            Message.assistant("你说过你叫张三，喜欢编程。")
        )
        print(f"  新消息数: {len(recovered.messages)}")
    else:
        print(f"  ❌ 会话恢复失败，创建了新会话")
    print()


# ============================================================================
# 第六部分：进阶 - 会话事件与观察者模式
# ============================================================================

class SessionEvent(Enum):
    """会话事件类型"""
    CREATED = "session_created"
    MESSAGE_ADDED = "message_added"
    EXPIRED = "session_expired"
    CLOSED = "session_closed"


class SessionObserver(ABC):
    """会话观察者抽象类"""

    @abstractmethod
    def on_event(self, event: SessionEvent, session: Session) -> None:
        """处理会话事件"""
        pass


class LoggingObserver(SessionObserver):
    """日志观察者：记录所有会话事件"""

    def on_event(self, event: SessionEvent, session: Session) -> None:
        print(f"  [事件] {event.value} | 会话: {session.session_id[:8]}... | "
              f"用户: {session.user_id}")


class MetricsObserver(SessionObserver):
    """指标观察者：统计会话数据"""

    def __init__(self):
        self.events_count: dict[str, int] = {}

    def on_event(self, event: SessionEvent, session: Session) -> None:
        event_name = event.value
        self.events_count[event_name] = (
            self.events_count.get(event_name, 0) + 1
        )

    def get_metrics(self) -> dict:
        return dict(self.events_count)


class ObservableSessionManager(SessionManager):
    """
    支持观察者模式的会话管理器

    当会话状态变化时，自动通知所有注册的观察者。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._observers: list[SessionObserver] = []

    def add_observer(self, observer: SessionObserver) -> None:
        """注册观察者"""
        self._observers.append(observer)

    def _notify(self, event: SessionEvent, session: Session) -> None:
        """通知所有观察者"""
        for observer in self._observers:
            observer.on_event(event, session)

    def create_session(self, user_id: str, **kwargs) -> Session:
        session = super().create_session(user_id, **kwargs)
        self._notify(SessionEvent.CREATED, session)
        return session

    def close_session(self, session_id: str) -> bool:
        session = self.store.get(session_id)
        result = super().close_session(session_id)
        if result and session:
            self._notify(SessionEvent.CLOSED, session)
        return result


def demonstrate_observer_pattern():
    """演示观察者模式在会话管理中的应用"""
    print("=" * 60)
    print("进阶：会话事件观察者模式")
    print("=" * 60)

    # 创建带观察者的管理器
    manager = ObservableSessionManager()

    # 注册观察者
    logging_observer = LoggingObserver()
    metrics_observer = MetricsObserver()
    manager.add_observer(logging_observer)
    manager.add_observer(metrics_observer)

    print("\n── 操作会话 ──")
    session = manager.create_session(user_id="observer_user")
    session.add_message(Message.user("测试消息"))
    manager.close_session(session.session_id)

    print(f"\n── 事件统计 ──")
    metrics = metrics_observer.get_metrics()
    for event, count in metrics.items():
        print(f"  {event}: {count} 次")
    print()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "💬" * 30)
    print("第六课：Session 模块 - 多用户会话管理")
    print("💬" * 30 + "\n")

    # 1. 为什么需要 Session
    why_session_matters()

    # 2. 会话生命周期
    demonstrate_session_lifecycle()

    # 3. 多用户隔离
    demonstrate_multi_user()

    # 4. 会话超时
    demonstrate_session_timeout()

    # 5. 会话恢复
    demonstrate_session_recovery()

    # 6. 观察者模式（进阶）
    demonstrate_observer_pattern()

    print("=" * 60)
    print("✅ 第六课学习完成！")
    print("=" * 60)
    print("""
 本课要点总结：
1. Session 代表一个用户与 Agent 的完整交互过程
2. 会话隔离确保不同用户的对话互不干扰
3. 会话生命周期：创建 → 活跃 → 超时 → 关闭
4. 会话存储支持内存和持久化两种实现
5. 会话管理器提供统一的创建/查找/清理接口
6. 进阶：观察者模式实现会话事件监听

 下一课：Agents 模块 - 我们将实现核心业务逻辑
    """)
