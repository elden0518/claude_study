"""
==============================================================================
第二十六课（进阶）：Event-Driven Architecture（事件驱动架构）
==============================================================================

【为什么需要单独一课？】
现有课程主要讲解了同步交互模式，没有涉及事件驱动的异步架构。
生产级 Agent 系统通常需要事件驱动来处理异步任务、解耦组件。

【学习目标】
- 理解事件驱动架构（EDA）的原理
- 掌握 Pub/Sub 发布订阅模式
- 掌握事件总线（Event Bus）实现
- 学会异步事件处理和消息队列
- 理解事件溯源（Event Sourcing）

【核心概念】
- Event（事件）
- Publisher/Subscriber（发布/订阅）
- Event Bus（事件总线）
- Message Queue（消息队列）
- Event Sourcing（事件溯源）
- CQRS（命令查询职责分离）

==============================================================================
"""

import asyncio
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


# ============================================================================
# 第一部分：事件驱动架构概览
# ============================================================================

def explain_eda_overview():
    """
    事件驱动架构（EDA）概览

    EDA 是一种软件架构模式，系统的组件之间通过
    产生和消费事件来通信，而非直接调用。
    """

    print("=" * 60)
    print("Event-Driven Architecture (EDA) 概览")
    print("=" * 60)

    print("""
  -- 什么是事件驱动架构？--

  传统同步模式:
    组件A --> 调用 --> 组件B --> 返回 --> 组件A
    (紧密耦合，阻塞等待)

  事件驱动模式:
    组件A --> 发布事件 --> [事件总线] --> 组件B 消费事件
                                      --> 组件C 消费事件
    (松耦合，异步非阻塞)


  -- EDA 的核心优势 --

  1. 松耦合: 发布者不知道订阅者是谁
  2. 可扩展: 新增消费者不影响生产者
  3. 异步: 不阻塞主流程
  4. 弹性: 单个组件故障不影响整体


  -- Agent 系统中的事件场景 --

  1. 用户消息事件: 用户发送消息 -> 多个处理器响应
  2. 工具调用事件: 工具执行完成 -> 触发后续处理
  3. 记忆事件: 新记忆存储 -> 触发索引更新
  4. 监控事件: 异常发生 -> 触发告警
  5. 定时事件: 会话超时 -> 触发清理
    """)


# ============================================================================
# 第二部分：事件模型
# ============================================================================

class EventPriority(Enum):
    """事件优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Event:
    """
    事件模型

    事件是系统中发生的有意义的事情。
    每个事件都有类型、数据和元信息。
    """
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    source: str = ""
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "priority": self.priority.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


# 预定义事件类型
class AgentEventType:
    """Agent 系统事件类型"""
    # 用户相关
    USER_MESSAGE = "user.message"
    USER_CONNECTED = "user.connected"
    USER_DISCONNECTED = "user.disconnected"

    # Agent 相关
    AGENT_START = "agent.start"
    AGENT_THINKING = "agent.thinking"
    AGENT_RESPONSE = "agent.response"
    AGENT_ERROR = "agent.error"

    # 工具相关
    TOOL_CALLED = "tool.called"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"

    # 记忆相关
    MEMORY_STORED = "memory.stored"
    MEMORY_RETRIEVED = "memory.retrieved"

    # 会话相关
    SESSION_CREATED = "session.created"
    SESSION_TIMEOUT = "session.timeout"
    SESSION_CLOSED = "session.closed"

    # 系统相关
    SYSTEM_HEALTH = "system.health"
    SYSTEM_ALERT = "system.alert"


# ============================================================================
# 第三部分：事件总线（Event Bus）
# ============================================================================

class EventBus:
    """
    事件总线

    【原理】
    事件总线是 EDA 的核心组件，负责：
    1. 接收发布者的事件
    2. 将事件路由到所有订阅了该类型的订阅者
    3. 支持同步和异步处理

    【模式】
    - 点对点: 一个事件只被一个消费者处理
    - 发布订阅: 一个事件被所有订阅者处理
    - 请求响应: 发送请求并等待响应
    """

    def __init__(self, name: str = "main"):
        self.name = name
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._history: List[Event] = []
        self._middleware: List[Callable] = []
        self._event_count = 0

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """
        订阅事件

        Args:
            event_type: 事件类型（支持通配符 *）
            handler: 处理函数
        """
        self._subscribers[event_type].append(handler)
        print(f"  [Subscribe] {handler.__name__} -> {event_type}")

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """取消订阅"""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    async def publish(self, event: Event) -> None:
        """
        发布事件

        将事件发送给所有匹配的订阅者
        """
        self._event_count += 1
        self._history.append(event)

        print(f"\n  [Publish] {event.event_type} (id: {event.event_id})")

        # 执行中间件
        for mw in self._middleware:
            event = mw(event)
            if event is None:
                print(f"  [Middleware] 事件被拦截")
                return

        # 找到匹配的订阅者
        handlers = []

        # 精确匹配
        handlers.extend(self._subscribers.get(event.event_type, []))

        # 通配符匹配 (如 "agent.*" 匹配 "agent.response")
        for pattern, subs in self._subscribers.items():
            if "*" in pattern:
                prefix = pattern.replace("*", "")
                if event.event_type.startswith(prefix):
                    handlers.extend(subs)

        # 执行所有处理器
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                print(f"  [Error] Handler {handler.__name__} failed: {e}")

    def add_middleware(self, middleware: Callable) -> None:
        """添加中间件（拦截/处理事件）"""
        self._middleware.append(middleware)

    def get_history(self, event_type: Optional[str] = None) -> List[Event]:
        """获取事件历史"""
        if event_type:
            return [e for e in self._history if e.event_type == event_type]
        return list(self._history)

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        type_counts = defaultdict(int)
        for event in self._history:
            type_counts[event.event_type] += 1
        return dict(type_counts)


# ============================================================================
# 第四部分：事件驱动的 Agent
# ============================================================================

class EventDrivenAgent:
    """
    事件驱动的 Agent

    【特点】
    - 所有操作都通过事件触发
    - 松耦合，各组件独立
    - 支持异步处理
    - 完整的事件追踪
    """

    def __init__(self, name: str, event_bus: EventBus):
        self.name = name
        self.event_bus = event_bus
        self._setup_handlers()

    def _setup_handlers(self):
        """注册事件处理器"""
        self.event_bus.subscribe(AgentEventType.USER_MESSAGE, self._on_user_message)
        self.event_bus.subscribe(AgentEventType.TOOL_COMPLETED, self._on_tool_completed)
        self.event_bus.subscribe(AgentEventType.TOOL_FAILED, self._on_tool_failed)

    async def _on_user_message(self, event: Event):
        """处理用户消息"""
        print(f"    [{self.name}] 收到用户消息: {event.data.get('content', '')[:30]}...")

        # 发布思考事件
        await self.event_bus.publish(Event(
            event_type=AgentEventType.AGENT_THINKING,
            data={"agent": self.name, "thought": "分析用户意图..."},
            source=self.name,
        ))

        # 模拟处理
        await asyncio.sleep(0.1)

        # 发布响应事件
        await self.event_bus.publish(Event(
            event_type=AgentEventType.AGENT_RESPONSE,
            data={"agent": self.name, "response": "这是 Agent 的回复"},
            source=self.name,
        ))

    async def _on_tool_completed(self, event: Event):
        """处理工具完成事件"""
        tool_name = event.data.get("tool_name", "unknown")
        print(f"    [{self.name}] 工具完成: {tool_name}")

    async def _on_tool_failed(self, event: Event):
        """处理工具失败事件"""
        tool_name = event.data.get("tool_name", "unknown")
        error = event.data.get("error", "unknown")
        print(f"    [{self.name}] 工具失败: {tool_name} - {error}")


# ============================================================================
# 第五部分：事件处理器（消费者）
# ============================================================================

class EventHandler:
    """事件处理器基类"""

    def __init__(self, name: str, event_bus: EventBus):
        self.name = name
        self.event_bus = event_bus
        self._handled_count = 0

    async def handle(self, event: Event):
        """处理事件"""
        self._handled_count += 1


class LoggingHandler(EventHandler):
    """
    日志处理器

    记录所有事件到日志
    """

    def __init__(self, event_bus: EventBus):
        super().__init__("LoggingHandler", event_bus)
        self.event_bus.subscribe("agent.*", self.handle)
        self.event_bus.subscribe("tool.*", self.handle)
        self._logs: List[Dict] = []

    async def handle(self, event: Event):
        await super().handle(event)
        log_entry = {
            "timestamp": event.timestamp.isoformat(),
            "type": event.event_type,
            "source": event.source,
            "data": str(event.data)[:50],
        }
        self._logs.append(log_entry)
        print(f"    [Logger] {event.event_type}: {str(event.data)[:40]}...")


class MonitoringHandler(EventHandler):
    """
    监控处理器

    监控异常事件并触发告警
    """

    def __init__(self, event_bus: EventBus):
        super().__init__("MonitoringHandler", event_bus)
        self.event_bus.subscribe(AgentEventType.AGENT_ERROR, self.handle)
        self.event_bus.subscribe(AgentEventType.TOOL_FAILED, self.handle)
        self.event_bus.subscribe(AgentEventType.SYSTEM_ALERT, self.handle)
        self._alerts: List[Dict] = []

    async def handle(self, event: Event):
        await super().handle(event)
        alert = {
            "timestamp": event.timestamp.isoformat(),
            "type": event.event_type,
            "severity": event.priority.value,
            "details": event.data,
        }
        self._alerts.append(alert)
        print(f"    [Alert] {event.priority.value.upper()}: {event.event_type}")


class MemoryIndexHandler(EventHandler):
    """
    记忆索引处理器

    当新记忆存储时，更新索引
    """

    def __init__(self, event_bus: EventBus):
        super().__init__("MemoryIndexHandler", event_bus)
        self.event_bus.subscribe(AgentEventType.MEMORY_STORED, self.handle)
        self._index_count = 0

    async def handle(self, event: Event):
        await super().handle(event)
        self._index_count += 1
        print(f"    [Index] 更新记忆索引 #{self._index_count}: {event.data.get('content', '')[:30]}...")


# ============================================================================
# 第六部分：消息队列（异步事件处理）
# ============================================================================

class MessageQueue:
    """
    消息队列

    【原理】
    消息队列实现异步事件处理：
    1. 生产者将事件放入队列
    2. 消费者从队列中取出事件处理
    3. 支持优先级排序
    4. 支持重试和死信队列

    【vs EventBus】
    - EventBus: 同步分发，适合进程内通信
    - MessageQueue: 异步缓冲，适合解耦和削峰
    """

    def __init__(self, name: str, max_size: int = 1000):
        self.name = name
        self.max_size = max_size
        self._queue: asyncio.Queue = None
        self._dead_letter: List[Event] = []
        self._processed_count = 0

    async def initialize(self):
        """初始化队列"""
        self._queue = asyncio.Queue(maxsize=self.max_size)

    async def enqueue(self, event: Event) -> bool:
        """
        入队

        Returns:
            是否成功入队
        """
        if self._queue.full():
            print(f"  [Queue:{self.name}] 队列已满，事件丢弃: {event.event_type}")
            return False

        await self._queue.put(event)
        return True

    async def dequeue(self) -> Optional[Event]:
        """出队"""
        if self._queue.empty():
            return None
        return await self._queue.get()

    async def process(self, handler: Callable, max_events: int = 10):
        """
        处理队列中的事件

        Args:
            handler: 事件处理函数
            max_events: 最多处理的事件数
        """
        processed = 0
        while processed < max_events and not self._queue.empty():
            event = await self._queue.get()
            try:
                await handler(event)
                self._processed_count += 1
                processed += 1
            except Exception as e:
                print(f"  [Queue:{self.name}] 处理失败: {e}")
                self._dead_letter.append(event)

    def get_stats(self) -> Dict[str, int]:
        return {
            "queue_size": self._queue.qsize() if self._queue else 0,
            "processed": self._processed_count,
            "dead_letter": len(self._dead_letter),
        }


# ============================================================================
# 第七部分：事件溯源（Event Sourcing）
# ============================================================================

class EventStore:
    """
    事件存储（事件溯源）

    【原理】
    事件溯源不是存储当前状态，而是存储所有事件：
    - 状态 = 初始状态 + 所有事件的累积
    - 可以回放事件重建任意时间点的状态
    - 完整的审计追踪

    【优势】
    - 完整的变更历史
    - 可以时间旅行（重建历史状态）
    - 支持 CQRS 模式
    """

    def __init__(self):
        self._events: List[Event] = []
        self._snapshots: Dict[int, Dict] = {}

    def append(self, event: Event) -> None:
        """追加事件"""
        self._events.append(event)

        # 每 10 个事件创建快照
        if len(self._events) % 10 == 0:
            self._snapshots[len(self._events)] = {
                "snapshot_at": len(self._events),
                "timestamp": datetime.now().isoformat(),
            }

    def replay(self, from_version: int = 0, to_version: int = -1) -> List[Event]:
        """
        回放事件

        Args:
            from_version: 起始版本
            to_version: 结束版本（-1 表示最新）
        """
        if to_version == -1:
            to_version = len(self._events)
        return self._events[from_version:to_version]

    def get_current_state(self) -> Dict:
        """获取当前状态摘要"""
        return {
            "total_events": len(self._events),
            "current_version": len(self._events),
            "snapshots": len(self._snapshots),
            "last_event": self._events[-1].event_type if self._events else None,
        }


# ============================================================================
# 第八部分：完整演示
# ============================================================================

async def demonstrate_event_bus():
    """演示事件总线"""

    print("\n" + "=" * 60)
    print("演示1: 事件总线")
    print("=" * 60)

    # 创建事件总线
    bus = EventBus("agent-bus")

    # 创建处理器
    agent = EventDrivenAgent("ResearchAgent", bus)
    logger = LoggingHandler(bus)
    monitor = MonitoringHandler(bus)
    memory_handler = MemoryIndexHandler(bus)

    # 发布事件
    print("\n  -- 发布用户消息事件 --")
    await bus.publish(Event(
        event_type=AgentEventType.USER_MESSAGE,
        data={"content": "帮我分析一下最新的AI发展趋势"},
        source="user_001",
    ))

    print("\n  -- 发布工具调用事件 --")
    await bus.publish(Event(
        event_type=AgentEventType.TOOL_CALLED,
        data={"tool_name": "web_search", "query": "AI发展趋势 2024"},
        source="ResearchAgent",
    ))

    await bus.publish(Event(
        event_type=AgentEventType.TOOL_COMPLETED,
        data={"tool_name": "web_search", "results_count": 15},
        source="ResearchAgent",
    ))

    print("\n  -- 发布记忆存储事件 --")
    await bus.publish(Event(
        event_type=AgentEventType.MEMORY_STORED,
        data={"content": "AI发展趋势分析结果..."},
        source="ResearchAgent",
    ))

    # 统计
    print(f"\n  -- 事件统计 --")
    stats = bus.get_stats()
    for event_type, count in stats.items():
        print(f"    {event_type}: {count}")


async def demonstrate_message_queue():
    """演示消息队列"""

    print("\n" + "=" * 60)
    print("演示2: 消息队列")
    print("=" * 60)

    queue = MessageQueue("task-queue")
    await queue.initialize()

    # 生产事件
    events = [
        Event(event_type="task.process", data={"task_id": "001", "type": "search"}),
        Event(event_type="task.process", data={"task_id": "002", "type": "analyze"}),
        Event(event_type="task.process", data={"task_id": "003", "type": "report"}),
    ]

    print("\n  -- 入队 --")
    for event in events:
        success = await queue.enqueue(event)
        print(f"    入队: {event.data['task_id']} -> {'OK' if success else 'FAIL'}")

    # 消费事件
    print("\n  -- 消费 --")

    async def task_handler(event: Event):
        print(f"    处理任务: {event.data['task_id']} ({event.data['type']})")

    await queue.process(task_handler)

    # 统计
    stats = queue.get_stats()
    print(f"\n  -- 队列统计 --")
    print(f"    已处理: {stats['processed']}")
    print(f"    队列中: {stats['queue_size']}")
    print(f"    死信: {stats['dead_letter']}")


async def demonstrate_event_sourcing():
    """演示事件溯源"""

    print("\n" + "=" * 60)
    print("演示3: 事件溯源")
    print("=" * 60)

    store = EventStore()

    # 记录对话事件
    events_data = [
        ("session.created", {"user": "张三", "session_id": "s001"}),
        ("user.message", {"content": "你好"}),
        ("agent.response", {"content": "你好！有什么可以帮你的？"}),
        ("user.message", {"content": "查一下天气"}),
        ("tool.called", {"tool": "weather_api"}),
        ("tool.completed", {"result": "晴天 25C"}),
        ("agent.response", {"content": "今天天气晴朗，25度"}),
        ("memory.stored", {"key": "weather_query", "value": "晴天"}),
    ]

    print("\n  -- 记录事件 --")
    for event_type, data in events_data:
        event = Event(event_type=event_type, data=data, source="demo")
        store.append(event)
        print(f"    [{event_type}] {str(data)[:40]}")

    # 回放
    print("\n  -- 回放事件 (版本 2-5) --")
    replayed = store.replay(from_version=2, to_version=5)
    for event in replayed:
        print(f"    [{event.event_type}] {str(event.data)[:40]}")

    # 状态
    state = store.get_current_state()
    print(f"\n  -- 当前状态 --")
    print(f"    总事件数: {state['total_events']}")
    print(f"    当前版本: {state['current_version']}")
    print(f"    快照数: {state['snapshots']}")


async def main():
    """主函数"""
    print("=" * 60)
    print("第26课: Event-Driven Architecture (事件驱动架构)")
    print("=" * 60)

    # 1. 概览
    explain_eda_overview()

    # 2. 事件总线演示
    await demonstrate_event_bus()

    # 3. 消息队列演示
    await demonstrate_message_queue()

    # 4. 事件溯源演示
    await demonstrate_event_sourcing()

    # 总结
    print("\n" + "=" * 60)
    print("课程总结")
    print("=" * 60)
    print("""
  本课介绍了事件驱动架构在 Agent 系统中的应用：

  核心概念：
  1. Event（事件）: 系统中有意义的状态变化
  2. EventBus（事件总线）: 事件的发布/订阅中心
  3. MessageQueue（消息队列）: 异步事件缓冲和处理
  4. EventSourcing（事件溯源）: 通过事件重建状态

  设计模式：
  - Pub/Sub: 发布者和订阅者解耦
  - 事件处理器: 独立的事件消费者
  - 中间件: 事件拦截和预处理
  - 死信队列: 处理失败的事件

  实际应用场景：
  - 多 Agent 协作: 通过事件总线通信
  - 异步任务处理: 消息队列削峰
  - 审计追踪: 事件溯源记录所有变更
  - 实时监控: 事件驱动的告警系统
""")


if __name__ == "__main__":
    asyncio.run(main())
