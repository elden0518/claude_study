"""
==============================================================================
第二十二课（补充）：容错模式与弹性设计
==============================================================================

【为什么需要单独一课？】
现有课程没有系统讲解 Agent 的容错机制。
生产级 Agent 必须能在故障中优雅恢复，保证系统可用性。

【学习目标】
- 掌握重试策略（指数退避、抖动）
- 理解断路器模式（Circuit Breaker）
- 学会超时处理和降级策略
- 掌握检查点与状态恢复
- 构建弹性 Agent 系统

【核心概念】
- Retry with Backoff（指数退避重试）
- Circuit Breaker（断路器）
- Timeout Handling（超时处理）
- Checkpoint & Recovery（检查点恢复）
- Graceful Degradation（优雅降级）

==============================================================================
"""

import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# 第一部分：重试策略
# ============================================================================

class BackoffStrategy(Enum):
    """退避策略"""
    FIXED = "fixed"           # 固定间隔
    LINEAR = "linear"         # 线性增长
    EXPONENTIAL = "exponential"  # 指数增长


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter: bool = True  # 添加随机抖动避免雪崩

    def get_delay(self, attempt: int) -> float:
        """计算第 N 次重试的延迟"""
        if self.strategy == BackoffStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == BackoffStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        else:  # EXPONENTIAL
            delay = self.base_delay * (2 ** attempt)

        delay = min(delay, self.max_delay)

        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)

        return delay


class RetryHandler:
    """
    重试处理器

    【原理】
    当操作失败时，按配置的策略自动重试：
    - 指数退避：每次重试间隔翻倍
    - 抖动：添加随机性避免多个客户端同时重试
    """

    def __init__(self, config: RetryConfig):
        self.config = config
        self._retry_count = 0
        self._total_retries = 0

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        执行函数，失败时自动重试

        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数

        Returns:
            函数返回值

        Raises:
            最后一次重试的异常
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    print(f"    [Retry] 第 {attempt} 次重试成功")
                    self._total_retries += attempt
                return result
            except Exception as e:
                last_exception = e
                self._retry_count += 1

                if attempt < self.config.max_retries:
                    delay = self.config.get_delay(attempt)
                    print(f"    [Retry] 第 {attempt + 1} 次失败: {e}")
                    print(f"    [Retry] {delay:.2f}s 后重试...")
                    time.sleep(delay)
                else:
                    print(f"    [Retry] 达到最大重试次数 ({self.config.max_retries})")

        raise last_exception

    def get_stats(self) -> Dict[str, int]:
        """获取重试统计"""
        return {
            "total_retries": self._total_retries,
            "retry_events": self._retry_count
        }


# ============================================================================
# 第二部分：断路器模式
# ============================================================================

class CircuitState(Enum):
    """断路器状态"""
    CLOSED = "closed"       # 正常（关闭）
    OPEN = "open"           # 断开（熔断）
    HALF_OPEN = "half_open"  # 半开（试探）


class CircuitBreaker:
    """
    断路器

    【原理】
    三种状态转换：
    - CLOSED: 正常状态，请求通过
    - OPEN: 熔断状态，请求直接失败
    - HALF_OPEN: 试探状态，允许少量请求通过

    状态转换：
    CLOSED --(失败率超阈值)--> OPEN
    OPEN --(冷却时间到)--> HALF_OPEN
    HALF_OPEN --(试探成功)--> CLOSED
    HALF_OPEN --(试探失败)--> OPEN
    """

    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 half_open_max: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """获取当前状态"""
        if self._state == CircuitState.OPEN:
            # 检查是否可以进入半开状态
            if self._last_failure_time:
                elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
        return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        通过断路器调用函数

        【流程】
        1. 检查状态
        2. OPEN 状态直接抛出异常
        3. HALF_OPEN 状态限制调用次数
        4. 根据成功/失败更新状态
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitBreakerOpenError("断路器已断开，请稍后重试")

        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max:
                raise CircuitBreakerOpenError("半开状态试探次数已达上限")
            self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """成功回调"""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                print("    [Circuit] 试探成功，断路器恢复为 CLOSED")
        else:
            self._failure_count = max(0, self._failure_count - 1)

    def _on_failure(self):
        """失败回调"""
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            print("    [Circuit] 试探失败，断路器重新断开")
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            print(f"    [Circuit] 失败次数达到阈值，断路器断开 (OPEN)")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None
        }


class CircuitBreakerOpenError(Exception):
    """断路器打开异常"""
    pass


# ============================================================================
# 第三部分：超时处理
# ============================================================================

class TimeoutHandler:
    """
    超时处理器

    【原理】
    为操作设置超时时间，超时后：
    - 取消操作
    - 执行降级逻辑
    - 记录超时事件
    """

    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds
        self._timeout_count = 0

    def execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """带超时的执行"""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"操作超时 ({self.timeout_seconds}s)")

        # 设置信号处理器（仅 Unix 系统支持）
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.timeout_seconds))

        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # 取消闹钟
            return result
        except TimeoutError:
            self._timeout_count += 1
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def get_stats(self) -> Dict[str, int]:
        return {"timeout_count": self._timeout_count}


# ============================================================================
# 第四部分：检查点与状态恢复
# ============================================================================

@dataclass
class Checkpoint:
    """检查点"""
    id: str
    state: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """
    检查点管理器

    【功能】
    - 保存 Agent 执行状态
    - 从故障中恢复
    - 支持断点续传
    """

    def __init__(self):
        self.checkpoints: Dict[str, Checkpoint] = {}
        self._current_checkpoint: Optional[str] = None

    def save(self, checkpoint_id: str, state: Dict[str, Any],
             metadata: Dict = None) -> Checkpoint:
        """保存检查点"""
        checkpoint = Checkpoint(
            id=checkpoint_id,
            state=state.copy(),
            metadata=metadata or {}
        )
        self.checkpoints[checkpoint_id] = checkpoint
        self._current_checkpoint = checkpoint_id
        print(f"    [Checkpoint] 保存: {checkpoint_id}")
        return checkpoint

    def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """加载检查点"""
        checkpoint = self.checkpoints.get(checkpoint_id)
        if checkpoint:
            print(f"    [Checkpoint] 恢复: {checkpoint_id}")
        return checkpoint

    def get_latest(self) -> Optional[Checkpoint]:
        """获取最新检查点"""
        if self._current_checkpoint:
            return self.checkpoints.get(self._current_checkpoint)
        return None

    def list_checkpoints(self) -> List[str]:
        """列出所有检查点"""
        return list(self.checkpoints.keys())


# ============================================================================
# 第五部分：优雅降级
# ============================================================================

class DegradationLevel(Enum):
    """降级级别"""
    NORMAL = 0       # 正常
    LIMITED = 1      # 功能受限
    BASIC = 2        # 基础功能
    EMERGENCY = 3    # 紧急模式


class GracefulDegradation:
    """
    优雅降级管理器

    【原理】
    当系统出现问题时，逐步降低功能但保持可用：
    - Level 0: 正常服务
    - Level 1: 禁用非核心功能
    - Level 2: 只提供基础回复
    - Level 3: 返回预设回复
    """

    def __init__(self):
        self._level = DegradationLevel.NORMAL
        self._fallback_responses = [
            "系统繁忙，请稍后再试。",
            "暂时无法处理您的请求。",
            "服务暂时不可用，感谢您的耐心。",
        ]

    @property
    def level(self) -> DegradationLevel:
        return self._level

    def degrade(self, reason: str = ""):
        """降级"""
        if self._level.value < DegradationLevel.EMERGENCY.value:
            self._level = DegradationLevel(self._level.value + 1)
            print(f"    [Degrade] 降级到 Level {self._level.value}: {reason}")

    def recover(self):
        """恢复"""
        if self._level.value > DegradationLevel.NORMAL.value:
            self._level = DegradationLevel(self._level.value - 1)
            print(f"    [Degrade] 恢复到 Level {self._level.value}")

    def execute(self, primary_func: Callable, fallback_func: Callable = None,
                *args, **kwargs) -> Any:
        """执行，根据降级级别选择功能"""
        if self._level == DegradationLevel.NORMAL:
            return primary_func(*args, **kwargs)
        elif self._level == DegradationLevel.LIMITED:
            if fallback_func:
                return fallback_func(*args, **kwargs)
            return primary_func(*args, **kwargs)
        elif self._level == DegradationLevel.BASIC:
            return {"response": "基础回复模式", "level": self._level.value}
        else:
            return {"response": random.choice(self._fallback_responses),
                    "level": self._level.value}


# ============================================================================
# 第六部分：弹性 Agent
# ============================================================================

class ResilientAgent:
    """
    弹性 Agent

    【整合所有容错模式】
    - 重试处理
    - 断路器保护
    - 超时控制
    - 检查点恢复
    - 优雅降级
    """

    def __init__(self, name: str):
        self.name = name
        self.retry_handler = RetryHandler(RetryConfig(max_retries=3, base_delay=0.1))
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)
        self.checkpoint_manager = CheckpointManager()
        self.degradation = GracefulDegradation()
        self._step_count = 0

    def process(self, task: str) -> Dict[str, Any]:
        """处理任务（带完整容错）"""
        print(f"\n{'='*60}")
        print(f"弹性 Agent 处理任务: {task}")
        print(f"{'='*60}")

        try:
            # 1. 保存检查点
            self.checkpoint_manager.save(
                f"step_{self._step_count}",
                {"task": task, "status": "started"}
            )

            # 2. 通过断路器和重试执行
            result = self.circuit_breaker.call(
                self.retry_handler.execute,
                self._do_process,
                task
            )

            # 3. 保存完成检查点
            self._step_count += 1
            self.checkpoint_manager.save(
                f"step_{self._step_count}",
                {"task": task, "status": "completed", "result": result}
            )

            return result

        except CircuitBreakerOpenError as e:
            print(f"    [!] 断路器断开: {e}")
            self.degradation.degrade("断路器断开")
            return self.degradation.execute(
                lambda t: {"error": str(e)},
                lambda t: {"response": "服务暂时不可用", "fallback": True},
                task
            )

        except Exception as e:
            print(f"    [!] 处理失败: {e}")
            self.degradation.degrade(str(e))
            return {"error": str(e), "fallback": True}

    def _do_process(self, task: str) -> Dict[str, Any]:
        """实际处理逻辑（模拟可能失败的操作）"""
        print(f"    [Process] 执行任务: {task}")

        # 模拟随机失败
        if random.random() < 0.3:
            raise ConnectionError("网络连接失败")

        return {
            "task": task,
            "result": f"成功处理: {task}",
            "timestamp": datetime.now().isoformat()
        }

    def recover_from_checkpoint(self) -> Optional[Dict]:
        """从检查点恢复"""
        checkpoint = self.checkpoint_manager.get_latest()
        if checkpoint:
            print(f"    [Recover] 从检查点恢复: {checkpoint.id}")
            return checkpoint.state
        return None

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "retry": self.retry_handler.get_stats(),
            "degradation_level": self.degradation.level.value,
            "checkpoints": self.checkpoint_manager.list_checkpoints()
        }


# ============================================================================
# 第七部分：完整示例
# ============================================================================

def demo_resilience_patterns():
    """演示容错模式"""

    print("=" * 60)
    print("容错模式完整演示")
    print("=" * 60)

    # 1. 重试策略演示
    print("\n[Step 1] 重试策略演示...")
    config = RetryConfig(max_retries=3, base_delay=0.1, strategy=BackoffStrategy.EXPONENTIAL)
    handler = RetryHandler(config)

    call_count = [0]
    def flaky_operation():
        call_count[0] += 1
        if call_count[0] < 3:
            raise ValueError(f"模拟失败 #{call_count[0]}")
        return "操作成功"

    try:
        result = handler.execute(flaky_operation)
        print(f"    结果: {result}")
        print(f"    重试统计: {handler.get_stats()}")
    except Exception as e:
        print(f"    最终失败: {e}")

    # 2. 断路器演示
    print("\n[Step 2] 断路器演示...")
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=2.0)

    def always_fail():
        raise RuntimeError("服务不可用")

    for i in range(5):
        try:
            cb.call(always_fail)
        except Exception as e:
            print(f"    调用 {i+1}: {type(e).__name__} - 状态: {cb.state.value}")

    # 3. 检查点演示
    print("\n[Step 3] 检查点演示...")
    cm = CheckpointManager()
    cm.save("step1", {"progress": 50, "data": "partial"})
    cm.save("step2", {"progress": 100, "data": "complete"})

    latest = cm.get_latest()
    print(f"    最新检查点: {latest.id if latest else 'None'}")
    print(f"    所有检查点: {cm.list_checkpoints()}")

    # 4. 弹性 Agent 演示
    print("\n[Step 4] 弹性 Agent 演示...")
    agent = ResilientAgent("DemoAgent")

    # 多次尝试（可能触发重试和断路器）
    for i in range(3):
        result = agent.process(f"任务 {i+1}")
        print(f"    结果: {result.get('result', result.get('error', 'unknown'))}")

    # 查看状态
    print(f"\n    Agent 状态: {agent.get_status()}")

    # 5. 降级演示
    print("\n[Step 5] 优雅降级演示...")
    degradation = GracefulDegradation()

    def primary():
        return "完整功能回复"

    def fallback():
        return "简化功能回复"

    print(f"    Level {degradation.level.value}: {degradation.execute(primary, fallback)}")
    degradation.degrade("测试降级")
    print(f"    Level {degradation.level.value}: {degradation.execute(primary, fallback)}")
    degradation.degrade("再次降级")
    print(f"    Level {degradation.level.value}: {degradation.execute(primary, fallback)}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("容错模式与弹性设计")
    print("=" * 60)

    demo_resilience_patterns()

    print("\n" + "=" * 60)
    print("课程总结")
    print("=" * 60)
    print("""
  本课介绍了 Agent 的容错模式：

  核心知识点：
  1. 重试策略：固定/线性/指数退避 + 抖动
  2. 断路器：CLOSED -> OPEN -> HALF_OPEN 状态转换
  3. 超时处理：为操作设置超时限制
  4. 检查点：保存/恢复执行状态
  5. 优雅降级：逐步降低功能但保持可用
  6. 弹性 Agent：整合所有容错模式

  设计原则：
  - Fail Fast: 快速失败，快速恢复
  - Fail Safe: 失败时保持安全状态
  - Fail Graceful: 优雅降级而非崩溃
    """)
