"""
==============================================================================
第二十八课（进阶）：State Machine & Workflow Orchestration（状态机与工作流编排）
==============================================================================

【为什么需要单独一课？】
现有课程没有系统讲解有限状态机（FSM）和 DAG 工作流编排。
这是构建复杂 Agent 工作流的核心技术。

【学习目标】
- 掌握有限状态机（FSM）在 Agent 中的应用
- 掌握 DAG（有向无环图）工作流编排
- 学会状态持久化和恢复
- 理解条件路由和并行执行
- 构建复杂的 Agent 工作流

【核心概念】
- Finite State Machine（有限状态机）
- State Transition（状态转换）
- DAG（有向无环图）
- Workflow Orchestration（工作流编排）
- State Persistence（状态持久化）
- Conditional Routing（条件路由）

==============================================================================
"""

import json
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ============================================================================
# 第一部分：状态机概览
# ============================================================================

def explain_state_machine_overview():
    """
    状态机在 Agent 系统中的应用

    Agent 的整个生命周期就是状态机：
    - 等待输入 -> 处理中 -> 等待工具 -> 处理工具结果 -> 回复
    """

    print("=" * 60)
    print("State Machine & Workflow Orchestration 概览")
    print("=" * 60)

    print("""
  -- 什么是有限状态机（FSM）？--

  FSM = 有限状态 + 转换条件 + 动作

  状态(State): 系统在某一时刻的情况
  转换(Transition): 从一个状态到另一个状态
  条件(Guard): 触发转换的条件

  -- Agent 中的状态机应用 --

  1. Agent 生命周期:
     IDLE -> THINKING -> TOOL_CALLING -> OBSERVING -> RESPONDING -> IDLE

  2. 会话状态:
     CREATED -> ACTIVE -> WAITING -> RESOLVED -> CLOSED

  3. 任务状态:
     PENDING -> RUNNING -> WAITING_APPROVAL -> COMPLETED/FAILED

  4. 工作流编排:
     多个步骤按 DAG 顺序执行
    """)


# ============================================================================
# 第二部分：有限状态机实现
# ============================================================================

class StateStatus(Enum):
    """状态"""
    IDLE = "idle"
    THINKING = "thinking"
    TOOL_CALLING = "tool_calling"
    OBSERVING = "observing"
    RESPONDING = "responding"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Transition:
    """状态转换"""
    from_state: StateStatus
    to_state: StateStatus
    event: str
    guard: Optional[Callable] = None
    action: Optional[Callable] = None
    description: str = ""


class StateMachine:
    """
    有限状态机

    【原理】
    1. 定义所有可能的状态
    2. 定义状态之间的转换规则
    3. 每个转换可以有条件和动作
    4. 当前状态 + 事件 -> 新状态

    【Agent 应用】
    - 控制 Agent 的执行流程
    - 防止非法状态转换
    - 记录状态变化历史
    """

    def __init__(self, name: str, initial_state: StateStatus):
        self.name = name
        self.current_state = initial_state
        self.initial_state = initial_state
        self._transitions: List[Transition] = []
        self._history: List[Dict] = []
        self._state_data: Dict[str, Any] = {}

    def add_transition(
        self,
        from_state: StateStatus,
        to_state: StateStatus,
        event: str,
        guard: Callable = None,
        action: Callable = None,
        description: str = "",
    ) -> "StateMachine":
        """添加转换规则"""
        self._transitions.append(
            Transition(from_state, to_state, event, guard, action, description)
        )
        return self

    def trigger(self, event: str, data: Dict = None) -> bool:
        """
        触发事件，尝试状态转换

        Args:
            event: 事件名称
            data: 事件数据

        Returns:
            是否转换成功
        """
        # 找到匹配的转换
        matching = [
            t for t in self._transitions
            if t.from_state == self.current_state and t.event == event
        ]

        if not matching:
            print(f"  [FSM:{self.name}] 无匹配转换: {self.current_state.value} + {event}")
            return False

        transition = matching[0]

        # 检查条件
        if transition.guard and not transition.guard(data):
            print(f"  [FSM:{self.name}] 条件不满足: {transition.description}")
            return False

        # 执行动作
        if transition.action:
            transition.action(data)

        # 记录历史
        old_state = self.current_state
        self._history.append({
            "from": old_state.value,
            "to": transition.to_state.value,
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "data": str(data)[:50] if data else None,
        })

        # 转换状态
        self.current_state = transition.to_state
        print(f"  [FSM:{self.name}] {old_state.value} -> {transition.to_state.value} (event: {event})")
        return True

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def reset(self):
        self.current_state = self.initial_state
        self._history.clear()


# ============================================================================
# 第三部分：Agent 状态机
# ============================================================================

class AgentStateMachine:
    """
    Agent 专用状态机

    封装了 Agent 典型的生命周期状态转换：
    IDLE -> THINKING -> TOOL_CALLING -> OBSERVING -> RESPONDING -> IDLE
    或者
    IDLE -> THINKING -> RESPONDING -> IDLE (不需要工具)
    """

    def __init__(self, agent_name: str = "Agent"):
        self.fsm = StateMachine(agent_name, StateStatus.IDLE)
        self._setup_transitions()

    def _setup_transitions(self):
        """设置 Agent 状态转换规则"""
        fsm = self.fsm

        # IDLE -> THINKING (收到用户输入)
        fsm.add_transition(
            StateStatus.IDLE, StateStatus.THINKING,
            "user_input",
            description="收到用户输入，开始思考",
        )

        # THINKING -> TOOL_CALLING (决定调用工具)
        fsm.add_transition(
            StateStatus.THINKING, StateStatus.TOOL_CALLING,
            "call_tool",
            description="决定调用工具",
        )

        # THINKING -> RESPONDING (直接回复)
        fsm.add_transition(
            StateStatus.THINKING, StateStatus.RESPONDING,
            "direct_response",
            description="直接回复，不需要工具",
        )

        # TOOL_CALLING -> OBSERVING (工具返回结果)
        fsm.add_transition(
            StateStatus.TOOL_CALLING, StateStatus.OBSERVING,
            "tool_result",
            description="工具执行完成",
        )

        # TOOL_CALLING -> FAILED (工具执行失败)
        fsm.add_transition(
            StateStatus.TOOL_CALLING, StateStatus.FAILED,
            "tool_error",
            description="工具执行失败",
        )

        # OBSERVING -> THINKING (继续思考)
        fsm.add_transition(
            StateStatus.OBSERVING, StateStatus.THINKING,
            "continue_thinking",
            description="基于工具结果继续思考",
        )

        # OBSERVING -> TOOL_CALLING (调用更多工具)
        fsm.add_transition(
            StateStatus.OBSERVING, StateStatus.TOOL_CALLING,
            "call_another_tool",
            description="需要调用更多工具",
        )

        # RESPONDING -> IDLE (回复完成)
        fsm.add_transition(
            StateStatus.RESPONDING, StateStatus.IDLE,
            "response_complete",
            description="回复完成，回到空闲",
        )

        # FAILED -> IDLE (从失败恢复)
        fsm.add_transition(
            StateStatus.FAILED, StateStatus.IDLE,
            "recover",
            description="从失败恢复",
        )

    def run_demo(self):
        """运行 Agent 状态机演示"""

        print(f"\n{'='*50}")
        print(f"Agent 状态机演示")
        print(f"{'='*50}")

        # 场景1: 简单问答（不需要工具）
        print("\n  -- 场景1: 简单问答 --")
        self.fsm.reset()
        self.fsm.trigger("user_input", {"content": "你好"})
        self.fsm.trigger("direct_response")
        self.fsm.trigger("response_complete")

        # 场景2: 需要工具调用
        print("\n  -- 场景2: 工具调用 --")
        self.fsm.reset()
        self.fsm.trigger("user_input", {"content": "查天气"})
        self.fsm.trigger("call_tool", {"tool": "weather_api"})
        self.fsm.trigger("tool_result", {"result": "晴天"})
        self.fsm.trigger("continue_thinking")
        self.fsm.trigger("direct_response")
        self.fsm.trigger("response_complete")

        # 场景3: 工具失败恢复
        print("\n  -- 场景3: 工具失败 --")
        self.fsm.reset()
        self.fsm.trigger("user_input", {"content": "发邮件"})
        self.fsm.trigger("call_tool", {"tool": "email_api"})
        self.fsm.trigger("tool_error", {"error": "连接超时"})
        self.fsm.trigger("recover")

        # 状态历史
        print(f"\n  -- 状态转换历史 --")
        for entry in self.fsm.get_history():
            print(f"    {entry['from']} -> {entry['to']} ({entry['event']})")


# ============================================================================
# 第四部分：DAG 工作流
# ============================================================================

@dataclass
class WorkflowTask:
    """工作流任务"""
    task_id: str
    name: str
    handler: Callable
    dependencies: List[str] = field(default_factory=list)
    result: Any = None
    status: str = "pending"  # pending/running/completed/failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class DAGWorkflow:
    """
    DAG 工作流引擎

    【原理】
    DAG = 有向无环图
    - 节点: 任务
    - 边: 依赖关系
    - 无环: 不能有循环依赖

    【执行规则】
    1. 没有依赖的任务可以立即执行
    2. 有依赖的任务必须等所有依赖完成
    3. 没有依赖关系的任务可以并行执行

    【vs 状态机】
    - 状态机: 线性流程，一次一个状态
    - DAG: 可以并行，多路径同时执行
    """

    def __init__(self, name: str):
        self.name = name
        self._tasks: Dict[str, WorkflowTask] = {}
        self._execution_order: List[str] = []

    def add_task(
        self, task_id: str, name: str, handler: Callable,
        dependencies: List[str] = None
    ) -> "DAGWorkflow":
        """添加任务"""
        self._tasks[task_id] = WorkflowTask(
            task_id=task_id,
            name=name,
            handler=handler,
            dependencies=dependencies or [],
        )
        return self

    def _topological_sort(self) -> List[str]:
        """
        拓扑排序

        确定任务的执行顺序，确保依赖关系正确
        """
        in_degree = {tid: 0 for tid in self._tasks}
        for task in self._tasks.values():
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[task.task_id] += 1

        # 从入度为0的节点开始
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            # 按层级处理（同一层可以并行）
            next_queue = []
            for current in queue:
                result.append(current)
                # 减少后续节点的入度
                for tid, task in self._tasks.items():
                    if current in task.dependencies:
                        in_degree[tid] -= 1
                        if in_degree[tid] == 0:
                            next_queue.append(tid)
            queue = next_queue

        if len(result) != len(self._tasks):
            raise ValueError("工作流存在循环依赖!")

        return result

    def execute(self, context: Dict = None) -> Dict[str, Any]:
        """
        执行工作流

        Args:
            context: 共享上下文数据

        Returns:
            所有任务的结果
        """
        context = context or {}
        results = {}

        print(f"\n{'='*50}")
        print(f"DAG 工作流执行: {self.name}")
        print(f"{'='*50}")

        # 拓扑排序
        order = self._topological_sort()
        print(f"\n  执行顺序: {' -> '.join(order)}")

        # 按顺序执行
        for task_id in order:
            task = self._tasks[task_id]

            # 检查依赖
            deps_ready = all(
                self._tasks[dep].status == "completed"
                for dep in task.dependencies
            )
            if not deps_ready:
                print(f"\n  [SKIP] {task.name}: 依赖未完成")
                task.status = "failed"
                continue

            # 执行任务
            print(f"\n  [RUN] {task.name} (id: {task_id})")
            task.status = "running"
            task.start_time = time.time()

            try:
                # 将依赖的结果传入
                dep_results = {
                    dep: self._tasks[dep].result
                    for dep in task.dependencies
                }
                task.result = task.handler(context, dep_results)
                task.status = "completed"
                results[task_id] = task.result
                print(f"    -> 完成: {str(task.result)[:50]}...")
            except Exception as e:
                task.status = "failed"
                print(f"    -> 失败: {e}")

            task.end_time = time.time()

        # 总结
        print(f"\n  -- 执行总结 --")
        for task in self._tasks.values():
            duration = (task.end_time - task.start_time) * 1000 if task.end_time and task.start_time else 0
            print(f"    {task.task_id}: {task.status} ({duration:.0f}ms)")

        return results


# ============================================================================
# 第五部分：条件路由
# ============================================================================

class ConditionalRouter:
    """
    条件路由器

    【原理】
    根据运行时条件决定执行路径：
    - 基于数据内容路由
    - 基于状态路由
    - 基于优先级路由

    【应用场景】
    - 根据用户意图路由到不同处理流程
    - 根据错误类型选择恢复策略
    - 根据负载选择处理节点
    """

    def __init__(self, name: str):
        self.name = name
        self._routes: List[Dict] = []
        self._default_route: Optional[Callable] = None

    def add_route(
        self, condition: Callable, handler: Callable, priority: int = 0,
        description: str = ""
    ) -> "ConditionalRouter":
        """添加路由规则"""
        self._routes.append({
            "condition": condition,
            "handler": handler,
            "priority": priority,
            "description": description,
        })
        # 按优先级排序
        self._routes.sort(key=lambda r: r["priority"], reverse=True)
        return self

    def set_default(self, handler: Callable) -> "ConditionalRouter":
        """设置默认路由"""
        self._default_route = handler
        return self

    def route(self, data: Any) -> Any:
        """路由数据"""
        print(f"\n  [Router:{self.name}] 路由决策...")

        for route in self._routes:
            if route["condition"](data):
                desc = route["description"] or "matched"
                print(f"    -> 匹配: {desc}")
                return route["handler"](data)

        if self._default_route:
            print(f"    -> 默认路由")
            return self._default_route(data)

        print(f"    -> 无匹配路由")
        return None


# ============================================================================
# 第六部分：状态持久化
# ============================================================================

class StatePersistence:
    """
    状态持久化

    【原理】
    将工作流状态保存到持久存储：
    - 支持中断后恢复
    - 支持长时间运行的工作流
    - 支持故障恢复

    【存储方式】
    - 内存: 快速但易失
    - 文件: 简单但单机
    - 数据库: 持久但复杂
    """

    def __init__(self, storage_type: str = "memory"):
        self.storage_type = storage_type
        self._storage: Dict[str, Dict] = {}

    def save(self, workflow_id: str, state: Dict) -> None:
        """保存状态"""
        self._storage[workflow_id] = {
            "state": state,
            "saved_at": datetime.now().isoformat(),
            "version": len(self._storage.get(workflow_id, {}).get("versions", [])) + 1,
        }
        print(f"  [Persist] 保存状态: {workflow_id} (version: {self._storage[workflow_id]['version']})")

    def load(self, workflow_id: str) -> Optional[Dict]:
        """加载状态"""
        entry = self._storage.get(workflow_id)
        if entry:
            print(f"  [Persist] 加载状态: {workflow_id} (saved: {entry['saved_at']})")
            return entry["state"]
        print(f"  [Persist] 未找到状态: {workflow_id}")
        return None

    def delete(self, workflow_id: str) -> bool:
        """删除状态"""
        if workflow_id in self._storage:
            del self._storage[workflow_id]
            return True
        return False

    def list_all(self) -> List[Dict]:
        """列出所有保存的状态"""
        return [
            {"id": wid, "saved_at": data["saved_at"], "version": data["version"]}
            for wid, data in self._storage.items()
        ]


# ============================================================================
# 第七部分：完整工作流演示
# ============================================================================

def demonstrate_dag_workflow():
    """演示 DAG 工作流"""

    print("\n" + "=" * 60)
    print("演示: DAG 工作流 - 数据处理管道")
    print("=" * 60)

    # 定义任务处理函数
    def load_data(context, deps):
        time.sleep(0.05)
        return {"records": 1000, "source": "database"}

    def validate_data(context, deps):
        time.sleep(0.03)
        return {"valid": 950, "invalid": 50}

    def transform_data(context, deps):
        time.sleep(0.04)
        return {"transformed": 950, "format": "json"}

    def enrich_data(context, deps):
        time.sleep(0.03)
        return {"enriched": True, "added_fields": 5}

    def aggregate_data(context, deps):
        time.sleep(0.02)
        return {"aggregated": True, "groups": 10}

    def export_data(context, deps):
        time.sleep(0.03)
        return {"exported": True, "format": "csv", "rows": 950}

    # 构建 DAG
    #
    #   load -> validate -> transform -> enrich -> aggregate -> export
    #
    workflow = DAGWorkflow("数据处理管道")
    workflow.add_task("load", "加载数据", load_data)
    workflow.add_task("validate", "验证数据", validate_data, ["load"])
    workflow.add_task("transform", "转换数据", transform_data, ["validate"])
    workflow.add_task("enrich", "丰富数据", enrich_data, ["transform"])
    workflow.add_task("aggregate", "聚合数据", aggregate_data, ["enrich"])
    workflow.add_task("export", "导出数据", export_data, ["aggregate"])

    results = workflow.execute()
    print(f"\n  最终结果: {results.get('export', {})}")


def demonstrate_conditional_routing():
    """演示条件路由"""

    print("\n" + "=" * 60)
    print("演示: 条件路由 - 意图分发")
    print("=" * 60)

    router = ConditionalRouter("意图路由")

    # 添加路由规则
    router.add_route(
        lambda data: "天气" in data.get("text", ""),
        lambda data: "调用天气API处理",
        description="天气查询",
    )
    router.add_route(
        lambda data: any(kw in data.get("text", "") for kw in ["计算", "多少"]),
        lambda data: "调用计算器处理",
        description="数学计算",
    )
    router.add_route(
        lambda data: any(kw in data.get("text", "") for kw in ["搜索", "查找"]),
        lambda data: "调用搜索引擎处理",
        description="信息搜索",
    )
    router.set_default(lambda data: "通用对话处理")

    # 测试路由
    test_inputs = [
        {"text": "北京今天天气怎么样"},
        {"text": "计算 123 * 456 等于多少"},
        {"text": "搜索最新的AI论文"},
        {"text": "你好，请介绍一下自己"},
    ]

    for inp in test_inputs:
        result = router.route(inp)
        print(f"    输入: {inp['text']} -> {result}")


def demonstrate_state_persistence():
    """演示状态持久化"""

    print("\n" + "=" * 60)
    print("演示: 状态持久化")
    print("=" * 60)

    persistence = StatePersistence("memory")

    # 保存多个工作流状态
    workflows = [
        ("wf_001", {"step": "validate", "progress": 0.5, "data": "处理中..."}),
        ("wf_002", {"step": "export", "progress": 0.9, "data": "即将完成"}),
        ("wf_003", {"step": "load", "progress": 0.1, "data": "刚开始"}),
    ]

    print("\n  -- 保存状态 --")
    for wf_id, state in workflows:
        persistence.save(wf_id, state)

    # 加载状态
    print("\n  -- 恢复状态 --")
    loaded = persistence.load("wf_001")
    if loaded:
        print(f"    恢复: step={loaded['step']}, progress={loaded['progress']}")

    # 列出所有
    print("\n  -- 所有保存的状态 --")
    for entry in persistence.list_all():
        print(f"    {entry['id']}: v{entry['version']} (saved: {entry['saved_at'][:19]})")


def main():
    """主函数"""
    print("=" * 60)
    print("第28课: State Machine & Workflow Orchestration")
    print("(状态机与工作流编排)")
    print("=" * 60)

    # 1. 概览
    explain_state_machine_overview()

    # 2. Agent 状态机
    agent_fsm = AgentStateMachine("DemoAgent")
    agent_fsm.run_demo()

    # 3. DAG 工作流
    demonstrate_dag_workflow()

    # 4. 条件路由
    demonstrate_conditional_routing()

    # 5. 状态持久化
    demonstrate_state_persistence()

    # 总结
    print("\n" + "=" * 60)
    print("课程总结")
    print("=" * 60)
    print("""
  本课介绍了状态机与工作流编排在 Agent 系统中的应用：

  核心概念：
  1. 有限状态机(FSM): 定义状态和转换规则，控制执行流程
  2. DAG工作流: 有向无环图，支持并行和依赖管理
  3. 条件路由: 根据运行时条件选择执行路径
  4. 状态持久化: 保存和恢复工作流状态

  设计模式：
  - 状态模式: 将状态封装为对象
  - 观察者模式: 状态变化通知
  - 策略模式: 条件路由选择
  - 模板方法: 工作流步骤定义

  实际应用场景：
  - Agent 生命周期管理: FSM 控制状态转换
  - 复杂任务编排: DAG 管理多步骤依赖
  - 智能路由: 根据意图分发到不同处理流程
  - 长任务恢复: 状态持久化支持断点续传
""")


if __name__ == "__main__":
    main()
