"""
==============================================================================
第十一课：进阶 - 多 Agent 协作与高级模式
==============================================================================

【学习目标】
- 理解多 Agent 协作的架构模式
- 掌握 Supervisor 模式（协调者模式）
- 掌握 Plan-and-Execute 模式
- 掌握 Self-Reflection 模式
- 学会设计 Agent 间通信机制

【核心概念】
- 多 Agent 系统
- Supervisor 模式
- 任务分解与分配
- Agent 间通信
- 自我反思与改进

【前置知识】
- 第七课：Agents 模块
- 所有前序课程

==============================================================================
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field


# ============================================================================
# 第一部分：为什么需要多 Agent 协作？
# ============================================================================

def why_multi_agent():
    """
    解释为什么需要多 Agent 协作

    单 Agent 的局限性：
    1. 上下文窗口有限，无法处理复杂任务
    2. 单一视角，可能遗漏重要信息
    3. 无法并行处理多个子任务
    4. 难以处理需要不同专业知识的任务

    多 Agent 的优势：
    1. 分工协作，每个 Agent 专注一个领域
    2. 并行处理，提高效率
    3. 多视角验证，提高质量
    4. 模块化，易于扩展和维护
    """

    scenarios = [
        {
            "scenario": "研究团队",
            "agents": ["研究员", "分析师", "写作者"],
            "workflow": "研究员收集信息 → 分析师分析 → 写作者撰写报告",
        },
        {
            "scenario": "代码审查",
            "agents": ["安全专家", "性能专家", "风格检查员"],
            "workflow": "并行审查 → 汇总意见 → 生成报告",
        },
        {
            "scenario": "客服系统",
            "agents": ["意图识别", "知识检索", "回复生成"],
            "workflow": "识别意图 → 检索知识 → 生成回复",
        },
        {
            "scenario": "内容创作",
            "agents": ["创意生成", "内容撰写", "质量审核"],
            "workflow": "生成创意 → 撰写内容 → 审核修改",
        },
    ]

    print("=" * 60)
    print("多 Agent 协作场景")
    print("=" * 60)
    for s in scenarios:
        print(f"\n  场景: {s['scenario']}")
        print(f"    Agent: {', '.join(s['agents'])}")
        print(f"    流程: {s['workflow']}")
    print()


# ============================================================================
# 第二部分：Agent 间通信机制
# ============================================================================

class MessageType(Enum):
    """消息类型"""
    TASK = "task"               # 任务分配
    RESULT = "result"           # 任务结果
    REQUEST = "request"         # 请求协助
    RESPONSE = "response"       # 协助响应
    BROADCAST = "broadcast"     # 广播消息


class AgentMessage(BaseModel):
    """Agent 间通信消息"""
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}")
    from_agent: str = Field(description="发送者")
    to_agent: str = Field(description="接收者")
    type: MessageType = Field(description="消息类型")
    content: str = Field(description="消息内容")
    metadata: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class MessageBus:
    """
    消息总线

    负责 Agent 之间的消息传递。
    支持：
    - 点对点消息
    - 广播消息
    - 消息历史记录
    """

    def __init__(self):
        self._history: list[AgentMessage] = []
        self._subscribers: dict[str, list] = {}

    def send(self, message: AgentMessage):
        """发送消息"""
        self._history.append(message)
        print(f"  [消息] {message.from_agent} → {message.to_agent}: "
              f"{message.content[:30]}...")

    def broadcast(self, from_agent: str, content: str):
        """广播消息"""
        msg = AgentMessage(
            from_agent=from_agent,
            to_agent="all",
            type=MessageType.BROADCAST,
            content=content,
        )
        self._history.append(msg)
        print(f"  [广播] {from_agent} → 所有Agent: {content[:30]}...")

    def get_history(self, agent_name: Optional[str] = None) -> list[AgentMessage]:
        """获取消息历史"""
        if agent_name:
            return [
                m for m in self._history
                if m.from_agent == agent_name or m.to_agent == agent_name
            ]
        return list(self._history)


# ============================================================================
# 第三部分：基础 Agent 类
# ============================================================================

class BaseAgent(ABC):
    """Agent 基类"""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.message_bus: Optional[MessageBus] = None

    def set_message_bus(self, bus: MessageBus):
        """设置消息总线"""
        self.message_bus = bus

    @abstractmethod
    async def process(self, task: str) -> str:
        """处理任务"""
        pass

    def send_message(self, to_agent: str, content: str,
                     msg_type: MessageType = MessageType.RESULT):
        """发送消息"""
        if self.message_bus:
            msg = AgentMessage(
                from_agent=self.name,
                to_agent=to_agent,
                type=msg_type,
                content=content,
            )
            self.message_bus.send(msg)


# ============================================================================
# 第四部分：Supervisor 模式（协调者模式）
# ============================================================================

class SupervisorAgent(BaseAgent):
    """
    Supervisor（协调者）Agent

    负责：
    1. 接收用户任务
    2. 分解任务为子任务
    3. 分配给合适的 Worker Agent
    4. 收集结果并汇总
    """

    def __init__(self, workers: list[BaseAgent]):
        super().__init__(name="Supervisor", role="协调者")
        self.workers = {w.name: w for w in workers}
        for worker in workers:
            worker.set_message_bus(self.message_bus or MessageBus())

    async def process(self, task: str) -> str:
        """处理任务：分解 → 分配 → 收集 → 汇总"""
        print(f"\n{'='*50}")
        print(f"Supervisor 收到任务: {task}")
        print(f"{'='*50}")

        # 1. 分解任务
        subtasks = self._decompose_task(task)
        print(f"\n  分解为 {len(subtasks)} 个子任务:")
        for i, (worker_name, subtask) in enumerate(subtasks, 1):
            print(f"    {i}. [{worker_name}] {subtask}")

        # 2. 分配并执行
        results = {}
        for worker_name, subtask in subtasks:
            if worker_name in self.workers:
                worker = self.workers[worker_name]
                print(f"\n  分配给 {worker_name}...")
                result = await worker.process(subtask)
                results[worker_name] = result
                print(f"  {worker_name} 完成: {result[:30]}...")

        # 3. 汇总结果
        final_result = self._synthesize_results(task, results)
        print(f"\n  最终结果: {final_result[:50]}...")

        return final_result

    def _decompose_task(self, task: str) -> list[tuple[str, str]]:
        """
        分解任务（简化版）

        实际项目中，这里会用 LLM 来智能分解任务。
        """
        # 根据关键词分配给不同的 worker
        subtasks = []

        if "研究" in task or "搜索" in task:
            subtasks.append(("Researcher", f"研究: {task}"))

        if "分析" in task or "数据" in task:
            subtasks.append(("Analyst", f"分析: {task}"))

        if "写" in task or "报告" in task or "总结" in task:
            subtasks.append(("Writer", f"撰写: {task}"))

        # 如果没有匹配，分配给第一个 worker
        if not subtasks and self.workers:
            first_worker = list(self.workers.keys())[0]
            subtasks.append((first_worker, task))

        return subtasks

    def _synthesize_results(self, task: str, results: dict) -> str:
        """汇总结果"""
        if not results:
            return "没有收到任何结果"

        parts = [f"{name}: {result}" for name, result in results.items()]
        return " | ".join(parts)


class ResearcherAgent(BaseAgent):
    """研究员 Agent"""

    def __init__(self):
        super().__init__(name="Researcher", role="信息收集")

    async def process(self, task: str) -> str:
        print(f"    [Researcher] 正在研究: {task[:30]}...")
        await asyncio.sleep(0.1)  # 模拟工作
        return f"收集到关于「{task[:20]}」的相关信息"


class AnalystAgent(BaseAgent):
    """分析师 Agent"""

    def __init__(self):
        super().__init__(name="Analyst", role="数据分析")

    async def process(self, task: str) -> str:
        print(f"    [Analyst] 正在分析: {task[:30]}...")
        await asyncio.sleep(0.1)
        return f"完成对「{task[:20]}」的分析"


class WriterAgent(BaseAgent):
    """写作者 Agent"""

    def __init__(self):
        super().__init__(name="Writer", role="内容撰写")

    async def process(self, task: str) -> str:
        print(f"    [Writer] 正在撰写: {task[:30]}...")
        await asyncio.sleep(0.1)
        return f"完成「{task[:20]}」的撰写"


async def demonstrate_supervisor_pattern():
    """演示 Supervisor 模式"""
    print("=" * 60)
    print("Supervisor 模式演示")
    print("=" * 60)

    # 创建 Worker Agents
    researcher = ResearcherAgent()
    analyst = AnalystAgent()
    writer = WriterAgent()

    # 创建 Supervisor
    supervisor = SupervisorAgent([researcher, analyst, writer])

    # 设置消息总线
    bus = MessageBus()
    supervisor.message_bus = bus
    for worker in [researcher, analyst, writer]:
        worker.set_message_bus(bus)

    # 执行任务
    await supervisor.process("研究 AI Agent 的发展趋势并写一份分析报告")


# ============================================================================
# 第五部分：Plan-and-Execute 模式
# ============================================================================

class PlanStep(BaseModel):
    """计划步骤"""
    id: int
    description: str
    agent: str
    status: str = "pending"  # pending/running/completed/failed
    result: Optional[str] = None


class PlanAndExecuteAgent:
    """
    Plan-and-Execute Agent

    工作流程：
    1. 制定完整计划
    2. 逐步执行计划
    3. 根据执行结果调整计划
    4. 直到所有步骤完成
    """

    def __init__(self, agents: dict[str, BaseAgent]):
        self.agents = agents
        self.plan: list[PlanStep] = []

    async def execute(self, task: str) -> str:
        """执行任务"""
        print(f"\n{'='*50}")
        print(f"Plan-and-Execute: {task}")
        print(f"{'='*50}")

        # 1. 制定计划
        self.plan = self._create_plan(task)
        print(f"\n  计划 ({len(self.plan)} 步):")
        for step in self.plan:
            print(f"    {step.id}. [{step.agent}] {step.description}")

        # 2. 执行计划
        for step in self.plan:
            print(f"\n  执行步骤 {step.id}...")
            step.status = "running"

            if step.agent in self.agents:
                agent = self.agents[step.agent]
                try:
                    result = await agent.process(step.description)
                    step.result = result
                    step.status = "completed"
                    print(f"    ✅ 完成: {result[:30]}...")
                except Exception as e:
                    step.result = str(e)
                    step.status = "failed"
                    print(f"    ❌ 失败: {e}")
            else:
                step.status = "failed"
                step.result = f"未知 Agent: {step.agent}"

        # 3. 汇总结果
        completed = sum(1 for s in self.plan if s.status == "completed")
        total = len(self.plan)
        final = f"任务完成: {completed}/{total} 步成功"
        print(f"\n  {final}")

        return final

    def _create_plan(self, task: str) -> list[PlanStep]:
        """创建计划（简化版）"""
        # 实际项目中用 LLM 生成计划
        return [
            PlanStep(id=1, description=f"第一步: {task}", agent="Researcher"),
            PlanStep(id=2, description=f"第二步: 分析结果", agent="Analyst"),
            PlanStep(id=3, description=f"第三步: 生成报告", agent="Writer"),
        ]


async def demonstrate_plan_and_execute():
    """演示 Plan-and-Execute 模式"""
    print("\n" + "=" * 60)
    print("Plan-and-Execute 模式演示")
    print("=" * 60)

    agents = {
        "Researcher": ResearcherAgent(),
        "Analyst": AnalystAgent(),
        "Writer": WriterAgent(),
    }

    planner = PlanAndExecuteAgent(agents)
    await planner.execute("分析市场趋势并生成报告")


# ============================================================================
# 第六部分：Self-Reflection 模式
# ============================================================================

class SelfReflectingAgent(BaseAgent):
    """
    自我反思 Agent

    工作流程：
    1. 生成初始方案
    2. 自我评估方案质量
    3. 根据评估改进方案
    4. 重复直到满意
    """

    def __init__(self, max_reflections: int = 3):
        super().__init__(name="Reflector", role="自我反思")
        self.max_reflections = max_reflections

    async def process(self, task: str) -> str:
        """处理任务（带自我反思）"""
        print(f"\n{'='*50}")
        print(f"Self-Reflection: {task}")
        print(f"{'='*50}")

        # 1. 生成初始方案
        current_solution = f"初始方案: 针对「{task[:20]}」的解决方案"
        print(f"\n  初始方案: {current_solution}")

        # 2. 自我反思循环
        for i in range(self.max_reflections):
            print(f"\n  ── 第 {i+1} 轮反思 ──")

            # 评估
            evaluation = self._evaluate(current_solution, task)
            print(f"    评估: {evaluation}")

            # 判断是否需要改进
            if "满意" in evaluation or i == self.max_reflections - 1:
                print(f"    结论: 方案已满足要求")
                break

            # 改进
            current_solution = self._improve(current_solution, evaluation)
            print(f"    改进后: {current_solution[:40]}...")

        return current_solution

    def _evaluate(self, solution: str, task: str) -> str:
        """评估方案（模拟）"""
        # 实际项目中用 LLM 评估
        evaluations = [
            "方案基本可行，但缺乏具体细节",
            "方案较好，可以增加更多数据支持",
            "方案满意，可以输出",
        ]
        return evaluations[min(len(solution) % 3, 2)]

    def _improve(self, solution: str, feedback: str) -> str:
        """根据反馈改进方案"""
        return f"{solution} [已根据反馈改进: {feedback[:20]}...]"


async def demonstrate_self_reflection():
    """演示 Self-Reflection 模式"""
    print("\n" + "=" * 60)
    print("Self-Reflection 模式演示")
    print("=" * 60)

    agent = SelfReflectingAgent(max_reflections=3)
    await agent.process("设计一个用户友好的注册流程")


# ============================================================================
# 第七部分：进阶 - Agent 设计模式对比
# ============================================================================

def compare_agent_patterns():
    """对比不同的 Agent 设计模式"""

    print("=" * 60)
    print("Agent 设计模式对比")
    print("=" * 60)

    patterns = [
        {
            "name": "ReAct",
            "structure": "单 Agent + 工具循环",
            "best_for": "通用任务",
            "complexity": "低",
            "token_efficiency": "中",
        },
        {
            "name": "Supervisor",
            "structure": "协调者 + 多个 Worker",
            "best_for": "多领域协作任务",
            "complexity": "中",
            "token_efficiency": "中",
        },
        {
            "name": "Plan-and-Execute",
            "structure": "计划器 + 执行器",
            "best_for": "复杂多步骤任务",
            "complexity": "中",
            "token_efficiency": "高",
        },
        {
            "name": "Self-Reflection",
            "structure": "生成器 + 评估器",
            "best_for": "高质量输出任务",
            "complexity": "中",
            "token_efficiency": "低",
        },
        {
            "name": "Debate",
            "structure": "多个 Agent 辩论",
            "best_for": "决策、评审",
            "complexity": "高",
            "token_efficiency": "低",
        },
        {
            "name": "Hierarchical",
            "structure": "多层 Supervisor",
            "best_for": "超大型任务",
            "complexity": "高",
            "token_efficiency": "中",
        },
    ]

    print(f"\n{'模式':<20} {'结构':<25} {'适合':<20} {'复杂度':<10}")
    print("-" * 75)
    for p in patterns:
        print(f"{p['name']:<20} {p['structure']:<25} {p['best_for']:<20} {p['complexity']:<10}")
    print()


# ============================================================================
# 第八部分：进阶 - 未来展望
# ============================================================================

def future_outlook():
    """Agent 开发的未来展望"""

    print("=" * 60)
    print("Agent 开发的未来展望")
    print("=" * 60)

    trends = [
        {
            "trend": "自主 Agent (Autonomous Agent)",
            "description": "Agent 能够完全自主地完成复杂任务，无需人类干预",
            "challenges": ["安全性", "可控性", "伦理问题"],
        },
        {
            "trend": "多模态 Agent",
            "description": "Agent 能够处理文本、图像、音频、视频等多种模态",
            "challenges": ["模型能力", "计算资源", "模态融合"],
        },
        {
            "trend": "Agent 生态系统",
            "description": "不同 Agent 之间可以互相发现、协作、交易",
            "challenges": ["标准化", "信任机制", "经济模型"],
        },
        {
            "trend": "具身 Agent (Embodied Agent)",
            "description": "Agent 与物理世界交互（机器人、IoT）",
            "challenges": ["实时性", "安全性", "硬件集成"],
        },
        {
            "trend": "个性化 Agent",
            "description": "Agent 能够学习用户偏好，提供个性化服务",
            "challenges": ["隐私保护", "长期记忆", "偏好建模"],
        },
    ]

    for t in trends:
        print(f"\n🔮 {t['trend']}")
        print(f"   描述: {t['description']}")
        print(f"   挑战: {', '.join(t['challenges'])}")
    print()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "" * 30)
    print("第十一课：进阶 - 多 Agent 协作与高级模式")
    print("🔬" * 30 + "\n")

    # 1. 为什么需要多 Agent
    why_multi_agent()

    # 2. Supervisor 模式
    asyncio.run(demonstrate_supervisor_pattern())

    # 3. Plan-and-Execute 模式
    asyncio.run(demonstrate_plan_and_execute())

    # 4. Self-Reflection 模式
    asyncio.run(demonstrate_self_reflection())

    # 5. 模式对比
    compare_agent_patterns()

    # 6. 未来展望
    future_outlook()

    print("=" * 60)
    print("✅ 第十一课学习完成！")
    print("=" * 60)
    print("""
 本课要点总结：
1. 多 Agent 协作可以处理更复杂的任务
2. Supervisor 模式：协调者分配任务给 Worker
3. Plan-and-Execute：先计划后执行，适合多步骤任务
4. Self-Reflection：自我评估和改进，提高输出质量
5. 不同模式有不同的适用场景和复杂度
6. 未来：自主 Agent、多模态、生态系统、具身 Agent

 下一课：总结与进阶路线图
    """)
