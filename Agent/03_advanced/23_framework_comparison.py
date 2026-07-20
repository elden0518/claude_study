"""
==============================================================================
第二十三课（补充）：主流 Agent 框架实战对比
==============================================================================

【为什么需要单独一课？】
现有课程只简单提到框架名称，没有深入对比和实战演示。
了解主流框架的特点和适用场景，有助于做出技术选型。

【学习目标】
- 掌握 LangGraph 的核心概念和使用
- 掌握 CrewAI 的角色扮演协作模式
- 掌握 AutoGen 的多 Agent 对话模式
- 学会根据场景选择合适的框架
- 理解各框架的优劣势

【核心概念】
- LangGraph（图状态机）
- CrewAI（角色扮演）
- AutoGen（多 Agent 对话）
- Framework Selection（框架选型）

==============================================================================
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# 第一部分：LangGraph 风格 - 图状态机
# ============================================================================

class LangGraphState:
    """
    LangGraph 状态

    【原理】
    LangGraph 使用图结构定义 Agent 工作流：
    - 节点（Node）：执行具体操作
    - 边（Edge）：定义转换关系
    - 状态（State）：在节点间传递的数据
    """

    def __init__(self):
        self.data: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        self.data[key] = value

    def update(self, updates: Dict[str, Any]):
        self.data.update(updates)


class LangGraphNode:
    """LangGraph 节点"""

    def __init__(self, name: str, handler: Callable[[LangGraphState], Dict]):
        self.name = name
        self.handler = handler

    def execute(self, state: LangGraphState) -> Dict:
        """执行节点逻辑"""
        print(f"    [Node: {self.name}]")
        updates = self.handler(state)
        state.update(updates)
        return updates


class LangGraphEdge:
    """LangGraph 边"""

    def __init__(self, from_node: str, to_node: str, condition: Callable = None):
        self.from_node = from_node
        self.to_node = to_node
        self.condition = condition  # 条件边

    def should_traverse(self, state: LangGraphState) -> bool:
        """是否应该遍历这条边"""
        if self.condition:
            return self.condition(state)
        return True


class SimpleLangGraph:
    """
    简化版 LangGraph

    【特点】
    - 图结构定义工作流
    - 条件路由
    - 状态持久化
    - 支持循环
    """

    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, LangGraphNode] = {}
        self.edges: List[LangGraphEdge] = []
        self.entry_point: Optional[str] = None

    def add_node(self, name: str, handler: Callable) -> "SimpleLangGraph":
        """添加节点"""
        self.nodes[name] = LangGraphNode(name, handler)
        return self

    def add_edge(self, from_node: str, to_node: str,
                 condition: Callable = None) -> "SimpleLangGraph":
        """添加边"""
        self.edges.append(LangGraphEdge(from_node, to_node, condition))
        return self

    def set_entry(self, node_name: str) -> "SimpleLangGraph":
        """设置入口节点"""
        self.entry_point = node_name
        return self

    def run(self, initial_state: Dict = None, max_steps: int = 10) -> LangGraphState:
        """运行图"""
        print(f"\n{'='*50}")
        print(f"LangGraph 运行: {self.name}")
        print(f"{'='*50}")

        state = LangGraphState()
        if initial_state:
            state.update(initial_state)

        current_node = self.entry_point
        steps = 0

        while current_node and steps < max_steps:
            steps += 1
            node = self.nodes.get(current_node)
            if not node:
                break

            print(f"\n  Step {steps}: 执行节点 [{current_node}]")
            node.execute(state)

            # 找下一条边
            next_node = None
            for edge in self.edges:
                if edge.from_node == current_node and edge.should_traverse(state):
                    next_node = edge.to_node
                    break

            current_node = next_node

        print(f"\n  运行完成，共 {steps} 步")
        return state


# ============================================================================
# 第二部分：CrewAI 风格 - 角色扮演协作
# ============================================================================

class CrewRole(Enum):
    """CrewAI 角色"""
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"
    MANAGER = "manager"


@dataclass
class CrewAgent:
    """CrewAI Agent"""
    name: str
    role: CrewRole
    goal: str
    backstory: str
    tools: List[str] = field(default_factory=list)

    def work(self, task: str) -> str:
        """执行任务"""
        print(f"    [{self.name}] ({self.role.value}) 正在处理: {task[:30]}...")
        # 模拟工作
        return f"[{self.name} 的产出] 基于角色 {self.role.value} 完成: {task}"


@dataclass
class CrewTask:
    """CrewAI 任务"""
    description: str
    assigned_agent: CrewAgent
    expected_output: str = ""


class Crew:
    """
    CrewAI 风格协作

    【特点】
    - 每个 Agent 有明确的角色和背景
    - 任务分配给特定角色的 Agent
    - 支持顺序执行和层级执行
    """

    def __init__(self, agents: List[CrewAgent], process: str = "sequential"):
        self.agents = agents
        self.process = process  # sequential / hierarchical
        self.tasks: List[CrewTask] = []

    def add_task(self, description: str, agent: CrewAgent,
                 expected_output: str = "") -> "Crew":
        """添加任务"""
        self.tasks.append(CrewTask(description, agent, expected_output))
        return self

    def run(self) -> List[str]:
        """执行所有任务"""
        print(f"\n{'='*50}")
        print(f"CrewAI 执行流程 ({self.process})")
        print(f"{'='*50}")

        results = []

        if self.process == "sequential":
            for i, task in enumerate(self.tasks, 1):
                print(f"\n  Task {i}: {task.description}")
                result = task.assigned_agent.work(task.description)
                results.append(result)
                print(f"    -> 完成")

        elif self.process == "hierarchical":
            # 层级执行：Manager 分配任务
            manager = next((a for a in self.agents if a.role == CrewRole.MANAGER), None)
            if manager:
                print(f"\n  [Manager] {manager.name} 分配任务...")
                for task in self.tasks:
                    print(f"\n  Task: {task.description}")
                    result = task.assigned_agent.work(task.description)
                    results.append(result)

        print(f"\n  所有任务完成，共 {len(results)} 个结果")
        return results


# ============================================================================
# 第三部分：AutoGen 风格 - 多 Agent 对话
# ============================================================================

@dataclass
class AutoGenAgent:
    """AutoGen Agent"""
    name: str
    system_message: str
    max_turns: int = 5

    def reply(self, message: str, sender: str = None) -> str:
        """回复消息"""
        print(f"    [{self.name}] 收到来自 {sender or 'user'}: {message[:30]}...")
        return f"[{self.name} 回复] 关于「{message[:20]}」的看法..."


class GroupChat:
    """
    AutoGen 群聊

    【特点】
    - 多个 Agent 在同一个聊天室
    - 轮流发言
    - 可以指定发言顺序或自动选择
    """

    def __init__(self, agents: List[AutoGenAgent], max_rounds: int = 5):
        self.agents = agents
        self.max_rounds = max_rounds
        self.messages: List[Dict[str, str]] = []

    def run(self, initial_message: str) -> List[Dict]:
        """运行群聊"""
        print(f"\n{'='*50}")
        print(f"AutoGen 群聊开始")
        print(f"{'='*50}")

        self.messages.append({"sender": "user", "content": initial_message})
        print(f"\n  [User]: {initial_message}")

        for round_num in range(self.max_rounds):
            print(f"\n  --- Round {round_num + 1} ---")

            for agent in self.agents:
                # 获取最新消息
                last_message = self.messages[-1]["content"] if self.messages else ""
                sender = self.messages[-1]["sender"] if self.messages else "user"

                # Agent 回复
                reply = agent.reply(last_message, sender)
                self.messages.append({"sender": agent.name, "content": reply})
                print(f"    [{agent.name}]: {reply}")

                # 检查是否结束
                if "TERMINATE" in reply:
                    print(f"\n  群聊结束（Agent 提议终止）")
                    return self.messages

        return self.messages


# ============================================================================
# 第四部分：框架对比
# ============================================================================

def compare_frameworks():
    """框架对比"""

    print("=" * 60)
    print("主流 Agent 框架对比")
    print("=" * 60)

    frameworks = [
        {
            "name": "LangGraph",
            "paradigm": "图状态机",
            "strengths": [
                "精确控制工作流",
                "支持复杂条件路由",
                "内置状态持久化",
                "可视化工作流",
            ],
            "weaknesses": [
                "学习曲线较陡",
                "需要预定义工作流",
                "灵活性受限于图结构",
            ],
            "use_cases": [
                "需要精确控制流程的场景",
                "复杂的多步骤任务",
                "需要可视化的工作流",
            ],
        },
        {
            "name": "CrewAI",
            "paradigm": "角色扮演协作",
            "strengths": [
                "直观的角色定义",
                "自然的任务分配",
                "支持顺序/层级流程",
                "易于理解和维护",
            ],
            "weaknesses": [
                "角色交互较简单",
                "不适合复杂工作流",
                "调试困难",
            ],
            "use_cases": [
                "内容创作团队",
                "研究分析团队",
                "多角色协作场景",
            ],
        },
        {
            "name": "AutoGen",
            "paradigm": "多 Agent 对话",
            "strengths": [
                "灵活的对话模式",
                "支持群聊",
                "Agent 间自由交流",
                "代码执行能力",
            ],
            "weaknesses": [
                "对话可能发散",
                "难以控制流程",
                "Token 消耗大",
            ],
            "use_cases": [
                "头脑风暴",
                "代码审查",
                "多视角讨论",
            ],
        },
    ]

    for fw in frameworks:
        print(f"\n  [{fw['name']}] - {fw['paradigm']}")
        print(f"    优势:")
        for s in fw["strengths"]:
            print(f"      + {s}")
        print(f"    劣势:")
        for w in fw["weaknesses"]:
            print(f"      - {w}")
        print(f"    适用场景:")
        for u in fw["use_cases"]:
            print(f"      * {u}")

    # 选型决策树
    print(f"\n{'='*60}")
    print("框架选型决策树")
    print(f"{'='*60}")
    print("""
  需要精确控制工作流？
  ├── 是 -> LangGraph
  └── 否 -> 需要多角色协作？
            ├── 是 -> CrewAI
            └── 否 -> 需要多视角讨论？
                      ├── 是 -> AutoGen
                      └── 否 -> 单 Agent 即可（不需要框架）
    """)


# ============================================================================
# 第五部分：完整示例
# ============================================================================

def demo_frameworks():
    """演示各框架"""

    # 1. LangGraph 演示
    print("\n" + "=" * 60)
    print("LangGraph 演示: 简单研究流程")
    print("=" * 60)

    graph = SimpleLangGraph("Research Flow")

    def research_node(state: LangGraphState) -> Dict:
        topic = state.get("topic", "unknown")
        return {"research_result": f"关于 {topic} 的研究数据"}

    def write_node(state: LangGraphState) -> Dict:
        research = state.get("research_result", "")
        return {"article": f"基于研究撰写的文章: {research[:30]}..."}

    def review_node(state: LangGraphState) -> Dict:
        article = state.get("article", "")
        return {"final": f"审核通过的文章: {article[:30]}..."}

    graph.add_node("research", research_node)
    graph.add_node("write", write_node)
    graph.add_node("review", review_node)
    graph.add_edge("research", "write")
    graph.add_edge("write", "review")
    graph.set_entry("research")

    result = graph.run({"topic": "AI Agent 发展趋势"})
    print(f"\n  最终结果: {result.get('final')}")

    # 2. CrewAI 演示
    print("\n" + "=" * 60)
    print("CrewAI 演示: 内容创作团队")
    print("=" * 60)

    researcher = CrewAgent(
        name="Alice",
        role=CrewRole.RESEARCHER,
        goal="收集和分析信息",
        backstory="资深研究员，擅长信息搜集"
    )

    writer = CrewAgent(
        name="Bob",
        role=CrewRole.WRITER,
        goal="撰写高质量内容",
        backstory="专业写手，文笔流畅"
    )

    reviewer = CrewAgent(
        name="Charlie",
        role=CrewRole.REVIEWER,
        goal="确保内容质量",
        backstory="资深编辑，注重细节"
    )

    crew = Crew([researcher, writer, reviewer], process="sequential")
    crew.add_task("研究 AI Agent 最新进展", researcher)
    crew.add_task("撰写技术博客", writer)
    crew.add_task("审核并发布", reviewer)

    results = crew.run()

    # 3. AutoGen 演示
    print("\n" + "=" * 60)
    print("AutoGen 演示: 技术讨论组")
    print("=" * 60)

    agent1 = AutoGenAgent(
        name="技术专家",
        system_message="你是一个技术专家，擅长架构设计"
    )

    agent2 = AutoGenAgent(
        name="产品经理",
        system_message="你是一个产品经理，关注用户体验"
    )

    agent3 = AutoGenAgent(
        name="测试工程师",
        system_message="你是一个测试工程师，关注质量保障"
    )

    group = GroupChat([agent1, agent2, agent3], max_rounds=2)
    messages = group.run("讨论如何设计一个高可用的 Agent 系统")

    print(f"\n  群聊消息总数: {len(messages)}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("主流 Agent 框架实战对比")
    print("=" * 60)

    compare_frameworks()
    demo_frameworks()

    print("\n" + "=" * 60)
    print("课程总结")
    print("=" * 60)
    print("""
  本课对比了三个主流 Agent 框架：

  核心知识点：
  1. LangGraph: 图状态机，精确控制工作流
  2. CrewAI: 角色扮演，自然的团队协作
  3. AutoGen: 多 Agent 对话，灵活讨论

  选型建议：
  - 需要精确控制 -> LangGraph
  - 需要角色协作 -> CrewAI
  - 需要多视角讨论 -> AutoGen
  - 简单任务 -> 不需要框架

  共同特点：
  - 都支持多 Agent 协作
  - 都提供工具调用能力
  - 都有状态管理机制
    """)
