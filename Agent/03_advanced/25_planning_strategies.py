"""
==============================================================================
第二十五课（进阶）：Planning Strategies（规划与推理策略）
==============================================================================

【为什么需要单独一课？】
现有课程简单提及 ReAct 和 Plan-and-Execute，但没有深入讲解
各种规划策略的原理、优劣和适用场景。
Planning 是 Agent 的核心能力，直接决定任务完成质量。

【学习目标】
- 掌握 ReAct 策略的完整实现
- 掌握 Plan-and-Execute 策略
- 理解 Tree-of-Thought (ToT) 推理
- 理解 Graph-of-Thought (GoT) 推理
- 掌握 Self-Reflection 自我反思策略
- 学会根据场景选择合适的规划策略

【核心概念】
- ReAct (Reasoning + Acting)
- Plan-and-Execute (先计划后执行)
- Tree-of-Thought (思维树)
- Graph-of-Thought (思维图)
- Self-Reflection (自我反思)
- Strategy Selection (策略选择)

==============================================================================
"""

import asyncio
import json
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================================
# 第一部分：Planning 策略概览
# ============================================================================

def explain_planning_overview():
    """
    Agent 规划策略概览

    Planning 是 Agent 的"执行功能"，决定了如何分解和完成任务。
    不同的策略适用于不同的场景。
    """

    print("=" * 60)
    print("Planning Strategies - 规划策略概览")
    print("=" * 60)

    strategies = [
        {
            "name": "ReAct",
            "principle": "推理和行动交替进行",
            "pros": "灵活、可处理意外情况",
            "cons": "每步都需要 LLM 调用，较慢",
            "best_for": "简单到中等复杂度的任务",
        },
        {
            "name": "Plan-and-Execute",
            "principle": "先制定完整计划，再逐步执行",
            "pros": "全局视角、减少中途偏离",
            "cons": "计划可能不够灵活",
            "best_for": "多步骤、可预测的任务",
        },
        {
            "name": "Tree-of-Thought",
            "principle": "探索多条推理路径，选择最优",
            "pros": "能找到更优解、避免局部最优",
            "cons": "计算成本高",
            "best_for": "需要创造性思维的任务",
        },
        {
            "name": "Graph-of-Thought",
            "principle": "思维以图结构组织，支持合并和循环",
            "pros": "最灵活、支持复杂推理",
            "cons": "实现复杂、成本高",
            "best_for": "复杂推理、需要综合多源信息",
        },
        {
            "name": "Self-Reflection",
            "principle": "执行后自我评估并改进",
            "pros": "持续改进、提高质量",
            "cons": "额外 LLM 调用开销",
            "best_for": "需要高质量输出的任务",
        },
    ]

    print(f"\n{'策略':<20} {'适用场景':<30}")
    print("-" * 55)
    for s in strategies:
        print(f"\n  {s['name']}")
        print(f"    原理: {s['principle']}")
        print(f"    优点: {s['pros']}")
        print(f"    缺点: {s['cons']}")
        print(f"    场景: {s['best_for']}")
    print()


# ============================================================================
# 第二部分：策略基类和共享组件
# ============================================================================

class ThoughtStep:
    """思维步骤"""

    def __init__(self, step_type: str, content: str, score: float = 0.0):
        self.step_type = step_type  # thought/action/observation/reflection
        self.content = content
        self.score = score
        self.timestamp = datetime.now()
        self.children: List["ThoughtStep"] = []

    def __repr__(self):
        return f"[{self.step_type}] {self.content[:50]}"


@dataclass
class PlanningResult:
    """规划结果"""
    task: str
    strategy: str
    steps: List[ThoughtStep] = field(default_factory=list)
    final_answer: str = ""
    total_llm_calls: int = 0
    total_time_ms: float = 0.0
    success: bool = True


class BasePlanningStrategy(ABC):
    """规划策略抽象基类"""

    def __init__(self, name: str):
        self.name = name
        self._llm_call_count = 0

    @abstractmethod
    def execute(self, task: str) -> PlanningResult:
        """执行规划策略"""
        pass

    def _simulate_llm_call(self, prompt: str) -> str:
        """模拟 LLM 调用"""
        self._llm_call_count += 1
        time.sleep(0.05)
        return f"[LLM 回复] 基于输入: {prompt[:40]}..."


# ============================================================================
# 第三部分：ReAct 策略实现
# ============================================================================

class ReActStrategy(BasePlanningStrategy):
    """
    ReAct (Reasoning + Acting) 策略

    【原理】
    交替进行推理(Thought)和行动(Action)：
    1. Thought: 分析当前情况，决定下一步
    2. Action: 调用工具执行操作
    3. Observation: 观察工具返回结果
    4. 重复 1-3 直到任务完成

    【流程图】
    Task -> Thought -> Action -> Observation -> Thought -> ... -> Final Answer
    """

    def __init__(self, max_iterations: int = 5):
        super().__init__("ReAct")
        self.max_iterations = max_iterations

    def execute(self, task: str) -> PlanningResult:
        result = PlanningResult(task=task, strategy=self.name)
        start_time = time.time()

        print(f"\n{'='*50}")
        print(f"ReAct 策略执行: {task}")
        print(f"{'='*50}")

        for i in range(self.max_iterations):
            print(f"\n  --- 迭代 {i+1} ---")

            # 1. Thought (推理)
            thought = self._simulate_llm_call(
                f"分析任务 '{task}'，当前第 {i+1} 步，应该做什么？"
            )
            step = ThoughtStep("thought", thought)
            result.steps.append(step)
            print(f"  Thought: {thought[:60]}...")

            # 2. 判断是否完成
            if i == self.max_iterations - 1 or random.random() < 0.3:
                # 生成最终答案
                answer = self._simulate_llm_call(
                    f"基于之前的推理，给出任务 '{task}' 的最终答案"
                )
                final_step = ThoughtStep("final_answer", answer)
                result.steps.append(final_step)
                result.final_answer = answer
                print(f"  Final Answer: {answer[:60]}...")
                break

            # 3. Action (行动)
            action = self._simulate_llm_call(f"决定调用哪个工具来完成第 {i+1} 步")
            action_step = ThoughtStep("action", action)
            result.steps.append(action_step)
            print(f"  Action: {action[:60]}...")

            # 4. Observation (观察)
            observation = f"工具执行成功，返回结果 #{i+1}"
            obs_step = ThoughtStep("observation", observation)
            result.steps.append(obs_step)
            print(f"  Observation: {observation}")

        result.total_llm_calls = self._llm_call_count
        result.total_time_ms = (time.time() - start_time) * 1000
        return result


# ============================================================================
# 第四部分：Plan-and-Execute 策略
# ============================================================================

class PlanAndExecuteStrategy(BasePlanningStrategy):
    """
    Plan-and-Execute 策略

    【原理】
    分两个阶段：
    1. Planning Phase: 一次性制定完整计划（步骤列表）
    2. Execution Phase: 按计划逐步执行，可动态调整

    【vs ReAct】
    - ReAct: 每步都重新推理（灵活但慢）
    - Plan-and-Execute: 先全局规划再执行（快但不够灵活）
    """

    def __init__(self):
        super().__init__("Plan-and-Execute")

    def execute(self, task: str) -> PlanningResult:
        result = PlanningResult(task=task, strategy=self.name)
        start_time = time.time()

        print(f"\n{'='*50}")
        print(f"Plan-and-Execute 策略执行: {task}")
        print(f"{'='*50}")

        # Phase 1: 制定计划
        print("\n  [Phase 1] 制定计划...")
        plan = self._create_plan(task)
        for i, step in enumerate(plan):
            plan_step = ThoughtStep("plan", step)
            result.steps.append(plan_step)
            print(f"    步骤 {i+1}: {step}")

        # Phase 2: 执行计划
        print("\n  [Phase 2] 执行计划...")
        for i, step in enumerate(plan):
            print(f"\n    --- 执行步骤 {i+1}/{len(plan)} ---")

            # 执行
            exec_result = self._simulate_llm_call(f"执行: {step}")
            exec_step = ThoughtStep("execution", f"{step} -> {exec_result}")
            result.steps.append(exec_step)
            print(f"    结果: {exec_result[:60]}...")

            # 检查是否需要调整计划
            if random.random() < 0.2:
                print(f"    [Re-plan] 检测到需要调整计划...")
                replan = self._simulate_llm_call("根据当前进度调整剩余计划")
                replan_step = ThoughtStep("replan", replan)
                result.steps.append(replan_step)

        # 生成最终答案
        answer = self._simulate_llm_call(f"基于所有执行结果，总结任务 '{task}' 的答案")
        result.final_answer = answer
        result.total_llm_calls = self._llm_call_count
        result.total_time_ms = (time.time() - start_time) * 1000

        print(f"\n  Final Answer: {answer[:60]}...")
        return result

    def _create_plan(self, task: str) -> List[str]:
        """制定计划"""
        plan_response = self._simulate_llm_call(f"为任务 '{task}' 制定详细步骤计划")
        # 模拟返回的步骤列表
        return [
            f"分析任务需求: {task[:30]}",
            "收集相关信息和数据",
            "分析和处理收集到的信息",
            "综合信息形成初步结论",
            "验证结论的准确性",
            "生成最终报告",
        ]


# ============================================================================
# 第五部分：Tree-of-Thought 策略
# ============================================================================

class TreeOfThoughtStrategy(BasePlanningStrategy):
    """
    Tree-of-Thought (思维树) 策略

    【原理】
    不沿着单一路径推理，而是：
    1. 在每一步生成多个候选思路（分支）
    2. 评估每个思路的质量（打分）
    3. 选择得分最高的路径继续
    4. 可以回溯探索其他路径

    【vs ReAct】
    - ReAct: 单路径线性推理
    - ToT: 多路径并行探索，择优前进

    【结构】
             Root
            / | \\
          T1  T2  T3     <- 第一层候选
         /|   |
       T1a T1b T2a       <- 第二层候选
       |
      T1a-1              <- 最优路径
    """

    def __init__(self, branching_factor: int = 3, max_depth: int = 3):
        super().__init__("Tree-of-Thought")
        self.branching_factor = branching_factor
        self.max_depth = max_depth

    def execute(self, task: str) -> PlanningResult:
        result = PlanningResult(task=task, strategy=self.name)
        start_time = time.time()

        print(f"\n{'='*50}")
        print(f"Tree-of-Thought 策略执行: {task}")
        print(f"  分支因子: {self.branching_factor}, 最大深度: {self.max_depth}")
        print(f"{'='*50}")

        # 构建思维树
        root = ThoughtStep("root", task)
        best_path = self._build_tree(root, task, depth=0, result=result)

        # 输出最优路径
        print(f"\n  [最优路径]")
        for i, step in enumerate(best_path):
            print(f"    {i+1}. [{step.step_type}] {step.content[:50]}... (score: {step.score:.2f})")

        result.final_answer = best_path[-1].content if best_path else ""
        result.total_llm_calls = self._llm_call_count
        result.total_time_ms = (time.time() - start_time) * 1000
        return result

    def _build_tree(
        self, node: ThoughtStep, task: str, depth: int, result: PlanningResult
    ) -> List[ThoughtStep]:
        """递归构建思维树，返回最优路径"""

        if depth >= self.max_depth:
            return [node]

        print(f"\n  [Depth {depth}] 生成 {self.branching_factor} 个候选思路...")

        candidates = []
        for i in range(self.branching_factor):
            # 生成候选思路
            thought = self._simulate_llm_call(
                f"任务: {task}, 当前: {node.content[:20]}, 候选思路 #{i+1}"
            )
            # 评估候选思路
            score = random.uniform(0.3, 0.95)
            candidate = ThoughtStep("thought", thought, score)
            candidates.append(candidate)
            print(f"    候选 {i+1}: score={score:.2f} | {thought[:40]}...")

        # 选择最优候选
        best = max(candidates, key=lambda c: c.score)
        print(f"  --> 选择最优: score={best.score:.2f}")

        # 递归扩展最优候选
        best_path = self._build_tree(best, task, depth + 1, result)
        return [node] + best_path


# ============================================================================
# 第六部分：Graph-of-Thought 策略
# ============================================================================

class GraphOfThoughtStrategy(BasePlanningStrategy):
    """
    Graph-of-Thought (思维图) 策略

    【原理】
    思维不再限制为树结构，而是图结构：
    - 节点：思维单元
    - 边：思维之间的关系（依赖、聚合、精炼）
    - 支持：合并（多个思路融合）、循环（迭代改进）

    【vs ToT】
    - ToT: 树结构，只能从上到下
    - GoT: 图结构，支持合并、循环、回溯

    【操作类型】
    - Generate: 生成新的思维节点
    - Transform: 转换/精炼思维
    - Aggregate: 合并多个思维
    - Loop: 循环改进直到满足条件
    """

    def __init__(self):
        super().__init__("Graph-of-Thought")
        self._graph: Dict[str, Dict] = {}

    def execute(self, task: str) -> PlanningResult:
        result = PlanningResult(task=task, strategy=self.name)
        start_time = time.time()

        print(f"\n{'='*50}")
        print(f"Graph-of-Thought 策略执行: {task}")
        print(f"{'='*50}")

        # Phase 1: 生成初始思维节点
        print("\n  [Phase 1] Generate - 生成初始思维")
        nodes = []
        for i in range(3):
            thought = self._simulate_llm_call(f"从多角度分析问题: {task}, 角度 #{i+1}")
            node_id = f"n{i}"
            nodes.append(node_id)
            self._graph[node_id] = {
                "type": "generate",
                "content": thought,
                "score": random.uniform(0.5, 0.8),
                "dependencies": [],
            }
            print(f"    {node_id}: {thought[:50]}...")

        # Phase 2: Transform - 精炼每个思维
        print("\n  [Phase 2] Transform - 精炼思维")
        for node_id in nodes:
            original = self._graph[node_id]["content"]
            refined = self._simulate_llm_call(f"精炼和改进: {original[:30]}")
            refined_id = f"{node_id}_refined"
            self._graph[refined_id] = {
                "type": "transform",
                "content": refined,
                "score": self._graph[node_id]["score"] + 0.1,
                "dependencies": [node_id],
            }
            print(f"    {node_id} -> {refined_id}: 精炼完成")

        # Phase 3: Aggregate - 合并所有精炼后的思维
        print("\n  [Phase 3] Aggregate - 合并思维")
        refined_nodes = [f"{n}_refined" for n in nodes]
        merged = self._simulate_llm_call(
            f"综合以下思路得出最终结论: {', '.join(refined_nodes)}"
        )
        merged_id = "merged"
        self._graph[merged_id] = {
            "type": "aggregate",
            "content": merged,
            "score": 0.92,
            "dependencies": refined_nodes,
        }
        print(f"    合并结果: {merged[:60]}...")

        # Phase 4: Loop - 检查质量，必要时迭代
        print("\n  [Phase 4] Loop - 质量检查")
        quality = self._simulate_llm_call(f"评估当前结果的质量: {merged[:30]}")
        print(f"    质量评估: {quality[:50]}...")
        print(f"    质量得分: 0.88 (通过阈值 0.80)")

        # 构建结果
        for node_id, node_data in self._graph.items():
            step = ThoughtStep(
                node_data["type"],
                node_data["content"],
                node_data["score"],
            )
            result.steps.append(step)

        result.final_answer = merged
        result.total_llm_calls = self._llm_call_count
        result.total_time_ms = (time.time() - start_time) * 1000
        return result


# ============================================================================
# 第七部分：Self-Reflection 策略
# ============================================================================

class SelfReflectionStrategy(BasePlanningStrategy):
    """
    Self-Reflection (自我反思) 策略

    【原理】
    Agent 执行任务后进行自我评估：
    1. Generate: 生成初始答案
    2. Evaluate: 评估答案质量（打分 + 反馈）
    3. Refine: 根据反馈改进答案
    4. 重复 2-3 直到质量达标

    【应用场景】
    - 代码生成 + 自动 review + 修复
    - 文章撰写 + 自我校对 + 修改
    - 数据分析 + 结果验证 + 补充
    """

    def __init__(self, quality_threshold: float = 0.85, max_refinements: int = 3):
        super().__init__("Self-Reflection")
        self.quality_threshold = quality_threshold
        self.max_refinements = max_refinements

    def execute(self, task: str) -> PlanningResult:
        result = PlanningResult(task=task, strategy=self.name)
        start_time = time.time()

        print(f"\n{'='*50}")
        print(f"Self-Reflection 策略执行: {task}")
        print(f"  质量阈值: {self.quality_threshold}, 最大改进次数: {self.max_refinements}")
        print(f"{'='*50}")

        # Step 1: 生成初始答案
        print("\n  [Step 1] 生成初始答案...")
        answer = self._simulate_llm_call(f"回答问题: {task}")
        gen_step = ThoughtStep("generate", answer, 0.0)
        result.steps.append(gen_step)
        print(f"    初始答案: {answer[:60]}...")

        # Step 2-3: 评估和改进循环
        current_answer = answer
        quality = 0.0

        for round_num in range(self.max_refinements):
            print(f"\n  [Round {round_num + 1}] 自我评估...")

            # 评估
            quality = random.uniform(0.55, 0.95)
            feedback = self._simulate_llm_call(
                f"评估答案质量 (当前得分: {quality:.2f}): {current_answer[:30]}"
            )
            eval_step = ThoughtStep("evaluate", feedback, quality)
            result.steps.append(eval_step)
            print(f"    质量得分: {quality:.2f}")
            print(f"    反馈: {feedback[:60]}...")

            if quality >= self.quality_threshold:
                print(f"    --> 质量达标 (>= {self.quality_threshold})，完成!")
                break

            # 改进
            print(f"    --> 质量未达标，进行改进...")
            current_answer = self._simulate_llm_call(
                f"根据反馈改进答案: {feedback[:20]}"
            )
            refine_step = ThoughtStep("refine", current_answer, quality)
            result.steps.append(refine_step)
            print(f"    改进后: {current_answer[:60]}...")

        result.final_answer = current_answer
        result.total_llm_calls = self._llm_call_count
        result.total_time_ms = (time.time() - start_time) * 1000
        return result


# ============================================================================
# 第八部分：策略选择器
# ============================================================================

class PlanningStrategySelector:
    """
    策略选择器

    根据任务特征自动选择最合适的规划策略。

    选择规则：
    - 简单任务 -> ReAct
    - 多步骤可预测 -> Plan-and-Execute
    - 需要创造性 -> Tree-of-Thought
    - 复杂综合推理 -> Graph-of-Thought
    - 需要高质量输出 -> Self-Reflection
    """

    TASK_COMPLEXITY_KEYWORDS = {
        "simple": ["查询", "计算", "翻译", "简单"],
        "moderate": ["分析", "比较", "总结", "报告"],
        "complex": ["设计", "创新", "优化", "综合", "多维度"],
    }

    @classmethod
    def select(cls, task: str) -> str:
        """根据任务选择策略"""
        task_lower = task.lower()

        # 检查复杂度
        for level, keywords in cls.TASK_COMPLEXITY_KEYWORDS.items():
            if any(kw in task_lower for kw in keywords):
                if level == "simple":
                    return "ReAct"
                elif level == "moderate":
                    return "Plan-and-Execute"
                else:
                    return "Tree-of-Thought"

        return "ReAct"  # 默认

    @classmethod
    def demonstrate(cls):
        """演示策略选择"""
        print("=" * 60)
        print("策略选择器演示")
        print("=" * 60)

        tasks = [
            "查询北京今天的天气",
            "分析三个竞品的优缺点并生成报告",
            "设计一个创新的推荐系统架构",
            "翻译这段话为英文",
            "综合多个数据源，设计最优解决方案",
        ]

        for task in tasks:
            selected = cls.select(task)
            print(f"\n  任务: {task}")
            print(f"  选择策略: {selected}")
        print()


# ============================================================================
# 第九部分：完整演示
# ============================================================================

def run_all_strategies():
    """运行所有策略的对比演示"""

    task = "分析公司Q3销售数据，找出增长最快的产品线并给出建议"

    strategies = [
        ReActStrategy(max_iterations=3),
        PlanAndExecuteStrategy(),
        TreeOfThoughtStrategy(branching_factor=3, max_depth=2),
        GraphOfThoughtStrategy(),
        SelfReflectionStrategy(quality_threshold=0.80, max_refinements=2),
    ]

    results = []
    for strategy in strategies:
        result = strategy.execute(task)
        results.append(result)

    # 对比总结
    print(f"\n\n{'='*60}")
    print("策略对比总结")
    print(f"{'='*60}")
    print(f"{'策略':<20} {'LLM调用':<10} {'耗时(ms)':<12} {'步骤数':<8}")
    print("-" * 55)
    for r in results:
        print(f"{r.strategy:<20} {r.total_llm_calls:<10} {r.total_time_ms:<12.0f} {len(r.steps):<8}")
    print()


def main():
    """主函数"""
    print("=" * 60)
    print("第25课: Planning Strategies (规划与推理策略)")
    print("=" * 60)

    # 1. 概览
    explain_planning_overview()

    # 2. 策略选择器
    PlanningStrategySelector.demonstrate()

    # 3. 运行所有策略对比
    run_all_strategies()

    # 总结
    print("=" * 60)
    print("课程总结")
    print("=" * 60)
    print("""
  本课介绍了 5 种 Agent 规划策略：

  1. ReAct: 推理+行动交替，灵活但较慢
  2. Plan-and-Execute: 先计划后执行，全局但不够灵活
  3. Tree-of-Thought: 多路径探索，找到更优解
  4. Graph-of-Thought: 图结构推理，最灵活但最复杂
  5. Self-Reflection: 自我评估改进，提高输出质量

  选择建议：
  - 简单任务 -> ReAct
  - 多步骤任务 -> Plan-and-Execute
  - 创造性任务 -> Tree-of-Thought
  - 复杂推理 -> Graph-of-Thought
  - 高质量要求 -> Self-Reflection
""")


if __name__ == "__main__":
    main()
