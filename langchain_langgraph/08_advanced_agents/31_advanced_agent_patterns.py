"""
主题：Agent 高级模式 —— Plan-and-Execute、Self-Reflection、ReWOO

学习目标：
  1. 理解 ReAct 的局限性（串行执行，效率低）
  2. 掌握 Plan-and-Execute（先规划后执行，适合复杂任务）
  3. 掌握 Self-Reflection（自我反思，持续改进）
  4. 掌握 ReWOO（减少 LLM 调用次数，降低成本）
  5. 学会根据场景选择合适的 Agent 模式

核心概念：
  ReAct = Reason + Act（交替进行，灵活但慢）
  Plan-and-Execute = Plan → Execute（先想清楚再做，结构化强）
  Self-Reflection = Act → Reflect → Improve（迭代优化）
  ReWOO = Reason + Observe without Observation（离线推理，高效）

前置知识：已完成 02_lc_advanced/11_tools_agents.py
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=512)


# =============================================================================
# Part 1：Plan-and-Execute Agent
# =============================================================================

def demo_plan_and_execute():
    """
    Plan-and-Execute 模式将任务分为两个阶段：
    
    Phase 1: Planning（规划）
    - 分析任务目标
    - 制定详细步骤计划
    - 确定所需工具和资源
    
    Phase 2: Execution（执行）
    - 按顺序执行每个步骤
    - 收集中间结果
    - 整合最终答案
    
    优势：
    ✅ 结构化强，易于调试
    ✅ 可并行执行独立步骤
    ✅ 计划可人工审核和修改
    
    劣势：
    ❌ 不够灵活，无法动态调整
    ❌ 计划错误会导致全盘失败
    """
    print("=" * 60)
    print("Part 1: Plan-and-Execute Agent")
    print("=" * 60)
    
    class PlanExecuteState(TypedDict):
        messages: Annotated[list, add_messages]
        task: str
        plan: List[str]       # 执行计划（步骤列表）
        step_results: List[str]  # 每步执行结果
        current_step: int
        final_answer: str
    
    # 规划器节点
    def planner_node(state: PlanExecuteState) -> dict:
        """生成执行计划"""
        print(f"\n  [Planner] 正在规划任务：{state['task']}")
        
        planner_prompt = ChatPromptTemplate.from_template(
            """你是一个任务规划专家。请将以下任务分解为具体的执行步骤。

任务：{task}

要求：
1. 每步应该清晰具体
2. 步骤之间逻辑连贯
3. 估计总步数（3-7 步）

输出格式（每行一个步骤）：
步骤1: ...
步骤2: ...
步骤3: ..."""
        )
        
        chain = planner_prompt | llm | StrOutputParser()
        plan_text = chain.invoke({"task": state["task"]})
        
        # 解析步骤
        steps = []
        for line in plan_text.strip().split("\n"):
            if "步骤" in line and ":" in line:
                step = line.split(":", 1)[1].strip()
                if step:
                    steps.append(step)
        
        print(f"  [Planner] 生成 {len(steps)} 个步骤")
        for i, step in enumerate(steps, 1):
            print(f"    {i}. {step}")
        
        return {"plan": steps, "current_step": 0}
    
    # 执行器节点
    def executor_node(state: PlanExecuteState) -> dict:
        """执行当前步骤"""
        current_step_idx = state["current_step"]
        
        if current_step_idx >= len(state["plan"]):
            print(f"\n  [Executor] 所有步骤已完成")
            return {}
        
        current_step = state["plan"][current_step_idx]
        print(f"\n  [Executor] 执行步骤 {current_step_idx + 1}: {current_step}")
        
        # 模拟执行（实际应调用工具或 LLM）
        execution_prompt = ChatPromptTemplate.from_template(
            """执行以下步骤：

任务背景：{task}
当前步骤：{step}
之前结果：{previous_results}

执行结果："""
        )
        
        chain = execution_prompt | llm | StrOutputParser()
        result = chain.invoke({
            "task": state["task"],
            "step": current_step,
            "previous_results": "; ".join(state["step_results"][-2:]) if state["step_results"] else "无"
        })
        
        print(f"  [Executor] 结果：{result[:100]}...")
        
        return {
            "step_results": state["step_results"] + [result],
            "current_step": current_step_idx + 1
        }
    
    # 路由函数：决定继续执行还是结束
    def should_continue(state: PlanExecuteState) -> str:
        if state["current_step"] < len(state["plan"]):
            return "execute"
        return "finalize"
    
    # 最终化节点
    def finalize_node(state: PlanExecuteState) -> dict:
        """整合所有步骤结果，生成最终答案"""
        print(f"\n  [Finalizer] 整合结果...")
        
        synthesize_prompt = ChatPromptTemplate.from_template(
            """基于以下执行步骤和结果，给出最终答案。

任务：{task}

执行步骤及结果：
{step_details}

最终答案："""
        )
        
        step_details = "\n".join([
            f"步骤{i+1}: {plan}\n结果: {result}"
            for i, (plan, result) in enumerate(zip(state["plan"], state["step_results"]))
        ])
        
        chain = synthesize_prompt | llm | StrOutputParser()
        final_answer = chain.invoke({
            "task": state["task"],
            "step_details": step_details
        })
        
        print(f"  [Finalizer] 完成")
        return {"final_answer": final_answer}
    
    # 构建图
    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("finalize", finalize_node)
    
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", should_continue, {
        "execute": "executor",
        "finalize": "finalize"
    })
    graph.add_edge("finalize", END)
    
    app = graph.compile()
    
    # 测试
    task = "分析 Python 语言的主要特点和应用场景"
    print(f"\n【任务】{task}")
    
    result = app.invoke({
        "messages": [HumanMessage(content=task)],
        "task": task,
        "plan": [],
        "step_results": [],
        "current_step": 0,
        "final_answer": ""
    })
    
    print(f"\n【最终答案】")
    print(result["final_answer"][:200])


# =============================================================================
# Part 2：Self-Reflection Agent
# =============================================================================

def demo_self_reflection():
    """
    Self-Reflection 模式让 Agent 能够自我评估和改进。
    
    工作流程：
    1. 生成初始答案
    2. 批判性评估答案质量
    3. 识别问题和改进点
    4. 生成改进版本
    5. 重复直到满意或达到最大迭代次数
    
    适用场景：
    - 代码生成（需要多次调试）
    - 创意写作（需要润色）
    - 复杂推理（需要验证）
    """
    print("\n" + "=" * 60)
    print("Part 2: Self-Reflection Agent")
    print("=" * 60)
    
    class ReflectionState(TypedDict):
        question: str
        draft: str           # 草稿答案
        critique: str        # 批评意见
        improved: str        # 改进版本
        iteration: int
        quality_score: float
    
    # 生成器节点
    def generator_node(state: ReflectionState) -> dict:
        """生成答案草稿"""
        print(f"\n  [Generator] 第 {state['iteration'] + 1} 轮生成")
        
        gen_prompt = ChatPromptTemplate.from_template(
            """回答问题：

问题：{question}

{'之前版本的批评：' + state['critique'] if state.get('critique') else ''}

请生成一个高质量的答案："""
        )
        
        chain = gen_prompt | llm | StrOutputParser()
        draft = chain.invoke({
            "question": state["question"],
            "critique": state.get("critique", "")
        })
        
        print(f"  [Generator] 草稿长度：{len(draft)} 字符")
        return {"draft": draft}
    
    # 反思器节点
    def critic_node(state: ReflectionState) -> dict:
        """评估答案质量并提供改进建议"""
        print(f"\n  [Critic] 评估答案质量")
        
        critic_prompt = ChatPromptTemplate.from_template(
            """你是一位严格的评审专家。请评估以下答案的质量。

问题：{question}
答案：{answer}

评估维度：
1. 准确性（是否有事实错误？）
2. 完整性（是否遗漏重要信息？）
3. 清晰度（表达是否清楚？）
4. 相关性（是否紧扣问题？）

请给出：
- 质量评分（0-10 分）
- 具体问题列表
- 改进建议"""
        )
        
        chain = critic_prompt | llm | StrOutputParser()
        critique = chain.invoke({
            "question": state["question"],
            "answer": state["draft"]
        })
        
        print(f"  [Critic] 评估完成")
        print(f"  评审意见：{critique[:150]}...")
        
        # 提取分数（简化处理）
        score = 7.0  # 默认分数，实际可用正则提取
        
        return {"critique": critique, "quality_score": score}
    
    # 决策函数：是否需要继续改进
    def should_improve(state: ReflectionState) -> str:
        max_iterations = 3
        min_quality = 8.0
        
        if state["iteration"] >= max_iterations:
            print(f"\n  [Decision] 达到最大迭代次数 ({max_iterations})")
            return "finish"
        
        if state["quality_score"] >= min_quality:
            print(f"\n  [Decision] 质量达标 ({state['quality_score']:.1f} >= {min_quality})")
            return "finish"
        
        print(f"\n  [Decision] 继续改进 (分数 {state['quality_score']:.1f} < {min_quality})")
        return "improve"
    
    # 构建图
    graph = StateGraph(ReflectionState)
    graph.add_node("generator", generator_node)
    graph.add_node("critic", critic_node)
    
    graph.add_edge(START, "generator")
    graph.add_edge("generator", "critic")
    graph.add_conditional_edges("critic", should_improve, {
        "improve": "generator",
        "finish": END
    })
    
    app = graph.compile()
    
    # 测试
    question = "解释量子计算的基本原理"
    print(f"\n【问题】{question}")
    
    result = app.invoke({
        "question": question,
        "draft": "",
        "critique": "",
        "improved": "",
        "iteration": 0,
        "quality_score": 0.0
    })
    
    print(f"\n【最终答案】")
    print(result["draft"][:200])
    print(f"\n【最终评分】{result['quality_score']:.1f}/10")


# =============================================================================
# Part 3：ReWOO（Reasoning Without Observation）
# =============================================================================

def demo_rewoo():
    """
    ReWOO 的核心思想：减少 LLM 调用次数，提高效率。
    
    传统 ReAct 的问题：
    - 每次工具调用都需要 LLM 参与（Think → Act → Observe）
    - 大量 token 浪费在重复的上下文上
    - 延迟高（多次往返 LLM）
    
    ReWOO 的改进：
    1. 一次性生成完整计划（包含所有工具调用）
    2. 批量执行所有工具（无需 LLM 等待）
    3. 最后用 LLM 整合结果
    
    优势：
    ✅ LLM 调用次数减少 50-70%
    ✅ 工具可并行执行
    ✅ 延迟显著降低
    
    劣势：
    ❌ 无法根据中间结果动态调整
    ❌ 计划错误难以纠正
    """
    print("\n" + "=" * 60)
    print("Part 3: ReWOO (Reasoning Without Observation)")
    print("=" * 60)
    
    print("""
  📊 ReWOO vs ReAct 对比：
  ──────────────────────────────────────────────
  
  ReAct 流程（5 次 LLM 调用）：
  ┌────────┐    ┌──────┐    ┌────────┐    ┌──────┐    ┌────────┐
  │ Think 1│ →  │ Act 1│ →  │ Think 2│ →  │ Act 2│ →  │ Answer │
  └────────┘    └──────┘    └────────┘    └──────┘    └────────┘
                 ↑              ↑
            等待观察1       等待观察2
  
  ReWOO 流程（2 次 LLM 调用）：
  ┌──────────────┐    ┌──────────────┐    ┌────────┐
  │ Plan All     │ →  │ Execute All  │ →  │ Answer │
  │ (Think 1+2)  │    │ (Act 1+2)    │    │        │
  └──────────────┘    └──────────────┘    └────────┘
                       ↑
                  并行执行，无需等待
  
  ⚡ 性能提升：
  • LLM 调用：5 次 → 2 次（减少 60%）
  • 延迟：~10s → ~4s（降低 60%）
  • Token 消耗：~2000 → ~800（节省 60%）
    """)
    
    # ReWOO 实现示例
    class ReWOOState(TypedDict):
        task: str
        plan: str          # 完整计划（含工具调用）
        observations: List[str]  # 所有工具执行结果
        answer: str
    
    def planner_node_rewoo(state: ReWOOState) -> dict:
        """一次性生成完整计划"""
        print(f"\n  [Planner] 生成完整计划")
        
        plan_prompt = ChatPromptTemplate.from_template(
            """为以下任务制定完整执行计划。

任务：{task}

可用工具：
- search_web: 搜索网络信息
- calculate: 数学计算
- get_date: 获取当前日期

输出格式：
#Plan: 步骤描述
#E1: search_web[查询内容]
#E2: calculate[表达式]
...

计划："""
        )
        
        chain = plan_prompt | llm | StrOutputParser()
        plan = chain.invoke({"task": state["task"]})
        
        print(f"  [Planner] 计划生成完成")
        print(f"  {plan[:200]}...")
        
        return {"plan": plan}
    
    def executor_node_rewoo(state: ReWOOState) -> dict:
        """解析计划并执行所有工具"""
        print(f"\n  [Executor] 执行所有工具调用")
        
        # 简化工具模拟
        observations = [
            "[Observation 1] 搜索结果...",
            "[Observation 2] 计算结果: 42",
        ]
        
        print(f"  [Executor] 执行完成，获得 {len(observations)} 个结果")
        return {"observations": observations}
    
    def answer_node_rewoo(state: ReWOOState) -> dict:
        """整合所有观察结果，生成最终答案"""
        print(f"\n  [Answer] 整合结果")
        
        answer_prompt = ChatPromptTemplate.from_template(
            """基于以下计划和观察结果，回答原始任务。

任务：{task}
计划：{plan}
观察结果：{observations}

最终答案："""
        )
        
        chain = answer_prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "task": state["task"],
            "plan": state["plan"],
            "observations": "\n".join(state["observations"])
        })
        
        return {"answer": answer}
    
    # 构建图
    graph = StateGraph(ReWOOState)
    graph.add_node("planner", planner_node_rewoo)
    graph.add_node("executor", executor_node_rewoo)
    graph.add_node("answer", answer_node_rewoo)
    
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "answer")
    graph.add_edge("answer", END)
    
    app = graph.compile()
    
    # 测试
    task = "计算 2024 年有多少天，并搜索闰年的定义"
    print(f"\n【任务】{task}")
    
    result = app.invoke({
        "task": task,
        "plan": "",
        "observations": [],
        "answer": ""
    })
    
    print(f"\n【最终答案】")
    print(result["answer"][:200])


# =============================================================================
# Part 4：模式选择指南
# =============================================================================

def demo_pattern_selection_guide():
    """
    如何根据场景选择合适的 Agent 模式。
    """
    print("\n" + "=" * 60)
    print("Part 4: Pattern Selection Guide")
    print("=" * 60)
    
    guide = """
  🎯 Agent 模式选择决策树：
  ──────────────────────────────────────────────
  
  Q1: 任务是否需要调用工具？
  ├─ No  → 直接使用 LLM（无需 Agent）
  └─ Yes → Q2
  
  Q2: 任务复杂度如何？
  ├─ 简单（单步工具调用）
  │   └─ → ReAct（灵活、易实现）
  │
  ├─ 中等（多步但有明确流程）
  │   └─ → Plan-and-Execute（结构化强）
  │
  └─ 复杂（需要迭代优化）
      └─ → Q3
  
  Q3: 是否需要自我改进？
  ├─ Yes → Self-Reflection（持续优化质量）
  └─ No  → Q4
  
  Q4: 对延迟和成本敏感吗？
  ├─ Yes → ReWOO（高效、低成本）
  └─ No  → ReAct（灵活性优先）
  
  📋 场景推荐：
  ──────────────────────────────────────────────
  
  客服机器人：
  ✓ ReAct（需要灵活应对各种问题）
  ✗ Plan-and-Execute（对话不可预测）
  
  数据分析报告：
  ✓ Plan-and-Execute（步骤明确：加载→清洗→分析→可视化）
  ✓ ReWOO（多个独立分析可并行）
  
  代码生成：
  ✓ Self-Reflection（需要测试和调试）
  ✗ ReWOO（需要根据错误动态调整）
  
  研究助手：
  ✓ ReAct（探索性强，路径不确定）
  ✓ Self-Reflection（需要验证事实准确性）
  
  批量文档处理：
  ✓ ReWOO（大量相似任务，追求效率）
  ✗ Self-Reflection（不需要迭代优化）
  
  ⚖️ 权衡因素：
  ──────────────────────────────────────────────
  
  延迟要求：
  • < 2s  → ReWOO
  • 2-5s  → Plan-and-Execute
  • > 5s  → ReAct / Self-Reflection
  
  成本预算：
  • 低    → ReWOO（LLM 调用少）
  • 中    → Plan-and-Execute
  • 高    → Self-Reflection（多次迭代）
  
  准确性要求：
  • 高    → Self-Reflection（自我验证）
  • 中    → Plan-and-Execute
  • 低    → ReWOO
  
  开发难度：
  • 简单  → ReAct（LangChain 原生支持）
  • 中等  → Plan-and-Execute / ReWOO
  • 复杂  → Self-Reflection（需设计评估标准）
    """
    print(guide)


def main():
    print("=" * 60)
    print("Agent 高级模式详解")
    print("=" * 60)
    
    demo_plan_and_execute()
    demo_self_reflection()
    demo_rewoo()
    demo_pattern_selection_guide()
    
    print("\n" + "=" * 60)
    print("学习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
