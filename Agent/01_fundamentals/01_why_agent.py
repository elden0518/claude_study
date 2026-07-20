"""
==============================================================================
第一课：为什么要学习 Agent 开发
==============================================================================

【学习目标】
- 理解什么是 AI Agent
- 掌握 Agent 与传统程序的核心区别
- 了解 Agent 的核心组成部分
- 认识主流 Agent 开发框架

【核心概念】
Agent = LLM（大脑）+ Memory（记忆）+ Tools（工具）+ Planning（规划）

【前置知识】
- 基本 Python 编程能力
- 了解 LLM 的基本概念（可选）

==============================================================================
"""

# ============================================================================
# 第一部分：什么是 Agent？
# ============================================================================
#
# Agent（智能体）是一种能够自主感知环境、做出决策并执行动作的软件系统。
# 与传统程序不同，Agent 不是按照固定的 if-else 逻辑执行，而是通过
# LLM 的推理能力来动态决定下一步行动。
#
# 核心公式：
#   Agent = LLM + Memory + Tools + Planning
#
# ┌─────────────────────────────────────────────────────────────┐
# │                      Agent 架构                              │
# │                                                             │
# │   ┌──────────┐    ┌──────────┐    ┌──────────────────┐     │
# │   │  LLM     │───▶│ Planning │───▶│   Tool Executor  │     │
# │   │  (大脑)  │───│  (规划)  │───│   (工具执行器)    │     │
# │   └──────────┘    └──────────┘    └──────────────────┘     │
# │        │                                    │               │
# │        ▼                                    ▼               │
# │   ┌──────────┐                      ┌──────────────┐       │
# │   │ Memory   │                      │  Environment │       │
# │   │ (记忆)   │                      │  (环境/工具)  │       │
# │   └──────────┘                      └──────────────┘       │
# └─────────────────────────────────────────────────────────────┘


# ============================================================================
# 第二部分：Agent vs 传统程序 vs 简单 LLM 调用
# ============================================================================

def compare_approaches():
    """对比三种不同的开发方式"""

    # ── 方式1：传统程序（固定逻辑） ─────────────────────────────
    # 特点：逻辑硬编码，确定性强，但灵活性差
    traditional_code = """
    def process_order(order):
        if order.amount > 1000:
            return send_to_manager(order)    # 固定分支
        elif order.amount > 100:
            return auto_approve(order)       # 固定分支
        else:
            return reject(order)             # 固定分支
    # 问题：无法处理未预见的情况，每增加一种情况就要修改代码
    """

    # ── 方式2：简单 LLM 调用（无状态问答） ──────────────────────
    # 特点：有智能但无记忆、无工具，只能回答不能行动
    simple_llm_code = """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "今天天气怎么样？"}]
    )
    # 问题：无法记住上下文，无法调用外部工具，每次都是独立问答
    """

    # ── 方式3：Agent（智能 + 记忆 + 工具 + 规划）────────────────
    # 特点：能思考、能记忆、能行动、能规划
    agent_code = """
    agent = Agent(
        llm=claude_model,           # 大脑：推理和决策
        memory=conversation_memory,  # 记忆：记住上下文
        tools=[search, calculator,  # 工具：执行具体操作
               email_sender, db_query],
        planning_strategy="react"    # 规划：决定如何使用工具
    )

    # Agent 可以自主完成复杂任务：
    # "帮我查一下北京明天的天气，如果下雨就提醒我带伞，
    #  并给同事发个消息说会议改到室内"
    result = agent.run(user_input)
    # Agent 会自动：
    # 1. 调用天气API查天气
    # 2. 判断是否下雨
    # 3. 如果需要，设置提醒
    # 4. 调用邮件/消息工具通知同事
    """

    print("=" * 60)
    print("三种开发方式对比")
    print("=" * 60)
    print(f"{'特性':<15} {'传统程序':<15} {'简单LLM':<15} {'Agent':<15}")
    print("-" * 60)
    print(f"{'智能推理':<15} {'❌':<15} {'✅':<15} {'✅':<15}")
    print(f"{'上下文记忆':<15} {'❌':<15} {'':<15} {'✅':<15}")
    print(f"{'工具调用':<15} {'✅(硬编码)':<15} {'❌':<15} {'✅(自主)':<15}")
    print(f"{'自主规划':<15} {'':<15} {'❌':<15} {'✅':<15}")
    print(f"{'动态决策':<15} {'❌':<15} {'❌':<15} {'✅':<15}")
    print(f"{'错误恢复':<15} {'❌':<15} {'❌':<15} {'✅':<15}")
    print()


# ============================================================================
# 第三部分：Agent 的核心组成（四大模块）
# ============================================================================

class AgentComponents:
    """
    Agent 的四大核心模块详解

    每个模块都有明确的职责，模块之间松耦合，可以独立替换和升级。
    """

    # ── 模块1：LLM（大脑）───────────────────────────────────────
    # 负责：推理、决策、理解自然语言
    # 类比：人的大脑，负责思考和判断
    LLM_MODULE = {
        "name": "LLM (大语言模型)",
        "role": "推理与决策中心",
        "examples": ["Claude", "GPT-4", "Gemini", "本地模型"],
        "responsibilities": [
            "理解用户意图",
            "制定行动计划",
            "决定使用哪个工具",
            "生成自然语言回复",
        ],
    }

    # ── 模块2：Memory（记忆）────────────────────────────────────
    # 负责：存储和检索历史信息
    # 类比：人的记忆系统（短期记忆 + 长期记忆）
    MEMORY_MODULE = {
        "name": "Memory (记忆系统)",
        "role": "上下文管理",
        "types": {
            "short_term": "当前对话的上下文（类似工作记忆）",
            "long_term": "跨会话的持久化记忆（类似长期记忆）",
            "episodic": "特定事件的记忆（类似情景记忆）",
            "semantic": "事实和知识的记忆（类似语义记忆）",
        },
    }

    # ── 模块3：Tools（工具）─────────────────────────────────────
    # 负责：执行具体操作（搜索、计算、API调用等）
    # 类比：人的双手，负责执行具体动作
    TOOLS_MODULE = {
        "name": "Tools (工具集)",
        "role": "执行具体操作",
        "examples": [
            "搜索引擎", "计算器", "数据库查询",
            "邮件发送", "文件操作", "API调用",
        ],
        "design_principles": [
            "每个工具职责单一",
            "工具描述要清晰（LLM 靠描述来决定用哪个工具）",
            "输入输出要有明确的 Schema",
        ],
    }

    # ── 模块4：Planning（规划）──────────────────────────────────
    # 负责：决定执行策略和步骤
    # 类比：人的执行功能（Executive Function）
    PLANNING_MODULE = {
        "name": "Planning (规划系统)",
        "role": "任务分解与执行策略",
        "strategies": {
            "react": "ReAct - 推理+行动交替进行",
            "plan_and_execute": "先制定完整计划，再逐步执行",
            "self_reflection": "执行后自我反思和改进",
            "tree_of_thought": "多路径探索，选择最优方案",
        },
    }

    @classmethod
    def print_overview(cls):
        """打印四大模块概览"""
        print("=" * 60)
        print("Agent 四大核心模块")
        print("=" * 60)
        for module in [cls.LLM_MODULE, cls.MEMORY_MODULE,
                       cls.TOOLS_MODULE, cls.PLANNING_MODULE]:
            print(f"\n📦 {module['name']}")
            print(f"   角色：{module['role']}")
            if "examples" in module:
                print(f"   示例：{', '.join(module['examples'])}")
            if "types" in module:
                for k, v in module["types"].items():
                    print(f"   - {k}: {v}")
            if "strategies" in module:
                for k, v in module["strategies"].items():
                    print(f"   - {k}: {v}")
        print()


# ============================================================================
# 第四部分：Agent 的工作流程（ReAct 模式）
# ============================================================================

def demonstrate_react_flow():
    """
    演示 ReAct (Reasoning + Acting) 模式的工作流程

    ReAct 是最经典的 Agent 模式：
    1. Thought（思考）：分析当前情况，决定下一步
    2. Action（行动）：调用工具执行
    3. Observation（观察）：获取工具返回结果
    4. 重复 1-3 直到任务完成
    """

    # 模拟一个完整的 ReAct 循环
    react_example = {
        "user_input": "北京今天多少度？和上海比哪个更热？",

        "step_1_thought": (
            "用户问了两个城市的温度，我需要分别查询。"
            "先查北京的天气。"
        ),
        "step_1_action": "call_tool: get_weather(city='北京')",
        "step_1_observation": "北京：晴天，28°C",

        "step_2_thought": (
            "已获取北京温度28°C。现在需要查上海的天气来比较。"
        ),
        "step_2_action": "call_tool: get_weather(city='上海')",
        "step_2_observation": "上海：多云，32°C",

        "step_3_thought": (
            "北京28°C，上海32°C。上海更热，差4度。"
            "现在可以回答用户了。"
        ),
        "step_3_action": "return_answer: "
                         "'北京今天28°C（晴天），上海32°C（多云）。"
                         "上海比北京热4度。'",
    }

    print("=" * 60)
    print("ReAct 模式工作流程演示")
    print("=" * 60)
    print(f"用户输入：{react_example['user_input']}\n")

    for i in range(1, 4):
        print(f"── 第 {i} 轮 ──")
        print(f"   Thought:    {react_example[f'step_{i}_thought']}")
        print(f"  🔧 Action:     {react_example[f'step_{i}_action']}")
        if f"step_{i}_observation" in react_example:
            print(f"  👁️ Observation: {react_example[f'step_{i}_observation']}")
        print()


# ============================================================================
# 第五部分：主流 Agent 框架对比
# ============================================================================

def compare_frameworks():
    """对比主流 Agent 开发框架"""

    frameworks = [
        {
            "name": "LangGraph",
            "strengths": ["图结构编排", "状态管理强", "人机协作"],
            "best_for": "复杂工作流、多Agent协作",
            "learning_curve": "中等",
        },
        {
            "name": "CrewAI",
            "strengths": ["角色定义清晰", "多Agent协作简单", "上手快"],
            "best_for": "多Agent团队协作",
            "learning_curve": "低",
        },
        {
            "name": "AutoGen",
            "strengths": ["多Agent对话", "微软生态", "灵活配置"],
            "best_for": "多Agent对话式协作",
            "learning_curve": "中等",
        },
        {
            "name": "原生 Python",
            "strengths": ["完全控制", "无框架依赖", "轻量级"],
            "best_for": "学习原理、简单Agent、定制化需求",
            "learning_curve": "高（需要自己实现一切）",
        },
    ]

    print("=" * 60)
    print("主流 Agent 框架对比")
    print("=" * 60)
    for fw in frameworks:
        print(f"\n🔷 {fw['name']}")
        print(f"   优势：{', '.join(fw['strengths'])}")
        print(f"   适合：{fw['best_for']}")
        print(f"   学习曲线：{fw['learning_curve']}")
    print()


# ============================================================================
# 第六部分：进阶知识 - Agent 开发的核心挑战
# ============================================================================

def advanced_challenges():
    """
    进阶：Agent 开发中的核心挑战与解决方案

    这些是实际开发中会遇到的问题，了解它们有助于写出更好的 Agent。
    """
    challenges = [
        {
            "challenge": "幻觉问题 (Hallucination)",
            "description": "Agent 可能编造不存在的工具调用结果",
            "solutions": [
                "工具调用结果必须来自真实执行",
                "添加结果验证层",
                "使用结构化输出减少歧义",
            ],
        },
        {
            "challenge": "无限循环 (Infinite Loop)",
            "description": "Agent 可能陷入重复调用同一工具的循环",
            "solutions": [
                "设置最大迭代次数",
                "检测重复动作模式",
                "添加超时机制",
            ],
        },
        {
            "challenge": "Token 成本控制",
            "description": "长对话和复杂推理会消耗大量 Token",
            "solutions": [
                "记忆压缩和摘要",
                "选择性上下文注入",
                "使用更小的模型做简单任务",
            ],
        },
        {
            "challenge": "工具选择准确性",
            "description": "Agent 可能选错工具或错误使用工具",
            "solutions": [
                "清晰的工具描述和文档",
                "Few-shot 示例引导",
                "工具调用的输入验证",
            ],
        },
        {
            "challenge": "安全性",
            "description": "Agent 可能被注入恶意指令或滥用工具",
            "solutions": [
                "工具权限控制",
                "输入/输出过滤",
                "人机审核关键操作",
            ],
        },
    ]

    print("=" * 60)
    print("Agent 开发核心挑战与解决方案（进阶）")
    print("=" * 60)
    for item in challenges:
        print(f"\n⚠️  {item['challenge']}")
        print(f"   描述：{item['description']}")
        print(f"   解决方案：")
        for sol in item["solutions"]:
            print(f"   - {sol}")
    print()


# ============================================================================
# 主入口：运行所有演示
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🤖" * 30)
    print("第一课：为什么要学习 Agent 开发")
    print("🤖" * 30 + "\n")

    # 1. 对比三种开发方式
    compare_approaches()

    # 2. 四大核心模块
    AgentComponents.print_overview()

    # 3. ReAct 工作流程
    demonstrate_react_flow()

    # 4. 框架对比
    compare_frameworks()

    # 5. 进阶挑战
    advanced_challenges()

    print("=" * 60)
    print("✅ 第一课学习完成！")
    print("=" * 60)
    print("""
 本课要点总结：
1. Agent = LLM + Memory + Tools + Planning
2. Agent 与传统程序的核心区别在于「自主决策」能力
3. ReAct 是最基础的 Agent 模式：思考→行动→观察→循环
4. 选择合适的框架取决于项目复杂度和团队情况
5. 生产级 Agent 需要解决幻觉、循环、成本、安全等问题

 下一课：项目架构总览 - 我们将设计一个完整的 Agent 项目结构
    """)
