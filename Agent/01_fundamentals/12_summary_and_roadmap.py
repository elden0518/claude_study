"""
==============================================================================
第十二课：总结与进阶路线图
==============================================================================

【学习目标】
- 回顾所有课程的核心知识点
- 理解知识体系的整体脉络
- 制定个人进阶学习计划
- 了解持续学习的资源和方法

【核心概念】
- 知识体系回顾
- 进阶学习路线
- 实践项目建议
- 持续学习资源

【前置知识】
- 所有前序课程

==============================================================================
"""


# ============================================================================
# 第一部分：完整知识体系回顾
# ============================================================================

def knowledge_review():
    """回顾所有课程的核心知识点"""

    courses = [
        {
            "lesson": "第一课：为什么要学习 Agent 开发",
            "key_points": [
                "Agent = LLM + Memory + Tools + Planning",
                "Agent 与传统程序的核心区别：自主决策",
                "ReAct 模式：Thought → Action → Observation",
                "主流框架对比：LangGraph、CrewAI、AutoGen",
            ],
        },
        {
            "lesson": "第二课：项目架构总览",
            "key_points": [
                "模块化设计：core/schema/memory/session/agents/tools/entry",
                "依赖方向：entry → agents → 基础设施模块",
                "设计模式：依赖注入、策略模式、观察者模式",
            ],
        },
        {
            "lesson": "第三课：Core 模块",
            "key_points": [
                "配置管理：dataclass + 环境变量 + 验证",
                "日志系统：分级日志、统一格式、模块化",
                "异常处理：层次化设计、携带上下文",
                "LLM 客户端：统一接口、Mock 实现、重试机制",
            ],
        },
        {
            "lesson": "第四课：Schema 模块",
            "key_points": [
                "Pydantic 模型：类型安全、自动验证",
                "Message：核心数据单元",
                "ToolSchema：工具定义，LLM 依赖它做选择",
                "AgentState：跟踪运行状态",
            ],
        },
        {
            "lesson": "第五课：Memory 模块",
            "key_points": [
                "短期记忆：滑动窗口管理上下文",
                "长期记忆：持久化存储重要信息",
                "向量记忆：语义检索相关记忆",
                "记忆压缩：摘要、过滤、时间衰减",
            ],
        },
        {
            "lesson": "第六课：Session 模块",
            "key_points": [
                "会话隔离：不同用户独立上下文",
                "生命周期：创建 → 活跃 → 超时 → 关闭",
                "会话存储：内存 vs Redis",
                "观察者模式：会话事件监听",
            ],
        },
        {
            "lesson": "第七课：Agents 模块",
            "key_points": [
                "工具系统：定义、注册、调用",
                "ReAct Agent：推理 + 行动循环",
                "决策循环：最大迭代次数防止无限循环",
                "错误处理：工具失败、迭代超限",
            ],
        },
        {
            "lesson": "第八课：入口文件与运行方式",
            "key_points": [
                "CLI 入口：交互式命令行",
                "API 服务：FastAPI HTTP 接口",
                "异步编程：并发处理请求",
                "优雅关闭：资源清理",
            ],
        },
        {
            "lesson": "第九课：测试驱动开发",
            "key_points": [
                "TDD 流程：Red → Green → Refactor",
                "Mock LLM：控制输出，确保可重复",
                "单元测试 + 集成测试",
                "边界情况测试",
            ],
        },
        {
            "lesson": "第十课：生产环境部署",
            "key_points": [
                "Docker 容器化",
                "配置管理：环境变量、密钥管理",
                "日志管理：分级、滚动、JSON",
                "监控告警：应用/LLM/系统指标",
                "水平扩展：无状态设计 + 负载均衡",
            ],
        },
        {
            "lesson": "第十一课：进阶 - 多 Agent 协作",
            "key_points": [
                "Supervisor 模式：协调者 + Worker",
                "Plan-and-Execute：先计划后执行",
                "Self-Reflection：自我评估改进",
                "Agent 间通信：消息总线",
            ],
        },
    ]

    print("=" * 60)
    print("完整知识体系回顾")
    print("=" * 60)

    for i, course in enumerate(courses, 1):
        print(f"\n📘 {course['lesson']}")
        for point in course["key_points"]:
            print(f"   • {point}")

    print()


# ============================================================================
# 第二部分：知识体系架构图
# ============================================================================

def show_architecture_map():
    """展示知识体系架构图"""

    print("=" * 60)
    print("Agent 开发知识体系架构")
    print("=" * 60)
    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │                        Agent 开发知识体系                        │
  │                                                                 │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │  基础层（第1-4课）                                        │   │
  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
  │  │  │ Agent    │ │ 项目     │ │ Core     │ │ Schema   │   │   │
  │  │  │ 概念     │ │ 架构     │ │ 模块     │ │ 模块     │   │   │
  │  │  │          │ │          │ │ 配置/日志 │ │ 数据模型 │   │   │
  │  │  │ 为什么   │ │ 模块化   │ │ 异常/LLM │ │ 消息/工具 │   │   │
  │  │  │ 学Agent  │ │ 设计模式 │ │ 客户端   │ │ 状态/响应 │   │   │
  │  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                              │                                  │
  │                              ▼                                  │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │  核心层（第5-7课）                                        │   │
  │  │  ──────────┐ ┌──────────┐ ┌──────────┐                 │   │
  │  │  │ Memory   │ │ Session  │ │ Agents   │                 │   │
  │  │  │ 模块     │ │ 模块     │ │ 模块     │                 │   │
  │  │  │          │ │          │ │          │                 │   │
  │  │  │ 短期记忆 │ │ 会话隔离 │ │ ReAct    │                 │   │
  │  │  │ 长期记忆 │ │ 生命周期 │ │ 工具系统 │                 │   │
  │  │  │ 向量记忆 │ │ 持久化   │ │ 决策循环 │                 │   │
  │  │  └──────────┘ └──────────┘ └──────────┘                 │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                              │                                  │
  │                              ▼                                  │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │  应用层（第8-10课）                                       │   │
  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                 │   │
  │  │  │ 入口     │ │ 测试     │ │ 生产     │                 │   │
  │  │  │ 与运行   │ │ 驱动开发 │ │ 部署     │                 │   │
  │  │  │          │ │          │ │          │                 │   │
  │  │  │ CLI/API  │ │ TDD      │ │ Docker   │                 │   │
  │  │  │ 异步     │ │ Mock     │ │ 监控     │                 │   │
  │  │  │ 优雅关闭 │ │ 集成测试 │ │ CI/CD    │                 │   │
  │  │  └──────────┘ └──────────┘ └──────────┘                 │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                              │                                  │
  │                              ▼                                  │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │  进阶层（第11-12课）                                      │   │
  │  │  ──────────┐ ┌──────────┐                               │   │
  │  │  │ 多Agent  │ │ 总结与   │                               │   │
  │  │  │ 协作     │ │ 进阶     │                               │   │
  │  │  │          │ │          │                               │   │
  │  │  │ Supervisor│ │ 知识回顾 │                               │   │
  │  │  │ Plan-Exec│ │ 学习路线 │                               │   │
  │  │  │ Self-Ref │ │ 实践项目 │                               │   │
  │  │  └──────────┘ └──────────                               │   │
  │  └─────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────┘
    """)


# ============================================================================
# 第三部分：学习路线建议
# ============================================================================

def learning_paths():
    """提供不同的学习路线建议"""

    print("=" * 60)
    print("学习路线建议")
    print("=" * 60)

    paths = [
        {
            "name": "快速上手路线（3-5天）",
            "target": "能构建简单的 Agent 应用",
            "courses": ["第1课", "第2课", "第7课", "第8课"],
            "project": "命令行聊天机器人",
            "time": "15-20小时",
        },
        {
            "name": "系统学习路线（2-3周）",
            "target": "能独立设计完整的 Agent 系统",
            "courses": ["第1-7课", "第9课"],
            "project": "带记忆的多轮对话 Agent",
            "time": "40-60小时",
        },
        {
            "name": "生产级路线（1-2月）",
            "target": "能构建和部署生产级 Agent 系统",
            "courses": ["全部12课"],
            "project": "多用户 Agent 服务平台",
            "time": "80-120小时",
        },
        {
            "name": "进阶专家路线（持续）",
            "target": "成为 Agent 开发专家",
            "courses": ["第11课", "开源项目", "论文阅读"],
            "project": "自主 Agent 系统",
            "time": "持续学习",
        },
    ]

    for path in paths:
        print(f"\n🎯 {path['name']}")
        print(f"   目标: {path['target']}")
        print(f"   课程: {', '.join(path['courses'])}")
        print(f"   项目: {path['project']}")
        print(f"   时间: {path['time']}")
    print()


# ============================================================================
# 第四部分：实践项目建议
# ============================================================================

def project_suggestions():
    """实践项目建议"""

    print("=" * 60)
    print("实践项目建议")
    print("=" * 60)

    projects = [
        {
            "name": "个人知识助手",
            "difficulty": "初级",
            "description": "能记住用户偏好，回答个人问题",
            "skills": ["Memory", "Session", "基础 Agent"],
            "features": [
                "记住用户姓名和偏好",
                "基于历史对话回答",
                "简单的工具调用（搜索、计算）",
            ],
        },
        {
            "name": "智能客服机器人",
            "difficulty": "中级",
            "description": "多用户客服系统，能处理常见问题",
            "skills": ["Session", "Memory", "Tools", "API"],
            "features": [
                "多用户会话隔离",
                "意图识别",
                "知识库检索",
                "人工转接",
            ],
        },
        {
            "name": "研究助手团队",
            "difficulty": "高级",
            "description": "多 Agent 协作完成研究任务",
            "skills": ["Multi-Agent", "Supervisor", "Planning"],
            "features": [
                "任务分解与分配",
                "多 Agent 协作",
                "结果汇总",
                "报告生成",
            ],
        },
        {
            "name": "自动化工作流平台",
            "difficulty": "专家",
            "description": "可配置的自动化工作流系统",
            "skills": ["全部技能", "生产部署", "监控"],
            "features": [
                "可视化工作流编辑",
                "自定义工具注册",
                "多租户支持",
                "监控与告警",
            ],
        },
    ]

    for proj in projects:
        print(f"\n🔷 {proj['name']}")
        print(f"   难度: {proj['difficulty']}")
        print(f"   描述: {proj['description']}")
        print(f"   技能: {', '.join(proj['skills'])}")
        print(f"   功能:")
        for feature in proj["features"]:
            print(f"     - {feature}")
    print()


# ============================================================================
# 第五部分：持续学习资源
# ============================================================================

def learning_resources():
    """持续学习资源"""

    print("=" * 60)
    print("持续学习资源")
    print("=" * 60)

    resources = {
        "官方文档": [
            ("Anthropic", "https://docs.anthropic.com/"),
            ("OpenAI", "https://platform.openai.com/docs"),
            ("LangChain", "https://python.langchain.com/"),
            ("LangGraph", "https://langchain-ai.github.io/langgraph/"),
        ],
        "开源项目": [
            ("LangChain", "https://github.com/langchain-ai/langchain"),
            ("LangGraph", "https://github.com/langchain-ai/langgraph"),
            ("CrewAI", "https://github.com/crewAIInc/crewAI"),
            ("AutoGen", "https://github.com/microsoft/autogen"),
        ],
        "论文与文章": [
            ("ReAct 论文", "Reasoning and Acting with Language Models"),
            ("Tree of Thoughts", "Tree of Thoughts: Deliberate Problem Solving"),
            ("Reflexion", "Reflexion: Language Agents with Verbal Reinforcement"),
            ("Agent Survey", "The Rise and Potential of Large Language Model Based Agents"),
        ],
        "社区与论坛": [
            ("LangChain Discord", "https://discord.gg/langchain"),
            ("Reddit r/LangChain", "https://reddit.com/r/LangChain"),
            ("Hacker News", "https://news.ycombinator.com/"),
        ],
        "实践平台": [
            ("Kaggle", "https://kaggle.com/ - AI 竞赛和数据集"),
            ("Hugging Face", "https://huggingface.co/ - 模型和数据集"),
            ("Papers with Code", "https://paperswithcode.com/ - 论文和代码"),
        ],
    }

    for category, items in resources.items():
        print(f"\n📚 {category}")
        for name, url in items:
            print(f"   • {name}: {url}")
    print()


# ============================================================================
# 第六部分：学习检查清单
# ============================================================================

def learning_checklist():
    """学习检查清单"""

    print("=" * 60)
    print("学习检查清单")
    print("=" * 60)
    print("""
 完成本课程后，你应该能够回答以下问题：

 基础概念
 ☐ 什么是 AI Agent？它与传统程序有什么区别？
 ☐ Agent 的四大核心模块是什么？
 ☐ ReAct 模式的工作流程是什么？

 架构设计
 ☐ 如何设计一个模块化的 Agent 项目？
 ☐ 各模块之间的依赖关系是什么？
 ☐ 常用的设计模式有哪些？

 核心模块
 ☐ 如何管理 Agent 的配置和日志？
 ☐ 如何定义消息和工具的数据结构？
 ☐ 短期记忆和长期记忆有什么区别？
 ☐ 如何管理多用户会话？

 Agent 实现
 ☐ 如何实现 ReAct Agent？
 ☐ 如何定义和注册工具？
 ☐ 如何处理 Agent 的错误和异常？

 生产部署
 ☐ 如何用 Docker 部署 Agent 服务？
  如何管理生产环境的配置？
 ☐ 如何监控 Agent 系统的健康状态？

 进阶知识
 ☐ Supervisor 模式的工作原理是什么？
  Plan-and-Execute 和 ReAct 有什么区别？
 ☐ 如何设计多 Agent 协作系统？

 如果你能回答以上所有问题，说明你已经掌握了 Agent 开发的核心知识！
    """)


# ============================================================================
# 第七部分：结语
# ============================================================================

def conclusion():
    """结语"""

    print("=" * 60)
    print("结语")
    print("=" * 60)
    print("""
 恭喜你完成了 Agent 开发的完整学习路线！

 通过这12节课，你已经：
 ✅ 理解了 Agent 的核心概念和架构
 ✅ 掌握了各模块的设计和实现
 ✅ 学会了测试和生产部署
 ✅ 了解了多 Agent 协作和高级模式

 接下来的建议：
 1. 选择一个实践项目，动手实现
 2. 阅读相关论文，深入理解原理
 3. 参与开源社区，贡献代码或文档
 4. 持续关注最新发展，AI 领域变化很快

 记住：
 - 学习最好的方式是动手实践
 - 不要害怕犯错，错误是最好的老师
 - 保持好奇心，持续学习

 祝你在 Agent 开发的道路上越走越远！🚀
    """)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🎓" * 30)
    print("第十二课：总结与进阶路线图")
    print("" * 30 + "\n")

    # 1. 知识回顾
    knowledge_review()

    # 2. 架构图
    show_architecture_map()

    # 3. 学习路线
    learning_paths()

    # 4. 实践项目
    project_suggestions()

    # 5. 学习资源
    learning_resources()

    # 6. 检查清单
    learning_checklist()

    # 7. 结语
    conclusion()

    print("=" * 60)
    print("✅ 全部课程学习完成！")
    print("=" * 60)
