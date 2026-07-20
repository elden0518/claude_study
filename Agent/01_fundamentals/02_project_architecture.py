"""
==============================================================================
第二课：项目架构总览
==============================================================================

【学习目标】
- 理解 Agent 项目的模块化设计思想
- 掌握标准项目目录结构
- 理解各模块之间的依赖关系
- 学会设计可扩展的 Agent 架构

【核心概念】
- 模块化设计（Modular Design）
- 依赖注入（Dependency Injection）
- 关注点分离（Separation of Concerns）

【前置知识】
- 第一课：Agent 的四大核心模块

==============================================================================
"""

import os
from pathlib import Path


# ============================================================================
# 第一部分：为什么需要好的项目架构？
# ============================================================================

def why_architecture_matters():
    """
    解释为什么 Agent 项目需要良好的架构设计

    糟糕的架构会导致：
    - 代码耦合严重，修改一个功能影响其他功能
    - 无法复用组件，每次都要重写
    - 难以测试，无法隔离验证
    - 难以扩展，添加新功能困难

    好的架构应该做到：
    - 每个模块职责单一
    - 模块之间通过接口通信
    - 可以轻松替换某个模块的实现
    - 易于测试和调试
    """

    # ── 反面示例：所有代码写在一个文件里 ───────────────────────
    bad_example = """
    # ❌ 糟糕的设计：所有逻辑混在一起
    def handle_user_input(user_input):
        # 解析用户输入
        # 查询数据库
        # 调用 LLM
        # 管理对话历史
        # 调用外部 API
        # 格式化输出
        # 保存日志
        # ... 500行代码 ...
        return response

    # 问题：
    # 1. 无法单独测试 LLM 调用逻辑
    # 2. 无法替换数据库实现
    # 3. 无法复用对话管理逻辑
    # 4. 代码越来越长，难以维护
    """

    # ── 正面示例：模块化设计 ────────────────────────────────────
    good_example = """
    # ✅ 好的设计：每个模块各司其职
    from core.config import Config
    from core.llm import LLMClient
    from memory.conversation_memory import ConversationMemory
    from agents.react_agent import ReactAgent
    from tools.registry import ToolRegistry

    # 组装 Agent（依赖注入）
    agent = ReactAgent(
        llm=LLMClient(config),
        memory=ConversationMemory(),
        tools=ToolRegistry().get_all_tools(),
    )

    # 好处：
    # 1. 可以单独替换 LLM（Claude → GPT）
    # 2. 可以单独替换 Memory（内存 → Redis）
    # 3. 每个模块可以独立测试
    # 4. 新工具只需注册，不需要修改 Agent 代码
    """

    print("=" * 60)
    print("架构设计对比")
    print("=" * 60)
    print(f"{'特性':<20} {'❌ 单体设计':<18} {'✅ 模块化设计':<18}")
    print("-" * 60)
    print(f"{'可测试性':<20} {'困难':<18} {'容易':<18}")
    print(f"{'可维护性':<20} {'差':<18} {'好':<18}")
    print(f"{'可扩展性':<20} {'差':<18} {'好':<18}")
    print(f"{'可复用性':<20} {'差':<18} {'好':<18}")
    print(f"{'团队协作':<20} {'冲突多':<18} {'并行开发':<18}")
    print()


# ============================================================================
# 第二部分：标准 Agent 项目目录结构
# ============================================================================

def show_project_structure():
    """
    展示标准的 Agent 项目目录结构

    这个结构遵循以下原则：
    1. 按功能模块组织（不是按技术分层）
    2. 每个模块有清晰的边界
    3. 核心逻辑与入口分离
    4. 配置与代码分离
    """

    structure = """
    agent_project/                    # 项目根目录
    │
    ├── core/                         # 核心模块 - 系统基础设施
    │   ├── __init__.py
    │   ├── config.py                 # 配置管理（环境变量、参数）
    │   ├── logger.py                 # 日志系统
    │   ├── exceptions.py             # 自定义异常定义
    │   └── llm_client.py            # LLM 客户端封装
    │
    ├── schema/                       # Schema 模块 - 数据结构定义
    │   ├── __init__.py
    │   ├── message.py                # 消息格式定义
    │   ├── tool.py                   # 工具 Schema 定义
    │   ├── state.py                  # Agent 状态定义
    │   └── response.py              # 响应格式定义
    │
    ├── memory/                       # Memory 模块 - 记忆管理
    │   ├── __init__.py
    │   ├── base.py                   # 记忆基类（接口定义）
    │   ├── short_term.py             # 短期记忆（对话上下文）
    │   ├── long_term.py              # 长期记忆（持久化存储）
    │   └── vector_memory.py          # 向量记忆（语义检索）
    │
    ├── session/                      # Session 模块 - 会话管理
    │   ├── __init__.py
    │   ├── base.py                   # 会话基类
    │   ├── manager.py                # 会话管理器（多用户）
    │   └── store.py                  # 会话存储（内存/Redis）
    │
    ├── agents/                       # Agents 模块 - 核心业务逻辑
    │   ├── __init__.py
    │   ├── base.py                   # Agent 基类
    │   ├── react_agent.py            # ReAct 模式 Agent
    │   ├── planner_agent.py          # Plan-and-Execute Agent
    │   └── multi_agent.py            # 多 Agent 协作
    │
    ├── tools/                        # Tools 模块 - 工具集
    │   ├── __init__.py
    │   ├── registry.py               # 工具注册中心
    │   ├── search.py                 # 搜索工具
    │   ├── calculator.py             # 计算工具
    │   └── api_caller.py             # API 调用工具
    │
    ├── entry/                        # 入口模块 - 运行方式
    │   ├── __init__.py
    │   ├── cli.py                    # 命令行入口
    │   └── api.py                    # API 服务入口
    │
    ├── tests/                        # 测试目录
    │   ├── __init__.py
    │   ├── test_core.py
    │   ├── test_agents.py
    │   └── test_tools.py
    │
    ├── .env                          # 环境变量（不提交到 Git）
    ├── .env.example                  # 环境变量示例
    ├── requirements.txt              # 依赖列表
    ├── main.py                       # 主入口文件
    └── README.md                     # 项目说明
    """

    print("=" * 60)
    print("标准 Agent 项目目录结构")
    print("=" * 60)
    print(structure)


# ============================================================================
# 第三部分：模块依赖关系图
# ============================================================================

def show_dependency_graph():
    """
    展示模块之间的依赖关系

    依赖原则：
    - 上层模块依赖下层模块
    - 同层模块之间不直接依赖
    - 依赖方向：entry → agents → (core, schema, memory, session, tools)

    ┌─────────────────────────────────────────────────────────────┐
    │  入口层 (entry/)                                             │
    │  cli.py / api.py                                            │
    └──────────────────────┬──────────────────────────────────────┘
                           │ 依赖
    ┌──────────────────────▼──────────────────────────────────────┐
    │  业务层 (agents/)                                            │
    │  ReactAgent / PlannerAgent / MultiAgent                     │
    ──┬──────────┬──────────┬──────────┬─────────────────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
    ┌────────┌────────┐┌────────┌────────┐┌────────
    │ core/  ││schema/ ││memory/ ││session/││ tools/ │
    │ 基础设施││ 数据结构││ 记忆   ││ 会话   ││ 工具   │
    └────────┘└────────┘└────────┘└────────┘└────────┘
    """

    print("=" * 60)
    print("模块依赖关系")
    print("=" * 60)
    print("""
    依赖方向（从上到下）：

    entry/          ← 入口层：CLI、API 服务
        │
        ▼
    agents/         ← 业务层：Agent 核心逻辑
        │
        ├────────────┬────────────┬────────────┬───────────┐
        ▼            ▼            ▼            ▼           ▼
    core/         schema/      memory/      session/     tools/
    基础设施       数据结构      记忆管理      会话管理      工具集

    规则：
    ✓ agents/ 可以依赖 core/schema/memory/session/tools
    ✓ entry/ 可以依赖 agents/ 和所有下层模块
     core/ 不应该依赖 agents/（避免循环依赖）
    ✗ memory/ 和 session/ 之间不直接依赖
    """)


# ============================================================================
# 第四部分：模块职责详解
# ============================================================================

def explain_modules():
    """详细解释每个模块的职责和设计要点"""

    modules = {
        "core/（核心模块）": {
            "职责": "提供系统级基础设施",
            "包含": [
                "config.py - 配置管理：读取 .env 文件，管理 API Key、模型参数等",
                "logger.py - 日志系统：统一的日志格式，支持不同级别",
                "exceptions.py - 异常定义：LLMError、ToolError 等自定义异常",
                "llm_client.py - LLM 客户端：封装 API 调用，统一接口",
            ],
            "设计要点": "无业务逻辑，纯基础设施，所有其他模块都依赖它",
        },
        "schema/（数据结构）": {
            "职责": "定义所有数据模型和接口契约",
            "包含": [
                "message.py - 消息格式：UserMessage、AIMessage、ToolMessage",
                "tool.py - 工具定义：ToolSchema（名称、描述、参数）",
                "state.py - 状态定义：AgentState（当前状态、历史等）",
                "response.py - 响应格式：AgentResponse（回复、工具调用等）",
            ],
            "设计要点": "使用 Pydantic 做数据验证，确保类型安全",
        },
        "memory/（记忆管理）": {
            "职责": "管理 Agent 的记忆系统",
            "包含": [
                "base.py - 抽象基类：定义 Memory 接口",
                "short_term.py - 短期记忆：当前对话的上下文窗口",
                "long_term.py - 长期记忆：跨会话的持久化记忆",
                "vector_memory.py - 向量记忆：基于语义的检索记忆",
            ],
            "设计要点": "通过基类实现可替换，内存/Redis/向量数据库可切换",
        },
        "session/（会话管理）": {
            "职责": "管理多用户会话",
            "包含": [
                "base.py - 会话基类：定义 Session 接口",
                "manager.py - 会话管理器：创建/销毁/查找会话",
                "store.py - 会话存储：内存存储或 Redis 存储",
            ],
            "设计要点": "会话隔离，每个用户独立上下文，支持并发",
        },
        "agents/（核心业务）": {
            "职责": "实现 Agent 的核心逻辑",
            "包含": [
                "base.py - Agent 基类：定义 Agent 接口",
                "react_agent.py - ReAct Agent：推理+行动循环",
                "planner_agent.py - 规划 Agent：先计划后执行",
                "multi_agent.py - 多 Agent：协作完成任务",
            ],
            "设计要点": "通过组合（非继承）使用其他模块",
        },
        "tools/（工具集）": {
            "职责": "提供 Agent 可调用的工具",
            "包含": [
                "registry.py - 工具注册中心：注册/查找/管理工具",
                "search.py - 搜索工具：网络搜索、知识库搜索",
                "calculator.py - 计算工具：数学计算、单位转换",
                "api_caller.py - API 工具：调用外部 REST API",
            ],
            "设计要点": "插件化设计，新工具只需注册即可使用",
        },
        "entry/（入口模块）": {
            "职责": "提供不同的运行方式",
            "包含": [
                "cli.py - 命令行：交互式终端对话",
                "api.py - API 服务：FastAPI HTTP 接口",
            ],
            "设计要点": "薄入口层，不包含业务逻辑",
        },
    }

    print("=" * 60)
    print("各模块职责详解")
    print("=" * 60)
    for name, info in modules.items():
        print(f"\n📁 {name}")
        print(f"   职责：{info['职责']}")
        print(f"   包含文件：")
        for item in info["包含"]:
            print(f"   - {item}")
        print(f"   设计要点：{info['设计要点']}")
    print()


# ============================================================================
# 第五部分：进阶 - 架构设计模式
# ============================================================================

def advanced_patterns():
    """
    进阶：Agent 项目中常用的架构设计模式

    这些模式可以帮助你的项目更加健壮和可扩展。
    """

    patterns = [
        {
            "name": "依赖注入 (Dependency Injection)",
            "concept": "不直接创建依赖对象，而是从外部传入",
            "example": """
    # ❌ 硬编码依赖
    class MyAgent:
        def __init__(self):
            self.llm = ClaudeClient()  # 硬编码，无法替换

    # ✅ 依赖注入
    class MyAgent:
        def __init__(self, llm: BaseLLM, memory: BaseMemory):
            self.llm = llm      # 可以传入任何 LLM 实现
            self.memory = memory  # 可以传入任何 Memory 实现
            """,
        },
        {
            "name": "策略模式 (Strategy Pattern)",
            "concept": "将算法/策略封装为可互换的组件",
            "example": """
    # 不同的规划策略可以互换
    class ReactStrategy(PlanningStrategy):
        def plan(self, state): ...

    class PlanExecuteStrategy(PlanningStrategy):
        def plan(self, state): ...

    # Agent 使用策略
    agent = Agent(strategy=ReactStrategy())
    agent = Agent(strategy=PlanExecuteStrategy())  # 轻松切换
            """,
        },
        {
            "name": "观察者模式 (Observer Pattern)",
            "concept": "当状态变化时自动通知相关组件",
            "example": """
    # 日志记录器观察 Agent 状态变化
    class LoggingObserver:
        def on_agent_action(self, action):
            logger.info(f"Agent 执行了: {action}")

        def on_agent_response(self, response):
            logger.info(f"Agent 回复: {response}")

    agent.add_observer(LoggingObserver())
            """,
        },
        {
            "name": "工厂模式 (Factory Pattern)",
            "concept": "根据配置动态创建不同类型的对象",
            "example": """
    class AgentFactory:
        @staticmethod
        def create(agent_type: str, **kwargs) -> BaseAgent:
            if agent_type == "react":
                return ReactAgent(**kwargs)
            elif agent_type == "planner":
                return PlannerAgent(**kwargs)
            else:
                raise ValueError(f"未知 Agent 类型: {agent_type}")

    # 通过配置创建
    agent = AgentFactory.create(config.agent_type, **config_params)
            """,
        },
    ]

    print("=" * 60)
    print("进阶：架构设计模式")
    print("=" * 60)
    for p in patterns:
        print(f"\n🔷 {p['name']}")
        print(f"   概念：{p['concept']}")
        print(f"   示例：")
        print(p["example"])
    print()


# ============================================================================
# 第六部分：动手实践 - 创建项目骨架
# ============================================================================

def create_project_skeleton(base_path: str):
    """
    自动创建项目骨架目录结构

    这个函数会创建标准的 Agent 项目目录结构，
    包含所有必要的 __init__.py 文件。
    """

    # 定义目录结构
    directories = [
        "core",
        "schema",
        "memory",
        "session",
        "agents",
        "tools",
        "entry",
        "tests",
    ]

    base = Path(base_path)

    print("=" * 60)
    print("创建项目骨架")
    print("=" * 60)

    for dir_name in directories:
        dir_path = base / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        # 创建 __init__.py
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text(f'"""{dir_name} 模块"""\n')
        print(f"  ✅ 创建目录: {dir_name}/")

    # 创建 .env.example
    env_example = base / ".env.example"
    if not env_example.exists():
        env_example.write_text(
            "# LLM 配置\n"
            "ANTHROPIC_API_KEY=your_key_here\n"
            "OPENAI_API_KEY=your_key_here\n"
            "LLM_MODEL=claude-sonnet-4-20250514\n"
            "\n"
            "# Redis 配置（可选）\n"
            "REDIS_URL=redis://localhost:6379\n"
            "\n"
            "# 日志配置\n"
            "LOG_LEVEL=INFO\n"
        )
    print(f"  ✅ 创建文件: .env.example")

    print(f"\n  项目骨架已创建在: {base_path}")
    print()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🏗️" * 30)
    print("第二课：项目架构总览")
    print("🏗️" * 30 + "\n")

    # 1. 为什么需要好架构
    why_architecture_matters()

    # 2. 标准目录结构
    show_project_structure()

    # 3. 依赖关系
    show_dependency_graph()

    # 4. 模块职责
    explain_modules()

    # 5. 进阶设计模式
    advanced_patterns()

    # 6. 创建项目骨架
    project_path = os.path.dirname(os.path.abspath(__file__))
    create_project_skeleton(project_path)

    print("=" * 60)
    print("✅ 第二课学习完成！")
    print("=" * 60)
    print("""
 本课要点总结：
1. 模块化设计让代码可维护、可测试、可扩展
2. 标准目录结构：core/schema/memory/session/agents/tools/entry
3. 依赖方向：entry → agents → 基础设施模块
4. 常用设计模式：依赖注入、策略模式、观察者模式、工厂模式
5. 每个模块职责单一，通过接口通信

 下一课：Core 模块 - 我们将实现配置管理、日志系统和异常处理
    """)
