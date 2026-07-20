"""
==============================================================================
第七课：Agents 模块 - 核心业务逻辑
==============================================================================

【学习目标】
- 理解 Agent 的核心工作原理
- 掌握 ReAct 模式（推理 + 行动）
- 学会定义和注册工具
- 掌握 Agent 的决策循环
- 实现完整的 Agent 系统

【核心概念】
- ReAct 模式：Thought → Action → Observation → 循环
- 工具注册与调用
- Agent 决策循环
- 错误处理与恢复

【前置知识】
- 第三课：Core 模块
- 第四课：Schema 模块
- 第五课：Memory 模块
- 第六课：Session 模块

==============================================================================
"""

import json
import uuid
import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable

from pydantic import BaseModel, Field

# ============================================================================
# 简化版依赖（便于独立运行）
# ============================================================================

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}")

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def tool_result(cls, content: str, tool_call_id: str) -> "Message":
        return cls(
            role=Role.TOOL, content=content, tool_call_id=tool_call_id
        )


# ============================================================================
# 第一部分：工具系统
# ============================================================================
#
# 工具是 Agent 执行具体操作的能力。
# 每个工具都有：
# - 名称：唯一标识
# - 描述：告诉 LLM 这个工具做什么
# - 参数 Schema：定义输入格式
# - 执行函数：实际执行逻辑


class Tool(BaseModel):
    """
    工具定义

    工具是 Agent 与外部世界交互的桥梁。
    LLM 通过工具描述来决定调用哪个工具。
    """

    name: str = Field(description="工具名称")
    description: str = Field(description="工具描述（LLM 靠这个理解工具）")
    parameters_schema: dict = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="参数 Schema（JSON Schema 格式）",
    )

    # 工具模型配置
    model_config = {"arbitrary_types_allowed": True}

    # 执行函数（不能直接用 Pydantic 序列化，所以单独处理）
    _execute_fn: Optional[Callable] = None

    def set_execute_fn(self, fn: Callable) -> "Tool":
        """设置执行函数"""
        self._execute_fn = fn
        return self

    async def execute(self, arguments: dict) -> str:
        """
        执行工具

        Args:
            arguments: 工具参数（由 LLM 生成）

        Returns:
            执行结果（字符串）
        """
        if self._execute_fn is None:
            return f"错误：工具 {self.name} 未设置执行函数"

        try:
            result = self._execute_fn(**arguments)
            # 如果是协程，await 它
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)
        except Exception as e:
            return f"工具执行错误: {e}"


class ToolRegistry:
    """
    工具注册中心

    管理所有可用工具，提供注册、查找、列表功能。
    Agent 通过注册中心获取可用工具列表。
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册工具"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        """获取所有工具"""
        return list(self._tools.values())

    def get_tool_schemas(self) -> list[dict]:
        """
        获取所有工具的 Schema（发送给 LLM）

        LLM 需要知道有哪些工具可用，以及每个工具的参数格式。
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                },
            }
            for tool in self._tools.values()
        ]

    def list_tools(self) -> str:
        """列出所有工具（用于调试）"""
        lines = ["可用工具:"]
        for tool in self._tools.values():
            lines.append(f"  - {tool.name}: {tool.description}")
        return "\n".join(lines)


# ============================================================================
# 第二部分：预定义工具实现
# ============================================================================

def create_demo_tools() -> ToolRegistry:
    """创建演示用的工具集"""

    registry = ToolRegistry()

    # ── 工具1：计算器 ───────────────────────────────────────────
    calculator = Tool(
        name="calculator",
        description="执行数学计算。支持加减乘除、幂运算等。"
                    "当用户需要进行数值计算时使用此工具。",
        parameters_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '2 + 3 * 4'",
                },
            },
            "required": ["expression"],
        },
    )

    def calc_expression(expression: str) -> str:
        """安全的数学表达式计算"""
        try:
            # 只允许安全的数学运算
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "错误：表达式包含不允许的字符"
            result = eval(expression)  # 注意：生产环境应使用更安全的解析器
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    calculator.set_execute_fn(calc_expression)
    registry.register(calculator)

    # ── 工具2：天气查询 ──────────────────────────────────────────
    weather = Tool(
        name="get_weather",
        description="查询指定城市的天气信息。返回温度、天气状况等。",
        parameters_schema={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称",
                },
            },
            "required": ["city"],
        },
    )

    # 模拟天气数据
    mock_weather_data = {
        "北京": {"temp": 28, "condition": "晴天", "humidity": 45},
        "上海": {"temp": 32, "condition": "多云", "humidity": 70},
        "广州": {"temp": 35, "condition": "小雨", "humidity": 85},
        "深圳": {"temp": 33, "condition": "阴天", "humidity": 75},
    }

    def get_weather(city: str) -> str:
        """查询天气（模拟）"""
        data = mock_weather_data.get(city)
        if data:
            return (
                f"{city}天气：{data['condition']}，"
                f"气温 {data['temp']}°C，湿度 {data['humidity']}%"
            )
        return f"未找到 {city} 的天气信息"

    weather.set_execute_fn(get_weather)
    registry.register(weather)

    # ─ 工具3：网络搜索 ──────────────────────────────────────────
    search = Tool(
        name="web_search",
        description="搜索互联网获取信息。当需要查找实时信息、新闻、"
                    "或特定资料时使用。",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
            },
            "required": ["query"],
        },
    )

    def web_search(query: str) -> str:
        """模拟网络搜索"""
        # 模拟搜索结果
        results = {
            "python": "Python 是一种广泛使用的高级编程语言，由 Guido van Rossum 创建。",
            "agent": "AI Agent 是能够自主感知、决策和行动的智能系统。",
            "天气": "可以通过天气API查询实时天气信息。",
        }
        for key, value in results.items():
            if key in query.lower():
                return f"搜索结果：{value}"
        return f"关于 '{query}' 的搜索结果：未找到精确匹配的信息。"

    search.set_execute_fn(web_search)
    registry.register(search)

    # ── 工具4：时间查询 ──────────────────────────────────────────
    time_tool = Tool(
        name="get_current_time",
        description="获取当前日期和时间。当用户询问时间、日期时使用。",
        parameters_schema={
            "type": "object",
            "properties": {},
        },
    )

    def get_time() -> str:
        return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    time_tool.set_execute_fn(get_time)
    registry.register(time_tool)

    return registry


# ============================================================================
# 第三部分：模拟 LLM 客户端（带工具调用能力）
# ============================================================================

class MockLLMWithTools:
    """
    模拟 LLM 客户端（支持工具调用）

    这个模拟客户端可以：
    1. 理解用户意图
    2. 决定是否需要调用工具
    3. 生成工具调用请求
    4. 根据工具结果生成回复
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.call_count = 0

    async def chat_with_tools(
        self,
        messages: list[Message],
    ) -> dict:
        """
        模拟 LLM 的响应（可能包含工具调用）

        Returns:
            {
                "content": str,          # 文本回复
                "tool_calls": list[dict], # 工具调用（如果有）
            }
        """
        self.call_count += 1
        last_msg = messages[-1].content if messages else ""

        # 根据用户输入决定行为
        response = self._decide_response(last_msg, messages)
        return response

    def _decide_response(self, user_input: str, history: list[Message]) -> dict:
        """根据输入决定响应（模拟 LLM 的决策过程）"""
        input_lower = user_input.lower()

        # 需要调用计算器的情况
        if any(kw in input_lower for kw in ["计算", "算一下", "+", "*", "/"]):
            # 提取表达式
            import re
            expr_match = re.search(r'[\d+\-*/().\s]+', user_input)
            expression = expr_match.group().strip() if expr_match else "2 + 2"
            return {
                "content": f"让我来计算 {expression}...",
                "tool_calls": [{
                    "id": f"call_{self.call_count}",
                    "name": "calculator",
                    "arguments": {"expression": expression},
                }],
            }

        # 需要查询天气的情况
        if "天气" in input_lower:
            # 提取城市名
            cities = ["北京", "上海", "广州", "深圳"]
            city = next((c for c in cities if c in user_input), "北京")
            return {
                "content": f"让我查一下{city}的天气...",
                "tool_calls": [{
                    "id": f"call_{self.call_count}",
                    "name": "get_weather",
                    "arguments": {"city": city},
                }],
            }

        # 需要搜索的情况
        if any(kw in input_lower for kw in ["搜索", "查找", "查一下"]):
            return {
                "content": "让我搜索一下...",
                "tool_calls": [{
                    "id": f"call_{self.call_count}",
                    "name": "web_search",
                    "arguments": {"query": user_input},
                }],
            }

        # 询问时间
        if any(kw in input_lower for kw in ["时间", "几点", "日期"]):
            return {
                "content": "让我查一下当前时间...",
                "tool_calls": [{
                    "id": f"call_{self.call_count}",
                    "name": "get_current_time",
                    "arguments": {},
                }],
            }

        # 普通对话（不需要工具）
        responses = {
            "你好": "你好！我是 AI 助手，有什么可以帮助你的？",
            "谢谢": "不客气！还有其他问题吗？",
            "再见": "再见！祝你有美好的一天！",
        }

        for key, response in responses.items():
            if key in input_lower:
                return {"content": response, "tool_calls": []}

        # 默认回复
        return {
            "content": f"我理解你的问题：「{user_input[:30]}」。"
                       f"我可以帮你计算、查天气、搜索信息等。",
            "tool_calls": [],
        }


# ============================================================================
# 第四部分：ReAct Agent 核心实现
# ============================================================================

class AgentState(Enum):
    """Agent 状态"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    RESPONDING = "responding"
    FINISHED = "finished"
    ERROR = "error"


class ReactAgent:
    """
    ReAct Agent 实现

    ReAct = Reasoning + Acting
    工作流程：
    1. Thought（思考）：分析当前情况
    2. Action（行动）：调用工具
    3. Observation（观察）：获取工具结果
    4. 重复 1-3 直到任务完成
    5. 生成最终回复

    ┌─────────────────────────────────────────────────────────────
    │                    ReAct 循环                                │
    │                                                             │
    │   ┌──────────┐    ┌──────────┐    ┌──────────────────┐     │
    │   │ Thought  │───▶│  Action  │───▶│  Observation     │     │
    │   │ (思考)   │    │  (行动)  │    │  (观察结果)       │     │
    │   └──────────    └──────────┘    └──────────────────┘     │
    │        ▲                                    │               │
    │        │                                    │               │
    │        └────────────────────────────────────┘               │
    │                    循环直到完成                               │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        llm: MockLLMWithTools,
        tool_registry: ToolRegistry,
        system_prompt: str = "你是一个有帮助的AI助手，可以使用工具来完成任务。",
        max_iterations: int = 5,
    ):
        """
        Args:
            llm: LLM 客户端
            tool_registry: 工具注册中心
            system_prompt: 系统提示词
            max_iterations: 最大迭代次数（防止无限循环）
        """
        self.llm = llm
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

        # 运行时状态
        self.state = AgentState.IDLE
        self.messages: list[Message] = []
        self.iteration = 0

    async def run(self, user_input: str) -> str:
        """
        运行 Agent 处理用户输入

        这是 Agent 的核心方法，执行完整的 ReAct 循环。

        Args:
            user_input: 用户输入

        Returns:
            Agent 的最终回复
        """
        print(f"\n{'='*50}")
        print(f"用户输入: {user_input}")
        print(f"{'='*50}")

        # 添加用户消息
        self.messages.append(Message.user(user_input))

        # ReAct 循环
        final_response = ""
        for i in range(self.max_iterations):
            self.iteration = i + 1
            print(f"\n── 第 {self.iteration} 轮 ──")

            # Step 1: Thought（思考）
            self.state = AgentState.THINKING
            print(f"  [思考中...]")

            # 调用 LLM 获取响应
            llm_response = await self.llm.chat_with_tools(self.messages)
            content = llm_response.get("content", "")
            tool_calls = llm_response.get("tool_calls", [])

            print(f"  LLM 回复: {content}")

            # Step 2: 判断是否需要调用工具
            if not tool_calls:
                # 不需要工具，直接回复
                self.state = AgentState.RESPONDING
                final_response = content
                self.messages.append(Message.assistant(content))
                print(f"  [直接回复，无需工具]")
                break

            # Step 3: Action（行动）- 执行工具调用
            self.state = AgentState.ACTING
            tool_results = []

            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                arguments = tool_call["arguments"]
                call_id = tool_call["id"]

                print(f"  [调用工具: {tool_name}]")
                print(f"    参数: {arguments}")

                # 查找并执行工具
                tool = self.tool_registry.get(tool_name)
                if tool:
                    result = await tool.execute(arguments)
                    print(f"    结果: {result}")
                    tool_results.append((call_id, result))
                else:
                    error_msg = f"未知工具: {tool_name}"
                    print(f"    错误: {error_msg}")
                    tool_results.append((call_id, error_msg))

            # Step 4: Observation（观察）- 将工具结果加入对话
            self.state = AgentState.OBSERVING
            for call_id, result in tool_results:
                tool_msg = Message.tool_result(result, call_id)
                self.messages.append(tool_msg)
                print(f"  [观察结果: {result[:50]}...]")

            # 添加 LLM 的中间回复
            self.messages.append(Message.assistant(content))

        else:
            # 达到最大迭代次数
            final_response = "抱歉，我无法完成这个任务（达到最大迭代次数）。"
            self.state = AgentState.ERROR
            print(f"\n  ⚠️ 达到最大迭代次数 ({self.max_iterations})")

        self.state = AgentState.FINISHED
        print(f"\n{'='*50}")
        print(f"最终回复: {final_response}")
        print(f"{'='*50}")

        return final_response

    def reset(self):
        """重置 Agent 状态"""
        self.state = AgentState.IDLE
        self.messages.clear()
        self.iteration = 0


# ============================================================================
# 第五部分：Agent 运行演示
# ============================================================================

async def demonstrate_simple_conversation():
    """演示简单对话（无需工具）"""
    print("\n" + "=" * 60)
    print("演示1：简单对话（无需工具）")
    print("=" * 60)

    registry = create_demo_tools()
    llm = MockLLMWithTools(registry)
    agent = ReactAgent(llm=llm, tool_registry=registry)

    # 简单问候
    await agent.run("你好")
    agent.reset()

    # 感谢
    await agent.run("谢谢")


async def demonstrate_tool_usage():
    """演示工具调用"""
    print("\n" + "=" * 60)
    print("演示2：工具调用")
    print("=" * 60)

    registry = create_demo_tools()
    llm = MockLLMWithTools(registry)
    agent = ReactAgent(llm=llm, tool_registry=registry)

    # 计算
    await agent.run("帮我计算 123 * 456")
    agent.reset()

    # 查天气
    await agent.run("北京今天天气怎么样？")
    agent.reset()

    # 搜索
    await agent.run("搜索一下什么是 AI Agent")


async def demonstrate_multi_step():
    """演示多步骤任务"""
    print("\n" + "=" * 60)
    print("演示3：多步骤任务")
    print("=" * 60)

    registry = create_demo_tools()
    llm = MockLLMWithTools(registry)
    agent = ReactAgent(llm=llm, tool_registry=registry, max_iterations=3)

    # 多步骤任务
    await agent.run("查一下上海和北京的天气，哪个更热？")


async def demonstrate_error_handling():
    """演示错误处理"""
    print("\n" + "=" * 60)
    print("演示4：错误处理")
    print("=" * 60)

    registry = create_demo_tools()
    llm = MockLLMWithTools(registry)

    # 设置很小的最大迭代次数，模拟复杂任务失败
    agent = ReactAgent(
        llm=llm,
        tool_registry=registry,
        max_iterations=1,  # 只允许1次迭代
    )

    # 复杂任务（需要多次迭代）
    await agent.run("帮我计算 100 * 200，然后查一下北京天气")


def demonstrate_tool_registry():
    """演示工具注册中心"""
    print("\n" + "=" * 60)
    print("工具注册中心演示")
    print("=" * 60)

    registry = create_demo_tools()

    # 列出所有工具
    print(registry.list_tools())

    # 获取工具 Schema（发送给 LLM）
    print("\n── 工具 Schema（发送给 LLM）──")
    schemas = registry.get_tool_schemas()
    for schema in schemas:
        func = schema["function"]
        print(f"  {func['name']}: {func['description'][:30]}...")


# ============================================================================
# 第六部分：进阶 - Agent 设计模式
# ============================================================================

def advanced_agent_patterns():
    """
    进阶：Agent 设计模式

    除了 ReAct，还有其他常用的 Agent 模式。
    """

    patterns = [
        {
            "name": "ReAct (Reasoning + Acting)",
            "description": "推理和行动交替进行",
            "workflow": "Thought → Action → Observation → 循环",
            "best_for": "通用任务，需要动态决策",
            "pros": ["灵活", "可解释", "易于调试"],
            "cons": ["可能效率低", "需要多次 LLM 调用"],
        },
        {
            "name": "Plan-and-Execute",
            "description": "先制定完整计划，再逐步执行",
            "workflow": "Plan → Execute Step 1 → Execute Step 2 → ... → Review",
            "best_for": "复杂多步骤任务",
            "pros": ["全局视野", "减少重复推理"],
            "cons": ["计划可能不准确", "不够灵活"],
        },
        {
            "name": "Self-Reflection",
            "description": "执行后自我反思和改进",
            "workflow": "Act → Reflect → Improve → Act → ...",
            "best_for": "需要高质量输出的任务",
            "pros": ["输出质量高", "能自我纠错"],
            "cons": ["速度慢", "Token 消耗大"],
        },
        {
            "name": "Tree of Thoughts",
            "description": "多路径探索，选择最优方案",
            "workflow": "生成多个方案 → 评估 → 选择最优 → 执行",
            "best_for": "创意任务、复杂决策",
            "pros": ["探索空间大", "能找到更优解"],
            "cons": ["Token 消耗非常大", "实现复杂"],
        },
    ]

    print("=" * 60)
    print("进阶：Agent 设计模式对比")
    print("=" * 60)
    for p in patterns:
        print(f"\n🔷 {p['name']}")
        print(f"   描述: {p['description']}")
        print(f"   流程: {p['workflow']}")
        print(f"   适合: {p['best_for']}")
        print(f"   优点: {', '.join(p['pros'])}")
        print(f"   缺点: {', '.join(p['cons'])}")
    print()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🤖" * 30)
    print("第七课：Agents 模块 - 核心业务逻辑")
    print("🤖" * 30)

    # 1. 工具注册中心
    demonstrate_tool_registry()

    # 2. 简单对话
    asyncio.run(demonstrate_simple_conversation())

    # 3. 工具调用
    asyncio.run(demonstrate_tool_usage())

    # 4. 多步骤任务
    asyncio.run(demonstrate_multi_step())

    # 5. 错误处理
    asyncio.run(demonstrate_error_handling())

    # 6. 进阶模式
    advanced_agent_patterns()

    print("=" * 60)
    print("✅ 第七课学习完成！")
    print("=" * 60)
    print("""
 本课要点总结：
1. 工具是 Agent 执行具体操作的能力，需要清晰的描述和参数定义
2. 工具注册中心管理所有可用工具，提供统一接口
3. ReAct 模式：思考→行动→观察→循环，直到任务完成
4. Agent 需要设置最大迭代次数防止无限循环
5. 错误处理：工具调用失败、达到最大迭代等
6. 进阶：Plan-and-Execute、Self-Reflection、Tree of Thoughts 等模式

 下一课：入口文件与运行方式 - 我们将实现 CLI 和 API 服务
    """)
