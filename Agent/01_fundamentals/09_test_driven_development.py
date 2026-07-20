"""
==============================================================================
第九课：测试驱动开发
==============================================================================

【学习目标】
- 理解测试驱动开发（TDD）在 Agent 项目中的应用
- 掌握单元测试编写方法
- 学会 Mock LLM 进行测试
- 掌握集成测试和端到端测试
- 学会测试 Agent 的边界情况

【核心概念】
- TDD 流程：Red → Green → Refactor
- Mock 对象：模拟外部依赖
- 单元测试 vs 集成测试
- 测试覆盖率

【前置知识】
- 所有前序课程

==============================================================================
"""

import asyncio
import json
from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock, AsyncMock, patch

from pydantic import BaseModel, Field
from enum import Enum


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
    tool_call_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# 第一部分：为什么 Agent 项目需要测试？
# ============================================================================

def why_testing_matters():
    """
    解释为什么 Agent 项目特别需要测试

    Agent 项目的测试挑战：
    1. LLM 输出不确定（同样的输入可能有不同的输出）
    2. 工具调用涉及外部服务
    3. 多轮对话的状态管理复杂
    4. 并发场景下的会话隔离

    测试的好处：
    1. 确保代码修改不会破坏现有功能
    2. 文档化预期行为
    3. 提高代码质量
    4. 支持重构
    """

    challenges = [
        {
            "challenge": "LLM 输出不确定性",
            "solution": "使用 Mock LLM，控制输出",
            "example": "MockLLM 返回固定响应，测试 Agent 逻辑",
        },
        {
            "challenge": "外部服务依赖",
            "solution": "Mock 工具调用",
            "example": "Mock 天气 API，测试工具调用逻辑",
        },
        {
            "challenge": "状态管理复杂",
            "solution": "单元测试每个状态转换",
            "example": "测试 Session 从 ACTIVE 到 EXPIRED 的转换",
        },
        {
            "challenge": "并发问题",
            "solution": "异步测试 + 压力测试",
            "example": "同时创建多个会话，验证隔离性",
        },
    ]

    print("=" * 60)
    print("Agent 项目测试的挑战与解决方案")
    print("=" * 60)
    for c in challenges:
        print(f"\n  挑战: {c['challenge']}")
        print(f"  方案: {c['solution']}")
        print(f"  示例: {c['example']}")
    print()


# ============================================================================
# 第二部分：TDD 流程（Red → Green → Refactor）
# ============================================================================

def explain_tdd_flow():
    """
    解释 TDD 流程

    TDD（Test-Driven Development）流程：
    1. Red：编写一个失败的测试
    2. Green：编写最少的代码让测试通过
    3. Refactor：重构代码，保持测试通过

    这个流程确保：
    - 每个功能都有测试覆盖
    - 代码只实现必要的功能
    - 重构时有测试保护
    """

    # 示例：开发一个消息计数器
    print("=" * 60)
    print("TDD 流程示例：消息计数器")
    print("=" * 60)

    print("""
  ── Step 1: Red（编写失败的测试）──

  # test_message_counter.py
  def test_count_messages():
      counter = MessageCounter()
      counter.add("user", "你好")
      counter.add("assistant", "你好！")
      assert counter.user_count == 1
      assert counter.assistant_count == 1
      assert counter.total == 2

  # 运行测试 → FAIL（因为 MessageCounter 还不存在）


  ── Step 2: Green（编写最少代码让测试通过）──

  class MessageCounter:
      def __init__(self):
          self.user_count = 0
          self.assistant_count = 0

      def add(self, role: str, content: str):
          if role == "user":
              self.user_count += 1
          elif role == "assistant":
              self.assistant_count += 1

      @property
      def total(self):
          return self.user_count + self.assistant_count

  # 运行测试 → PASS


  ── Step 3: Refactor（重构代码）──

  class MessageCounter:
      def __init__(self):
          self._counts = {"user": 0, "assistant": 0}

      def add(self, role: str, content: str):
          if role in self._counts:
              self._counts[role] += 1

      @property
      def user_count(self):
          return self._counts["user"]

      @property
      def assistant_count(self):
          return self._counts["assistant"]

      @property
      def total(self):
          return sum(self._counts.values())

  # 运行测试 → 仍然 PASS（重构没有破坏功能）
    """)


# ============================================================================
# 第三部分：单元测试示例
# ============================================================================

# ── 被测代码 ─────────────────────────────────────────────────────

class MessageCounter:
    """消息计数器（被测对象）"""

    def __init__(self):
        self._counts: dict[str, int] = {"user": 0, "assistant": 0, "system": 0}

    def add(self, role: str, content: str):
        if role in self._counts:
            self._counts[role] += 1

    @property
    def user_count(self) -> int:
        return self._counts["user"]

    @property
    def assistant_count(self) -> int:
        return self._counts["assistant"]

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def reset(self):
        for key in self._counts:
            self._counts[key] = 0


class SimpleTool:
    """简单工具（被测对象）"""

    def __init__(self, name: str, execute_fn):
        self.name = name
        self._execute_fn = execute_fn

    async def execute(self, arguments: dict) -> str:
        try:
            result = self._execute_fn(**arguments)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)
        except Exception as e:
            return f"错误: {e}"


# ── 测试代码 ─────────────────────────────────────────────────────

def test_message_counter():
    """测试消息计数器"""
    print("=" * 60)
    print("单元测试：消息计数器")
    print("=" * 60)

    counter = MessageCounter()

    # 测试1：初始状态
    assert counter.user_count == 0, "初始用户计数应为 0"
    assert counter.assistant_count == 0, "初始 AI 计数应为 0"
    assert counter.total == 0, "初始总数应为 0"
    print("  ✅ 测试1：初始状态")

    # 测试2：添加消息
    counter.add("user", "你好")
    assert counter.user_count == 1
    assert counter.total == 1
    print("  ✅ 测试2：添加用户消息")

    counter.add("assistant", "你好！")
    assert counter.assistant_count == 1
    assert counter.total == 2
    print("  ✅ 测试3：添加 AI 消息")

    # 测试3：多次添加
    counter.add("user", "天气怎么样")
    counter.add("user", "谢谢")
    assert counter.user_count == 3
    assert counter.total == 4
    print("  ✅ 测试4：多次添加")

    # 测试4：重置
    counter.reset()
    assert counter.total == 0
    print("  ✅ 测试5：重置")

    # 测试5：未知角色
    counter.add("unknown", "测试")
    assert counter.total == 0  # 未知角色不应计数
    print("  ✅ 测试6：未知角色不计数")

    print()


async def test_tool_execution():
    """测试工具执行"""
    print("=" * 60)
    print("单元测试：工具执行")
    print("=" * 60)

    # 测试1：正常执行
    def add_numbers(a: int, b: int) -> int:
        return a + b

    tool = SimpleTool("adder", add_numbers)
    result = await tool.execute({"a": 3, "b": 5})
    assert result == "8", f"期望 '8'，得到 '{result}'"
    print(f"  ✅ 测试1：正常执行 - 3 + 5 = {result}")

    # 测试2：错误处理
    def divide(a: int, b: int) -> float:
        return a / b

    div_tool = SimpleTool("divider", divide)
    result = await div_tool.execute({"a": 10, "b": 0})
    assert "错误" in result, f"期望错误信息，得到 '{result}'"
    print(f"  ✅ 测试2：除零错误处理 - {result}")

    # 测试3：异步函数
    async def async_fetch(url: str) -> str:
        return f"Content from {url}"

    async_tool = SimpleTool("fetcher", async_fetch)
    result = await async_tool.execute({"url": "https://example.com"})
    assert "example.com" in result
    print(f"  ✅ 测试3：异步函数 - {result}")

    print()


# ============================================================================
# 第四部分：Mock LLM 测试
# ============================================================================

class MockLLM:
    """
    Mock LLM 客户端

    用于测试，返回预定义的响应。
    可以记录调用历史，验证 Agent 是否正确调用 LLM。
    """

    def __init__(self, responses: Optional[list[str]] = None):
        """
        Args:
            responses: 预定义的响应列表（按顺序返回）
        """
        self.responses = responses or ["默认回复"]
        self.call_count = 0
        self.call_history: list[dict] = []

    async def chat(self, messages: list[Message], **kwargs) -> str:
        """模拟 LLM 调用"""
        self.call_count += 1
        self.call_history.append({
            "messages": messages,
            "kwargs": kwargs,
            "timestamp": datetime.now(),
        })

        # 按顺序返回响应
        response_idx = (self.call_count - 1) % len(self.responses)
        return self.responses[response_idx]

    def get_last_call(self) -> Optional[dict]:
        """获取最后一次调用信息"""
        return self.call_history[-1] if self.call_history else None

    def reset(self):
        """重置状态"""
        self.call_count = 0
        self.call_history.clear()


def test_mock_llm():
    """测试 Mock LLM"""
    print("=" * 60)
    print("Mock LLM 测试")
    print("=" * 60)

    # 创建 Mock LLM
    mock_llm = MockLLM(responses=["回复1", "回复2", "回复3"])

    # 测试1：第一次调用
    response = asyncio.run(mock_llm.chat([Message.user("你好")]))
    assert response == "回复1"
    assert mock_llm.call_count == 1
    print(f"  ✅ 测试1：第一次调用返回 '{response}'")

    # 测试2：第二次调用
    response = asyncio.run(mock_llm.chat([Message.user("继续")]))
    assert response == "回复2"
    assert mock_llm.call_count == 2
    print(f"  ✅ 测试2：第二次调用返回 '{response}'")

    # 测试3：调用历史
    last_call = mock_llm.get_last_call()
    assert last_call is not None
    assert len(last_call["messages"]) == 1
    print(f"  ✅ 测试3：调用历史记录正确")

    # 测试4：重置
    mock_llm.reset()
    assert mock_llm.call_count == 0
    assert len(mock_llm.call_history) == 0
    print(f"  ✅ 测试4：重置后状态清空")

    print()


# ============================================================================
# 第五部分：集成测试示例
# ============================================================================

class SimpleAgent:
    """简单 Agent（用于集成测试）"""

    def __init__(self, llm: MockLLM):
        self.llm = llm
        self.messages: list[Message] = []

    async def run(self, user_input: str) -> str:
        """运行 Agent"""
        self.messages.append(Message.user(user_input))
        response = await self.llm.chat(self.messages)
        self.messages.append(Message.assistant(response))
        return response


def test_agent_integration():
    """测试 Agent 集成"""
    print("=" * 60)
    print("集成测试：Agent + Mock LLM")
    print("=" * 60)

    # 创建 Mock LLM 和 Agent
    mock_llm = MockLLM(responses=[
        "你好！有什么可以帮助你的？",
        "北京今天晴天，28°C。",
        "不客气！还有其他问题吗？",
    ])
    agent = SimpleAgent(llm=mock_llm)

    # 测试1：第一轮对话
    response = asyncio.run(agent.run("你好"))
    assert response == "你好！有什么可以帮助你的？"
    assert len(agent.messages) == 2  # 用户 + AI
    print(f"  ✅ 测试1：第一轮对话")
    print(f"     用户: 你好")
    print(f"     AI:   {response}")

    # 测试2：第二轮对话（有上下文）
    response = asyncio.run(agent.run("北京天气怎么样？"))
    assert response == "北京今天晴天，28°C。"
    assert len(agent.messages) == 4  # 2轮对话
    print(f"  ✅ 测试2：第二轮对话（有上下文）")
    print(f"     用户: 北京天气怎么样？")
    print(f"     AI:   {response}")

    # 测试3：验证 LLM 收到了完整历史
    last_call = mock_llm.get_last_call()
    assert len(last_call["messages"]) == 3  # 用户 + AI + 用户
    print(f"  ✅ 测试3：LLM 收到完整历史 ({len(last_call['messages'])} 条)")

    # 测试4：调用次数
    assert mock_llm.call_count == 2
    print(f"  ✅ 测试4：LLM 调用次数 = {mock_llm.call_count}")

    print()


# ============================================================================
# 第六部分：边界情况测试
# ============================================================================

def test_edge_cases():
    """测试边界情况"""
    print("=" * 60)
    print("边界情况测试")
    print("=" * 60)

    counter = MessageCounter()

    # 测试1：空输入
    counter.add("user", "")
    assert counter.user_count == 1  # 空字符串也应计数
    print("  ✅ 测试1：空字符串消息")

    # 测试2：超长消息
    long_msg = "a" * 100000
    counter.add("user", long_msg)
    assert counter.user_count == 2
    print("  ✅ 测试2：超长消息")

    # 测试3：特殊字符
    special_msg = "!@#$%^&*()_+{}|:<>?"
    counter.add("user", special_msg)
    assert counter.user_count == 3
    print("  ✅ 测试3：特殊字符消息")

    # 测试4：Unicode 字符
    unicode_msg = "你好 🌍 🚀 🤖"
    counter.add("user", unicode_msg)
    assert counter.user_count == 4
    print("  ✅ 测试4：Unicode 字符消息")

    # 测试5：大量消息
    counter.reset()
    for i in range(1000):
        counter.add("user", f"消息 {i}")
    assert counter.user_count == 1000
    assert counter.total == 1000
    print("  ✅ 测试5：大量消息 (1000条)")

    print()


# ============================================================================
# 第七部分：使用 pytest 风格的测试
# ============================================================================

def demonstrate_pytest_style():
    """
    演示 pytest 风格的测试

    实际项目中使用 pytest 运行测试：
    pytest tests/ -v --cov=agent --cov-report=html
    """

    print("=" * 60)
    print("pytest 风格测试示例")
    print("=" * 60)

    test_code = '''
    # tests/test_agent.py

    import pytest
    from agent.core.llm_client import MockLLMClient
    from agent.agents.react_agent import ReactAgent
    from agent.tools.registry import ToolRegistry


    @pytest.fixture
    def mock_llm():
        """创建 Mock LLM 的 fixture"""
        return MockLLMClient(responses=["测试回复"])


    @pytest.fixture
    def tool_registry():
        """创建工具注册中心的 fixture"""
        registry = ToolRegistry()
        # 注册测试工具
        registry.register(create_test_tool())
        return registry


    @pytest.fixture
    def agent(mock_llm, tool_registry):
        """创建 Agent 的 fixture"""
        return ReactAgent(
            llm=mock_llm,
            tool_registry=tool_registry,
            max_iterations=3,
        )


    class TestReactAgent:
        """Agent 测试类"""

        def test_simple_conversation(self, agent):
            """测试简单对话"""
            response = agent.run("你好")
            assert response is not None
            assert len(response) > 0

        def test_tool_calling(self, agent):
            """测试工具调用"""
            response = agent.run("计算 2 + 2")
            assert "4" in response or "计算" in response

        def test_max_iterations(self, agent):
            """测试最大迭代次数"""
            agent.max_iterations = 1
            # 复杂任务应该在规定时间内停止
            response = agent.run("复杂任务...")
            assert response is not None

        @pytest.mark.asyncio
        async def test_concurrent_sessions(self, agent):
            """测试并发会话"""
            tasks = [
                agent.run(f"用户{i}的问题")
                for i in range(10)
            ]
            results = await asyncio.gather(*tasks)
            assert len(results) == 10
    '''

    print(test_code)
    print("\n运行方式:")
    print("  pytest tests/ -v          # 运行所有测试")
    print("  pytest tests/ -k test_tool  # 运行特定测试")
    print("  pytest --cov=agent        # 生成覆盖率报告")
    print()


# ============================================================================
# 第八部分：测试最佳实践
# ============================================================================

def testing_best_practices():
    """测试最佳实践"""

    print("=" * 60)
    print("测试最佳实践")
    print("=" * 60)

    practices = [
        {
            "practice": "AAA 模式",
            "description": "Arrange（准备）→ Act（执行）→ Assert（断言）",
            "example": """
    def test_add_message():
        # Arrange（准备）
        counter = MessageCounter()

        # Act（执行）
        counter.add("user", "你好")

        # Assert（断言）
        assert counter.user_count == 1
            """,
        },
        {
            "practice": "测试命名规范",
            "description": "test_模块_功能_场景",
            "examples": [
                "test_agent_run_simple_conversation",
                "test_agent_run_with_tool_call",
                "test_agent_run_max_iterations_exceeded",
            ],
        },
        {
            "practice": "Mock 外部依赖",
            "description": "不依赖真实的 LLM API 或外部服务",
            "tools": [
                "unittest.mock.MagicMock",
                "unittest.mock.AsyncMock",
                "pytest-mock",
            ],
        },
        {
            "practice": "测试覆盖率目标",
            "description": "核心逻辑 80%+，工具层 90%+",
            "levels": {
                "核心逻辑": "80%+",
                "工具层": "90%+",
                "集成测试": "关键路径 100%",
            },
        },
        {
            "practice": "CI/CD 集成",
            "description": "每次提交自动运行测试",
            "steps": [
                "代码提交触发 CI",
                "安装依赖",
                "运行单元测试",
                "运行集成测试",
                "生成覆盖率报告",
                "失败时阻止合并",
            ],
        },
    ]

    for p in practices:
        print(f"\n🔷 {p['practice']}")
        print(f"   说明: {p['description']}")
        if "example" in p:
            print(f"   示例: {p['example']}")
        if "examples" in p:
            for ex in p["examples"]:
                print(f"   - {ex}")
        if "tools" in p:
            print(f"   工具: {', '.join(p['tools'])}")
        if "levels" in p:
            for level, target in p["levels"].items():
                print(f"   - {level}: {target}")
        if "steps" in p:
            for i, step in enumerate(p["steps"], 1):
                print(f"   {i}. {step}")
    print()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "" * 30)
    print("第九课：测试驱动开发")
    print("" * 30 + "\n")

    # 1. 为什么需要测试
    why_testing_matters()

    # 2. TDD 流程
    explain_tdd_flow()

    # 3. 单元测试
    test_message_counter()
    asyncio.run(test_tool_execution())

    # 4. Mock LLM
    test_mock_llm()

    # 5. 集成测试
    test_agent_integration()

    # 6. 边界情况
    test_edge_cases()

    # 7. pytest 风格
    demonstrate_pytest_style()

    # 8. 最佳实践
    testing_best_practices()

    print("=" * 60)
    print("✅ 第九课学习完成！")
    print("=" * 60)
    print("""
 本课要点总结：
1. TDD 流程：Red → Green → Refactor
2. 单元测试：测试单个组件的功能
3. Mock LLM：控制 LLM 输出，确保测试可重复
4. 集成测试：测试组件之间的协作
5. 边界情况：空输入、超长消息、特殊字符等
6. 最佳实践：AAA 模式、命名规范、覆盖率目标

 下一课：生产环境部署 - 我们将学习 Docker 部署和监控
    """)
