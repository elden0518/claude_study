"""
==============================================================================
第三课：Core 模块 - 系统的基石
==============================================================================

【学习目标】
- 掌握配置管理：环境变量、参数验证、默认值
- 掌握日志系统：统一格式、分级日志、日志输出
- 掌握异常处理：自定义异常、异常层次、优雅降级
- 掌握 LLM 客户端封装：统一接口、重试机制

【核心概念】
- 配置即代码（Configuration as Code）
- 防御性编程（Defensive Programming）
- 单一职责原则（Single Responsibility）

【前置知识】
- 第二课：项目架构总览

==============================================================================
"""

import os
import logging
import asyncio
from typing import Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# 第三方库
from dotenv import load_dotenv

# ============================================================================
# 第一部分：配置管理（config.py）
# ============================================================================
#
# 配置管理是系统的基础，所有模块都需要读取配置。
# 好的配置管理应该：
# 1. 从环境变量读取敏感信息（API Key 等）
# 2. 提供合理的默认值
# 3. 支持不同环境（开发/测试/生产）
# 4. 启动时验证配置完整性


class Environment(Enum):
    """运行环境枚举"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class LLMConfig:
    """
    LLM 相关配置

    使用 dataclass 而不是 dict 的好处：
    - 类型安全（IDE 可以自动补全）
    - 有默认值
    - 可以添加验证逻辑
    """
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7
    api_key: str = ""
    base_url: Optional[str] = None

    def validate(self):
        """验证配置是否完整"""
        if not self.api_key:
            raise ValueError(
                "LLM API Key 未配置！请设置 ANTHROPIC_API_KEY 环境变量"
            )
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature 必须在 0-2 之间")
        if self.max_tokens < 1:
            raise ValueError("max_tokens 必须大于 0")


@dataclass
class SystemConfig:
    """
    系统全局配置

    聚合所有子配置，提供统一的配置入口。
    """
    # 运行环境
    env: Environment = Environment.DEVELOPMENT

    # LLM 配置
    llm: LLMConfig = field(default_factory=LLMConfig)

    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Agent 配置
    max_iterations: int = 10       # Agent 最大迭代次数
    tool_timeout: int = 30         # 工具调用超时（秒）

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "SystemConfig":
        """
        从环境变量加载配置

        这是最常用的配置加载方式：
        1. 加载 .env 文件
        2. 从 os.environ 读取值
        3. 组装成配置对象
        """
        # 加载 .env 文件（如果存在）
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        # 从环境变量构建配置
        config = cls()
        config.llm.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        config.llm.model = os.getenv(
            "LLM_MODEL", "claude-sonnet-4-20250514"
        )
        config.log_level = os.getenv("LOG_LEVEL", "INFO")

        env_str = os.getenv("APP_ENV", "development")
        config.env = Environment(env_str)

        return config

    def validate(self):
        """验证所有配置"""
        self.llm.validate()
        print(f"✅ 配置验证通过 (环境: {self.env.value})")


# ============================================================================
# 第二部分：日志系统（logger.py）
# ============================================================================
#
# 日志系统对于调试和生产监控至关重要。
# 好的日志应该：
# 1. 格式统一（时间、级别、模块、消息）
# 2. 分级输出（DEBUG/INFO/WARNING/ERROR）
# 3. 支持输出到文件和控制台
# 4. 不同模块有不同的 logger 名称


def setup_logger(
    name: str = "agent",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    创建并配置 Logger

    Args:
        name: Logger 名称（通常是模块名）
        level: 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        log_file: 日志文件路径（可选）

    Returns:
        配置好的 Logger 实例

    使用示例：
        logger = setup_logger("core.config")
        logger.info("配置加载成功")
        logger.error("配置验证失败", exc_info=True)
    """

    # 创建 Logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 避免重复添加 Handler
    if logger.handlers:
        return logger

    # 定义日志格式
    formatter = logging.Formatter(
        # 格式：时间 | 级别 | 模块名 | 消息
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 Handler（总是添加）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件 Handler（可选）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 创建全局 Logger 实例
# 其他模块通过 get_logger("模块名") 获取子 Logger
_global_logger = setup_logger("agent")


def get_logger(name: str) -> logging.Logger:
    """
    获取子 Logger

    子 Logger 会继承父 Logger 的配置（级别、Handler 等）。
    这样不同模块的日志可以通过名称区分。

    使用示例：
        logger = get_logger("core.config")    # agent.core.config
        logger = get_logger("agents.react")   # agent.agents.react
    """
    return logging.getLogger(f"agent.{name}")


# ============================================================================
# 第三部分：异常处理（exceptions.py）
# ============================================================================
#
# 自定义异常层次结构的好处：
# 1. 可以针对性地捕获特定类型的异常
# 2. 携带更多上下文信息
# 3. 统一的异常处理逻辑


class AgentBaseError(Exception):
    """
    Agent 系统基础异常类

    所有自定义异常都继承自这个类。
    这样可以用 except AgentBaseError 捕获所有 Agent 相关异常。
    """

    def __init__(self, message: str, code: str = "", details: Any = None):
        """
        Args:
            message: 人类可读的错误信息
            code: 错误代码（用于程序化判断）
            details: 额外详细信息（用于调试）
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details

    def to_dict(self) -> dict:
        """将异常信息转换为字典（用于日志记录或 API 响应）"""
        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": str(self.details) if self.details else None,
        }


class LLMError(AgentBaseError):
    """
    LLM 调用相关异常

    子类化：
    - RateLimitError: 频率限制
    - AuthenticationError: 认证失败
    - ModelError: 模型错误
    """
    pass


class RateLimitError(LLMError):
    """API 频率限制"""
    def __init__(self, message: str = "API 调用频率超限", retry_after: int = 60):
        super().__init__(message, code="RATE_LIMIT")
        self.retry_after = retry_after


class AuthenticationError(LLMError):
    """认证失败（API Key 无效等）"""
    def __init__(self, message: str = "API 认证失败"):
        super().__init__(message, code="AUTH_ERROR")


class ModelError(LLMError):
    """模型返回异常"""
    def __init__(self, message: str, model: str = ""):
        super().__init__(message, code="MODEL_ERROR", details={"model": model})


class ToolError(AgentBaseError):
    """
    工具调用相关异常

    当工具执行失败时抛出。
    """
    def __init__(self, tool_name: str, message: str, result: Any = None):
        super().__init__(
            message,
            code="TOOL_ERROR",
            details={"tool": tool_name, "result": result},
        )
        self.tool_name = tool_name


class MemoryError_(AgentBaseError):
    """记忆系统异常（避免与内置 MemoryError 冲突，加下划线）"""
    pass


class SessionError(AgentBaseError):
    """会话管理异常"""
    pass


# ── 异常使用示例 ─────────────────────────────────────────────────────────

def demonstrate_exception_hierarchy():
    """演示异常层次结构和使用方式"""

    print("=" * 60)
    print("异常层次结构")
    print("=" * 60)
    print("""
    Exception
    └── AgentBaseError          # 所有 Agent 异常的基类
        ├── LLMError            # LLM 相关异常
        │   ├── RateLimitError  # 频率限制
        │   ├── AuthenticationError  # 认证失败
        │   └── ModelError      # 模型错误
        ├── ToolError           # 工具调用异常
        ├── MemoryError_        # 记忆系统异常
        └── SessionError        # 会话管理异常
    """)

    # ── 使用示例 1：针对性捕获 ───────────────────────────────────
    print("── 示例1：针对性捕获 ──")
    try:
        raise RateLimitError(retry_after=30)
    except RateLimitError as e:
        print(f"  捕获到频率限制: {e.message}, {e.retry_after}秒后重试")
    except LLMError as e:
        # 这个分支不会执行，因为 RateLimitError 已被上面的 except 捕获
        print(f"  其他 LLM 错误: {e.message}")
    except AgentBaseError as e:
        # 这个分支捕获所有 Agent 异常
        print(f"  Agent 异常: {e.message}")

    # ── 使用示例 2：统一捕获所有 Agent 异常 ──────────────────────
    print("\n── 示例2：统一捕获 ──")
    exceptions = [
        AuthenticationError(),
        ToolError("search", "搜索服务不可用"),
        SessionError("会话已过期"),
    ]
    for exc in exceptions:
        try:
            raise exc
        except AgentBaseError as e:
            print(f"  [{e.code}] {e.message}")

    # ── 使用示例 3：异常信息序列化 ───────────────────────────────
    print("\n── 示例3：异常信息序列化 ──")
    error = ToolError("calculator", "计算超时", details={"timeout": 30})
    print(f"  序列化: {error.to_dict()}")
    print()


# ============================================================================
# 第四部分：LLM 客户端封装（llm_client.py）
# ============================================================================
#
# LLM 客户端封装的目的：
# 1. 统一接口（不管底层用 Claude 还是 GPT，调用方式一样）
# 2. 内置重试机制（网络波动时自动重试）
# 3. 统一的错误处理
# 4. 支持流式输出


class BaseLLMClient:
    """
    LLM 客户端基类（抽象接口）

    定义所有 LLM 客户端必须实现的方法。
    这样上层代码不依赖具体的 LLM 实现。
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = get_logger("core.llm_client")

    async def chat(
        self,
        messages: list[dict],
        **kwargs,
    ) -> str:
        """
        发送聊天请求（同步模式）

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            **kwargs: 额外参数（temperature, max_tokens 等）

        Returns:
            AI 的文本回复
        """
        raise NotImplementedError("子类必须实现 chat 方法")

    async def chat_stream(
        self,
        messages: list[dict],
        **kwargs,
    ):
        """
        发送聊天请求（流式模式）

        Yields:
            文本片段（逐块返回）
        """
        raise NotImplementedError("子类必须实现 chat_stream 方法")


class MockLLMClient(BaseLLMClient):
    """
    模拟 LLM 客户端（用于测试和学习）

    不需要真实的 API Key，可以模拟 LLM 的响应。
    在开发和测试阶段非常有用。
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        # 使用默认配置（不需要 API Key）
        super().__init__(config or LLMConfig())
        self.call_count = 0  # 记录调用次数（用于测试）

    async def chat(
        self,
        messages: list[dict],
        **kwargs,
    ) -> str:
        """模拟 LLM 响应"""
        self.call_count += 1
        last_message = messages[-1]["content"] if messages else ""

        self.logger.info(f"MockLLM 收到消息: {last_message[:50]}...")

        # 根据关键词返回不同的模拟响应
        if "天气" in last_message:
            return "北京今天晴天，气温28°C，适合出行。"
        elif "计算" in last_message or "+" in last_message or "*" in last_message:
            return "计算结果是 42。"
        elif "搜索" in last_message or "查" in last_message:
            return "搜索到以下结果：\n1. 第一条相关信息\n2. 第二条相关信息"
        else:
            return f"我收到了你的消息：「{last_message[:30]}」。这是一个模拟回复。"

    async def chat_stream(self, messages: list[dict], **kwargs):
        """模拟流式响应"""
        response = await self.chat(messages, **kwargs)
        # 逐字返回，模拟流式效果
        for char in response:
            yield char
            await asyncio.sleep(0.02)  # 模拟网络延迟


class ClaudeLLMClient(BaseLLMClient):
    """
    Claude (Anthropic) LLM 客户端

    封装 Anthropic API 的调用逻辑。
    实际使用时需要有效的 ANTHROPIC_API_KEY。
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None  # 延迟初始化

    def _get_client(self):
        """延迟创建客户端（避免导入时失败）"""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(
                    api_key=self.config.api_key,
                )
            except ImportError:
                raise AgentBaseError(
                    "anthropic 库未安装，请运行: pip install anthropic",
                    code="IMPORT_ERROR",
                )
        return self._client

    async def chat(
        self,
        messages: list[dict],
        **kwargs,
    ) -> str:
        """调用 Claude API"""
        client = self._get_client()

        # 转换消息格式（Anthropic 格式）
        system_message = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                chat_messages.append(msg)

        # 合并参数
        params = {
            "model": self.config.model,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": chat_messages,
        }
        if system_message:
            params["system"] = system_message

        try:
            response = await client.messages.create(**params)
            return response.content[0].text

        except Exception as e:
            self.logger.error(f"Claude API 调用失败: {e}")
            raise LLMError(f"LLM 调用失败: {e}", code="API_ERROR")


# ── LLM 客户端使用演示 ─────────────────────────────────────────────────

async def demonstrate_llm_client():
    """演示 LLM 客户端的使用"""

    print("=" * 60)
    print("LLM 客户端演示")
    print("=" * 60)

    # ── 使用 Mock 客户端（不需要 API Key）─────────────────────────
    print("\n── Mock LLM 客户端 ──")
    mock_client = MockLLMClient()

    # 普通对话
    response = await mock_client.chat([
        {"role": "user", "content": "北京今天天气怎么样？"}
    ])
    print(f"  用户: 北京今天天气怎么样？")
    print(f"  AI:   {response}")

    # 流式对话
    print("\n── 流式输出演示 ──")
    print("  用户: 请告诉我一个有趣的事实")
    print("  AI:   ", end="")
    async for chunk in mock_client.chat_stream([
        {"role": "user", "content": "请告诉我一个有趣的事实"}
    ]):
        print(chunk, end="", flush=True)
    print()  # 换行

    print(f"\n  总调用次数: {mock_client.call_count}")
    print()


# ============================================================================
# 第五部分：配置 + 日志 + 异常 综合演示
# ============================================================================

def demonstrate_core_integration():
    """综合演示 Core 模块的三个组件如何协同工作"""

    print("=" * 60)
    print("Core 模块综合演示")
    print("=" * 60)

    # ── 1. 配置加载 ──────────────────────────────────────────────
    print("\n── 1. 配置加载 ──")
    try:
        config = SystemConfig.from_env()
        # 注意：如果没有设置 API Key，验证会失败
        # config.validate()
        print(f"  模型: {config.llm.model}")
        print(f"  环境: {config.env.value}")
        print(f"  最大迭代: {config.max_iterations}")
    except Exception as e:
        print(f"  配置加载: {e}")

    # ── 2. 日志系统 ─────────────────────────────────────────────
    print("\n── 2. 日志系统 ──")
    logger = get_logger("demo")
    logger.debug("这是 DEBUG 级别日志（默认不显示）")
    logger.info("这是 INFO 级别日志（系统正常运行信息）")
    logger.warning("这是 WARNING 级别日志（需要注意但不影响运行）")

    # ── 3. 异常处理 ──────────────────────────────────────────────
    print("\n── 3. 异常处理 ──")

    # 模拟一个完整的错误处理流程
    def simulate_tool_call(tool_name: str):
        """模拟工具调用，可能抛出异常"""
        if tool_name == "unavailable":
            raise ToolError(tool_name, "服务不可用", details={"status": 503})
        elif tool_name == "timeout":
            raise ToolError(tool_name, "调用超时", details={"timeout": 30})
        else:
            logger.info(f"工具 {tool_name} 调用成功")
            return {"result": "success"}

    # 使用 try-except 处理可能的异常
    tools_to_test = ["search", "unavailable", "calculator"]
    for tool in tools_to_test:
        try:
            result = simulate_tool_call(tool)
            print(f"  ✅ {tool}: {result}")
        except ToolError as e:
            logger.error(f"工具调用失败: {e.message}")
            print(f"  ❌ {tool}: {e.message} (代码: {e.code})")
    print()


# ============================================================================
# 第六部分：进阶 - 带重试机制的 LLM 客户端
# ============================================================================

async def demonstrate_retry_mechanism():
    """
    进阶：使用 tenacity 库实现自动重试

    在生产环境中，API 调用可能因为网络波动、频率限制等原因失败。
    自动重试机制可以提高系统的可靠性。
    """

    print("=" * 60)
    print("进阶：重试机制演示")
    print("=" * 60)

    # 注意：这里用模拟方式演示，实际使用 tenacity 装饰器
    # from tenacity import retry, stop_after_attempt, wait_exponential

    call_attempts = 0
    max_attempts = 3

    async def unreliable_api_call():
        """模拟一个不稳定的 API 调用"""
        nonlocal call_attempts
        call_attempts += 1
        print(f"  第 {call_attempts} 次尝试...", end=" ")

        if call_attempts < 3:
            print(" 失败（模拟网络错误）")
            raise LLMError("网络超时", code="TIMEOUT")
        else:
            print("✅ 成功！")
            return "API 响应数据"

    # 手动实现重试逻辑（理解原理）
    print("\n── 手动重试逻辑 ──")
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = await unreliable_api_call()
            print(f"  结果: {result}")
            break
        except LLMError as e:
            last_error = e
            if attempt < max_attempts:
                wait_time = 2 ** attempt  # 指数退避：2s, 4s, 8s...
                print(f"  等待 {wait_time} 秒后重试...")
            else:
                print(f"  已达最大重试次数，放弃。错误: {e.message}")

    print()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "⚙️" * 30)
    print("第三课：Core 模块 - 系统的基石")
    print("⚙️" * 30 + "\n")

    # 1. 配置管理演示
    print("📋 配置管理")
    print("-" * 60)
    config = SystemConfig()
    config.llm.api_key = "sk-test-demo-key"  # 模拟设置 API Key
    config.validate()
    print()

    # 2. 日志系统演示
    print("📝 日志系统")
    print("-" * 60)
    demo_logger = get_logger("main")
    demo_logger.info("Core 模块演示开始")
    print()

    # 3. 异常层次演示
    demonstrate_exception_hierarchy()

    # 4. LLM 客户端演示
    asyncio.run(demonstrate_llm_client())

    # 5. 综合演示
    demonstrate_core_integration()

    # 6. 重试机制（进阶）
    asyncio.run(demonstrate_retry_mechanism())

    print("=" * 60)
    print("✅ 第三课学习完成！")
    print("=" * 60)
    print("""
 本课要点总结：
1. 配置管理：使用 dataclass + 环境变量，提供默认值和验证
2. 日志系统：统一格式、分级输出、模块化 Logger
3. 异常处理：层次化设计，携带上下文信息，可序列化
4. LLM 客户端：统一接口、Mock 实现、重试机制
5. Core 模块是所有其他模块的基础，必须稳定可靠

 下一课：Schema 模块 - 我们将定义 Agent 系统的数据结构
    """)
