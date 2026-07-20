"""
==============================================================================
第四课：Schema 模块 - 数据结构的艺术
==============================================================================

【学习目标】
- 掌握 Pydantic 数据模型定义
- 理解 Agent 系统中的核心数据结构
- 学会定义消息格式、工具 Schema、Agent 状态
- 掌握数据验证和序列化

【核心概念】
- Pydantic 模型（类型安全的数据结构）
- 消息格式（Message Protocol）
- 工具 Schema（Tool Definition）
- 状态管理（State Management）

【前置知识】
- 第三课：Core 模块（配置、日志、异常）

==============================================================================
"""

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Literal

# Pydantic 是 Python 最流行的数据验证库
# 它提供了：类型检查、数据验证、序列化/反序列化、文档生成
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# 第一部分：为什么需要 Schema 模块？
# ============================================================================

def why_schema_matters():
    """
    解释为什么 Agent 系统需要严格的数据结构定义

    没有 Schema 的问题：
    - 数据格式不统一，模块之间传递数据容易出错
    - 无法在编译时发现类型错误
    - LLM 返回的数据无法验证
    - 工具参数没有约束，容易传错

    有 Schema 的好处：
    - 类型安全：IDE 自动补全，编译时检查
    - 自动验证：数据不符合格式时立即报错
    - 自动序列化：轻松转换为 JSON
    - 文档生成：Schema 本身就是文档
    """

    # ─ 反面示例：用 dict 传递数据 ───────────────────────────────
    bad_example = """
    # ❌ 使用 dict，没有类型约束
    message = {
        "role": "user",
        "content": "你好",
        "timestamp": "2024-01-01",  # 字符串还是 datetime？
        "metadata": {...}           # 结构不确定
    }

    # 问题：
    # 1. 拼写错误不会报错：message["roel"] = "user"
    # 2. 类型不确定：timestamp 是字符串还是 datetime？
    # 3. 没有默认值：每次都要手动设置所有字段
    # 4. 无法验证：content 为空时不会报错
    """

    # ── 正面示例：用 Pydantic 模型 ────────────────────────────────
    good_example = """
    # ✅ 使用 Pydantic 模型，有类型约束和验证
    class Message(BaseModel):
        role: Literal["user", "assistant", "system"]
        content: str = Field(min_length=1)  # 不能为空
        timestamp: datetime = Field(default_factory=datetime.now)
        metadata: dict = Field(default_factory=dict)

    # 好处：
    # 1. 拼写错误会立即报错：message.roel = "user"  # AttributeError
    # 2. 类型明确：timestamp 一定是 datetime 对象
    # 3. 有默认值：timestamp 和 metadata 自动设置
    # 4. 自动验证：content="" 会抛出 ValidationError
    """

    print("=" * 60)
    print("为什么需要 Schema 模块")
    print("=" * 60)
    print(f"{'特性':<20} {'❌ Dict':<18} {'✅ Pydantic':<18}")
    print("-" * 60)
    print(f"{'类型安全':<20} {'':<18} {'✅':<18}")
    print(f"{'自动验证':<20} {'❌':<18} {'✅':<18}")
    print(f"{'IDE 支持':<20} {'❌':<18} {'✅':<18}")
    print(f"{'序列化':<20} {'手动':<18} {'自动':<18}")
    print(f"{'文档生成':<20} {'❌':<18} {'✅':<18}")
    print()


# ============================================================================
# 第二部分：消息格式定义（message.py）
# ============================================================================
#
# 消息是 Agent 系统中最基本的数据单元。
# 所有模块（LLM、Memory、Session）都围绕消息工作。


class Role(Enum):
    """消息角色枚举"""
    SYSTEM = "system"       # 系统消息（设定 Agent 行为）
    USER = "user"           # 用户消息
    ASSISTANT = "assistant" # AI 回复
    TOOL = "tool"           # 工具调用结果


class Message(BaseModel):
    """
    标准消息模型

    这是 Agent 系统中最核心的数据结构。
    所有对话、工具调用、系统提示都通过 Message 传递。

    字段说明：
    - role: 消息角色（system/user/assistant/tool）
    - content: 消息内容（文本）
    - tool_calls: 工具调用信息（仅 assistant 角色使用）
    - tool_call_id: 工具调用 ID（仅 tool 角色使用）
    - metadata: 额外元数据
    - timestamp: 消息创建时间
    - id: 唯一标识符
    """

    role: Role = Field(description="消息角色")
    content: str = Field(default="", description="消息内容")

    # 工具调用相关字段（仅特定角色使用）
    tool_calls: Optional[list[dict]] = Field(
        default=None,
        description="工具调用列表（assistant 角色）",
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="工具调用 ID（tool 角色，关联到具体的 tool_call）",
    )

    # 元数据
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="额外元数据",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="消息创建时间",
    )
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="消息唯一 ID",
    )

    # ── 便捷方法 ─────────────────────────────────────────────────

    @classmethod
    def user(cls, content: str, **kwargs) -> "Message":
        """快速创建用户消息"""
        return cls(role=Role.USER, content=content, **kwargs)

    @classmethod
    def assistant(cls, content: str, **kwargs) -> "Message":
        """快速创建 AI 回复消息"""
        return cls(role=Role.ASSISTANT, content=content, **kwargs)

    @classmethod
    def system(cls, content: str, **kwargs) -> "Message":
        """快速创建系统消息"""
        return cls(role=Role.SYSTEM, content=content, **kwargs)

    @classmethod
    def tool_result(
        cls, content: str, tool_call_id: str, **kwargs
    ) -> "Message":
        """快速创建工具结果消息"""
        return cls(
            role=Role.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            **kwargs,
        )

    def to_llm_format(self) -> dict:
        """
        转换为 LLM API 需要的格式

        不同的 LLM API 需要不同的消息格式，
        这个方法提供统一的转换接口。
        """
        msg = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    def to_dict(self) -> dict:
        """转换为普通字典（用于序列化）"""
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# ============================================================================
# 第三部分：工具 Schema 定义（tool.py）
# ============================================================================
#
# 工具 Schema 定义了 Agent 可以调用的工具。
# LLM 通过工具的描述和参数来决定使用哪个工具。


class ToolParameter(BaseModel):
    """
    工具参数定义

    描述工具的一个参数，包括名称、类型、描述等。
    """
    name: str = Field(description="参数名称")
    type: str = Field(description="参数类型 (string/integer/number/boolean)")
    description: str = Field(description="参数描述（LLM 靠这个理解参数用途）")
    required: bool = Field(default=False, description="是否必填")
    default: Any = Field(default=None, description="默认值")
    enum: Optional[list[str]] = Field(
        default=None,
        description="可选值列表（用于枚举类型参数）",
    )


class ToolSchema(BaseModel):
    """
    工具 Schema 定义

    完整描述一个工具，包括名称、功能描述、参数列表。
    这个 Schema 会被发送给 LLM，让 LLM 理解工具的能力。

    重要：工具的描述（description）必须清晰准确，
    因为 LLM 完全依赖描述来决定是否使用这个工具。
    """

    name: str = Field(description="工具名称（英文，无空格）")
    description: str = Field(
        description="工具功能描述（中文，清晰说明用途和使用场景）"
    )
    parameters: list[ToolParameter] = Field(
        default_factory=list,
        description="参数列表",
    )

    # ── 便捷方法 ─────────────────────────────────────────────────

    def to_openai_format(self) -> dict:
        """转换为 OpenAI function calling 格式"""
        properties = {}
        required = []
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_format(self) -> dict:
        """转换为 Anthropic tool use 格式"""
        properties = {}
        required = []
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


# ── 预定义常用工具 ─────────────────────────────────────────────────

# 搜索工具
SEARCH_TOOL = ToolSchema(
    name="web_search",
    description="搜索互联网获取最新信息。当用户询问实时信息、新闻、"
                "或需要查找特定资料时使用此工具。",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="搜索关键词",
            required=True,
        ),
        ToolParameter(
            name="num_results",
            type="integer",
            description="返回结果数量",
            required=False,
            default="5",
        ),
    ],
)

# 计算工具
CALCULATOR_TOOL = ToolSchema(
    name="calculator",
    description="执行数学计算。支持加减乘除、幂运算、开方等。"
                "当用户需要进行数值计算时使用。",
    parameters=[
        ToolParameter(
            name="expression",
            type="string",
            description="数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'",
            required=True,
        ),
    ],
)

# 天气查询工具
WEATHER_TOOL = ToolSchema(
    name="get_weather",
    description="查询指定城市的当前天气信息，包括温度、天气状况、湿度等。",
    parameters=[
        ToolParameter(
            name="city",
            type="string",
            description="城市名称，如 '北京'、'上海'",
            required=True,
        ),
        ToolParameter(
            name="unit",
            type="string",
            description="温度单位",
            required=False,
            default="celsius",
            enum=["celsius", "fahrenheit"],
        ),
    ],
)


# ============================================================================
# 第四部分：Agent 状态定义（state.py）
# ============================================================================
#
# Agent 状态记录了 Agent 运行时的所有信息。
# 每次循环（Thought → Action → Observation）都会更新状态。


class AgentState(Enum):
    """Agent 运行状态"""
    IDLE = "idle"               # 空闲，等待输入
    THINKING = "thinking"       # 正在思考（LLM 推理中）
    ACTING = "acting"           # 正在执行工具调用
    OBSERVING = "observing"     # 正在处理工具返回结果
    RESPONDING = "responding"   # 正在生成回复
    ERROR = "error"             # 出错
    FINISHED = "finished"       # 任务完成


class ToolCallRecord(BaseModel):
    """工具调用记录"""
    tool_name: str = Field(description="工具名称")
    arguments: dict = Field(description="调用参数")
    result: Optional[str] = Field(default=None, description="执行结果")
    success: bool = Field(default=False, description="是否成功")
    timestamp: datetime = Field(default_factory=datetime.now)


class AgentStateModel(BaseModel):
    """
    Agent 运行时状态

    记录了 Agent 的完整运行状态，包括：
    - 当前状态（思考/行动/完成等）
    - 对话历史
    - 工具调用记录
    - 迭代次数
    - 最终输出

    这个状态对象在 Agent 的每次循环中都会被更新。
    """

    # 基本状态
    state: AgentState = Field(
        default=AgentState.IDLE,
        description="当前运行状态",
    )
    iteration: int = Field(
        default=0,
        description="当前迭代次数",
    )
    max_iterations: int = Field(
        default=10,
        description="最大迭代次数（防止无限循环）",
    )

    # 对话历史
    messages: list[Message] = Field(
        default_factory=list,
        description="完整的对话历史",
    )

    # 工具调用记录
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description="工具调用历史记录",
    )

    # 输出
    final_output: Optional[str] = Field(
        default=None,
        description="最终输出结果",
    )

    # 元数据
    session_id: Optional[str] = Field(
        default=None,
        description="所属会话 ID",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # ── 便捷方法 ─────────────────────────────────────────────────

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self.messages.append(message)
        self.updated_at = datetime.now()

    def add_tool_call(self, record: ToolCallRecord):
        """记录工具调用"""
        self.tool_calls.append(record)
        self.updated_at = datetime.now()

    def increment_iteration(self):
        """增加迭代计数"""
        self.iteration += 1
        self.updated_at = datetime.now()

    @property
    def is_finished(self) -> bool:
        """是否已完成"""
        return self.state == AgentState.FINISHED

    @property
    def should_stop(self) -> bool:
        """是否应该停止（完成或超出最大迭代）"""
        return self.is_finished or self.iteration >= self.max_iterations

    def get_summary(self) -> dict:
        """获取状态摘要（用于日志和监控）"""
        return {
            "state": self.state.value,
            "iteration": f"{self.iteration}/{self.max_iterations}",
            "messages": len(self.messages),
            "tool_calls": len(self.tool_calls),
            "has_output": self.final_output is not None,
        }


# ============================================================================
# 第五部分：响应格式定义（response.py）
# ============================================================================

class ToolCallInfo(BaseModel):
    """工具调用信息（包含在响应中）"""
    tool_name: str
    arguments: dict
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])


class AgentResponse(BaseModel):
    """
    Agent 响应格式

    Agent 每次处理用户输入后，返回这个结构化的响应。
    包含文本回复、工具调用信息、状态等。
    """

    # 文本回复
    content: str = Field(description="AI 的文本回复")

    # 工具调用（如果有）
    tool_calls: list[ToolCallInfo] = Field(
        default_factory=list,
        description="需要执行的工具调用",
    )

    # 状态信息
    is_final: bool = Field(
        default=True,
        description="是否是最终回复（False 表示还需要继续处理）",
    )
    state: AgentState = Field(default=AgentState.FINISHED)

    # 元数据
    usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token 使用量",
    )
    latency_ms: int = Field(
        default=0,
        description="处理耗时（毫秒）",
    )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "content": self.content,
            "tool_calls": [tc.model_dump() for tc in self.tool_calls],
            "is_final": self.is_final,
            "state": self.state.value,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
        }


# ============================================================================
# 第六部分：Schema 使用演示
# ============================================================================

def demonstrate_message():
    """演示消息模型的使用"""
    print("=" * 60)
    print("消息模型 (Message) 演示")
    print("=" * 60)

    # 创建不同类型的消息
    sys_msg = Message.system("你是一个有帮助的AI助手。")
    user_msg = Message.user("北京今天天气怎么样？")
    ai_msg = Message.assistant("让我帮你查一下北京的天气。")
    tool_msg = Message.tool_result("北京：晴天，28°C", tool_call_id="call_001")

    print(f"\n  系统消息: role={sys_msg.role.value}, content={sys_msg.content}")
    print(f"  用户消息: role={user_msg.role.value}, content={user_msg.content}")
    print(f"  AI 消息:   role={ai_msg.role.value}, content={ai_msg.content}")
    print(f"  工具消息: role={tool_msg.role.value}, "
          f"tool_call_id={tool_msg.tool_call_id}")

    # 转换为 LLM 格式
    print(f"\n  LLM 格式: {user_msg.to_llm_format()}")

    # 验证：空内容会报错
    print("\n── 数据验证演示 ──")
    try:
        # 注意：这里 content 有默认值 ""，所以不会报错
        # 如果要强制非空，需要 Field(min_length=1)
        msg = Message(role=Role.USER)
        print(f"  默认 content: '{msg.content}' (空字符串)")
    except Exception as e:
        print(f"  验证错误: {e}")
    print()


def demonstrate_tool_schema():
    """演示工具 Schema 的使用"""
    print("=" * 60)
    print("工具 Schema (ToolSchema) 演示")
    print("=" * 60)

    tools = [SEARCH_TOOL, CALCULATOR_TOOL, WEATHER_TOOL]

    for tool in tools:
        print(f"\n   {tool.name}")
        print(f"     描述: {tool.description}")
        print(f"     参数:")
        for param in tool.parameters:
            req = " [必填]" if param.required else " [可选]"
            print(f"       - {param.name} ({param.type}): "
                  f"{param.description}{req}")

    # 转换为不同 API 格式
    print(f"\n── OpenAI 格式 ──")
    openai_format = WEATHER_TOOL.to_openai_format()
    print(f"  {json.dumps(openai_format, indent=2, ensure_ascii=False)[:200]}...")

    print(f"\n── Anthropic 格式 ──")
    anthropic_format = WEATHER_TOOL.to_anthropic_format()
    print(f"  {json.dumps(anthropic_format, indent=2, ensure_ascii=False)[:200]}...")
    print()


def demonstrate_agent_state():
    """演示 Agent 状态模型的使用"""
    print("=" * 60)
    print("Agent 状态 (AgentStateModel) 演示")
    print("=" * 60)

    # 创建 Agent 状态
    state = AgentStateModel(
        max_iterations=5,
        session_id="session_001",
    )

    print(f"  初始状态: {state.get_summary()}")

    # 模拟 Agent 运行过程
    state.state = AgentState.THINKING
    state.add_message(Message.user("帮我查天气"))
    print(f"  思考中:   {state.get_summary()}")

    state.state = AgentState.ACTING
    state.add_tool_call(ToolCallRecord(
        tool_name="get_weather",
        arguments={"city": "北京"},
    ))
    state.increment_iteration()
    print(f"  行动中:   {state.get_summary()}")

    state.state = AgentState.OBSERVING
    state.add_message(Message.tool_result("北京：晴天，28°C", "call_001"))
    print(f"  观察中:   {state.get_summary()}")

    state.state = AgentState.RESPONDING
    state.add_message(Message.assistant("北京今天晴天，28°C。"))
    state.final_output = "北京今天晴天，28°C。"
    state.state = AgentState.FINISHED
    print(f"  已完成:   {state.get_summary()}")

    print(f"\n  是否完成: {state.is_finished}")
    print(f"  是否应停止: {state.should_stop}")
    print()


def demonstrate_response():
    """演示响应格式的使用"""
    print("=" * 60)
    print("Agent 响应 (AgentResponse) 演示")
    print("=" * 60)

    # 最终回复
    response = AgentResponse(
        content="北京今天晴天，28°C，适合出行。",
        is_final=True,
        state=AgentState.FINISHED,
        usage={"input_tokens": 150, "output_tokens": 80},
        latency_ms=1200,
    )
    print(f"  回复内容: {response.content}")
    print(f"  是否最终: {response.is_final}")
    print(f"  Token用量: {response.usage}")
    print(f"  耗时: {response.latency_ms}ms")

    # 需要调用工具的回复
    tool_response = AgentResponse(
        content="正在为你查询天气...",
        tool_calls=[
            ToolCallInfo(
                tool_name="get_weather",
                arguments={"city": "北京"},
            )
        ],
        is_final=False,
        state=AgentState.ACTING,
    )
    print(f"\n  工具调用回复:")
    print(f"  内容: {tool_response.content}")
    print(f"  工具: {tool_response.tool_calls[0].tool_name}")
    print(f"  参数: {tool_response.tool_calls[0].arguments}")
    print(f"  是否最终: {tool_response.is_final}")
    print()


# ============================================================================
# 第七部分：进阶 - Schema 验证与自定义
# ============================================================================

def advanced_schema_patterns():
    """
    进阶：Pydantic 高级用法

    包括自定义验证器、模型验证、嵌套模型等。
    """

    print("=" * 60)
    print("进阶：Pydantic 高级用法")
    print("=" * 60)

    # ── 示例1：自定义字段验证器 ─────────────────────────────────
    print("\n── 自定义字段验证器 ──")

    class ValidatedToolSchema(ToolSchema):
        """带验证的工具 Schema"""

        @field_validator("name")
        @classmethod
        def validate_name(cls, v: str) -> str:
            """验证工具名称格式"""
            if not v.replace("_", "").isalnum():
                raise ValueError(
                    f"工具名称只能包含字母、数字和下划线: {v}"
                )
            if len(v) > 50:
                raise ValueError(f"工具名称不能超过50个字符: {v}")
            return v

        @field_validator("description")
        @classmethod
        def validate_description(cls, v: str) -> str:
            """验证描述不能为空"""
            if not v.strip():
                raise ValueError("工具描述不能为空")
            return v.strip()

    # 正常创建
    tool = ValidatedToolSchema(
        name="my_tool",
        description="  一个测试工具  ",  # 会自动 strip
        parameters=[],
    )
    print(f"  ✅ 名称: '{tool.name}', 描述: '{tool.description}'")

    # 验证失败
    try:
        ValidatedToolSchema(
            name="invalid name!",  # 包含空格和特殊字符
            description="测试",
            parameters=[],
        )
    except Exception as e:
        print(f"  ❌ 验证失败: {e}")

    # ── 示例2：模型级别验证器 ───────────────────────────────────
    print("\n── 模型级别验证器 ──")

    class ConstrainedAgentState(AgentStateModel):
        """带约束的 Agent 状态"""

        @model_validator(mode="after")
        def validate_iterations(self):
            """验证迭代次数不超过最大值"""
            if self.iteration > self.max_iterations:
                raise ValueError(
                    f"迭代次数 ({self.iteration}) 超过最大值 "
                    f"({self.max_iterations})"
                )
            return self

    state = ConstrainedAgentState(max_iterations=5, iteration=3)
    print(f"  ✅ 迭代 {state.iteration}/{state.max_iterations}")

    try:
        ConstrainedAgentState(max_iterations=5, iteration=10)
    except Exception as e:
        print(f"  ❌ 验证失败: {e}")

    print()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "" * 30)
    print("第四课：Schema 模块 - 数据结构的艺术")
    print("📐" * 30 + "\n")

    # 1. 为什么需要 Schema
    why_schema_matters()

    # 2. 消息模型演示
    demonstrate_message()

    # 3. 工具 Schema 演示
    demonstrate_tool_schema()

    # 4. Agent 状态演示
    demonstrate_agent_state()

    # 5. 响应格式演示
    demonstrate_response()

    # 6. 进阶用法
    advanced_schema_patterns()

    print("=" * 60)
    print("✅ 第四课学习完成！")
    print("=" * 60)
    print("""
 本课要点总结：
1. Pydantic 提供类型安全、自动验证、序列化的数据模型
2. Message 是 Agent 系统的核心数据单元
3. ToolSchema 定义了工具的能力，LLM 依赖它做工具选择
4. AgentStateModel 跟踪 Agent 的完整运行状态
5. AgentResponse 是 Agent 处理后的结构化输出
6. 进阶：自定义验证器、模型验证器确保数据质量

 下一课：Memory 模块 - 我们将实现短期记忆、长期记忆和向量记忆
    """)
