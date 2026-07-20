"""
第30课：Function Calling 深度实践

核心知识点：
1. Function Calling 原理 - LLM 如何决定调用哪个函数
2. JSON Schema 定义规范 - 工具参数的精确描述
3. 并行工具调用 - 一次调用多个工具
4. 嵌套函数与复杂参数 - 数组/对象/枚举参数
5. 工具调用错误处理 - 优雅恢复策略
6. Prompt Caching - 提示缓存优化性能与成本
7. 不同模型的 Function Calling 差异 - Claude/GPT/开源模型

架构要点：
- Function Calling 是 Agent 系统的核心能力
- LLM 通过 JSON Schema 理解工具参数格式
- 工具描述质量直接影响 LLM 的选择准确率
- 并行调用可以显著降低延迟
- 错误恢复保证系统鲁棒性

实际应用场景：
- Agent 需要同时查询多个数据源
- 复杂参数校验和类型转换
- 工具调用失败后的重试与降级
- 高频调用场景下的成本优化
"""

import asyncio
import json
import time
import hashlib
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


# ============================================================
# 第一部分：Function Calling 原理
# ============================================================

print("\n" + "=" * 60)
print("第30课：Function Calling 深度实践")
print("=" * 60)

print("""
-- Function Calling 工作原理 --

  用户输入: "北京今天天气怎么样？帮我算一下100*15"
       |
       v
  [LLM 理解意图]
       |
       v
  [决策: 需要调用哪些工具?]
       |
       +---> 工具1: get_weather(city="北京", date="今天")
       |
       +---> 工具2: calculator(expression="100*15")
       |
       v
  [并行执行工具调用]
       |
       v
  [将工具结果返回给 LLM]
       |
       v
  [LLM 整合结果生成回复]
       |
       v
  "北京今天晴，25度。100乘以15等于1500。"

-- Function Calling 三阶段 --

  阶段1: 工具定义 (Tool Definition)
    用 JSON Schema 描述工具名称、功能、参数

  阶段2: 工具选择 (Tool Selection)
    LLM 根据用户输入 + 工具描述，决定调用哪些工具

  阶段3: 工具执行 (Tool Execution)
    系统执行工具，将结果返回给 LLM

-- 核心概念 --

  JSON Schema: 描述工具参数的标准格式
    - type: 参数类型 (string/number/boolean/object/array)
    - description: 参数说明（LLM 靠这个理解参数含义）
    - required: 必填参数列表
    - enum: 参数可选值
    - default: 默认值
""")


# ============================================================
# 第二部分：JSON Schema 定义规范
# ============================================================

@dataclass
class ParameterDef:
    """参数定义"""
    name: str
    param_type: str  # string, number, boolean, object, array
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Any = None
    items_type: Optional[str] = None  # array 元素的类型
    properties: Optional[Dict] = None  # object 的属性


class SchemaBuilder:
    """
    JSON Schema 构建器
    
    【功能】
    - 简化 JSON Schema 定义过程
    - 自动处理 required 字段
    - 支持嵌套对象和数组
    - 生成文档字符串
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.parameters: List[ParameterDef] = []
    
    def add_param(self, name: str, param_type: str, description: str,
                  required: bool = True, enum: List[str] = None,
                  default: Any = None, items_type: str = None,
                  properties: Dict = None) -> "SchemaBuilder":
        """添加参数定义"""
        self.parameters.append(ParameterDef(
            name=name, param_type=param_type,
            description=description, required=required,
            enum=enum, default=default,
            items_type=items_type, properties=properties
        ))
        return self
    
    def build(self) -> Dict:
        """构建完整的 JSON Schema"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.param_type,
                "description": param.description
            }
            
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            if param.items_type:
                prop["items"] = {"type": param.items_type}
            if param.properties:
                prop["properties"] = param.properties
            
            properties[param.name] = prop
            
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
                    "required": required
                }
            }
        }
    
    def display(self):
        """展示生成的 Schema"""
        schema = self.build()
        print(f"\n  -- Schema: {self.name} --")
        print(f"  描述: {self.description}")
        print(f"  参数:")
        for param in self.parameters:
            req_mark = "[必填]" if param.required else "[可选]"
            enum_str = f" 可选值={param.enum}" if param.enum else ""
            print(f"    {req_mark} {param.name} ({param.param_type}): "
                  f"{param.description}{enum_str}")
        print(f"\n  JSON Schema:")
        print(f"  {json.dumps(schema, indent=2, ensure_ascii=False)[:500]}...")


def demo_schema_building():
    """演示 Schema 构建"""
    print("\n" + "=" * 60)
    print("演示1: JSON Schema 构建")
    print("=" * 60)
    
    # 示例1: 简单参数
    print("\n  -- 示例1: 天气查询工具 --")
    weather_schema = (
        SchemaBuilder("get_weather", "查询指定城市的天气信息")
        .add_param("city", "string", "城市名称，如：北京、上海")
        .add_param("date", "string", "日期，格式: YYYY-MM-DD 或 '今天'/'明天'",
                   required=False, default="今天")
        .add_param("unit", "string", "温度单位",
                   required=False, enum=["celsius", "fahrenheit"], default="celsius")
    )
    weather_schema.display()
    
    # 示例2: 数组参数
    print("\n  -- 示例2: 批量搜索工具 --")
    search_schema = (
        SchemaBuilder("batch_search", "同时搜索多个关键词")
        .add_param("queries", "array", "搜索关键词列表",
                   items_type="string")
        .add_param("max_results", "number", "每个关键词最大返回数",
                   required=False, default=5)
        .add_param("search_type", "string", "搜索类型",
                   enum=["web", "news", "academic", "images"])
    )
    search_schema.display()
    
    # 示例3: 嵌套对象参数
    print("\n  -- 示例3: 发送邮件工具 --")
    email_schema = (
        SchemaBuilder("send_email", "发送电子邮件")
        .add_param("to", "string", "收件人邮箱地址")
        .add_param("subject", "string", "邮件主题")
        .add_param("body", "string", "邮件正文内容")
        .add_param("priority", "string", "优先级",
                   enum=["low", "normal", "high", "urgent"])
        .add_param("attachments", "array", "附件文件路径列表",
                   required=False, items_type="string")
        .add_param("cc", "array", "抄送人列表",
                   required=False, items_type="string")
    )
    email_schema.display()


# ============================================================
# 第三部分：并行工具调用
# ============================================================

@dataclass
class ToolCallRequest:
    """工具调用请求"""
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed


@dataclass
class ToolCallResult:
    """工具调用结果"""
    call_id: str
    tool_name: str
    result: Any
    latency_ms: float
    status: str  # success, error


class ParallelExecutor:
    """
    并行工具执行器
    
    【原理】
    当 LLM 决定同时调用多个工具时，
    并行执行器可以同时执行这些工具，显著降低总延迟。
    
    【策略】
    - asyncio.gather: 并发执行所有工具
    - 超时控制: 单个工具有最大执行时间
    - 错误隔离: 一个工具失败不影响其他工具
    """
    
    def __init__(self, tools: Dict[str, Callable]):
        self.tools = tools
        self.execution_log: List[ToolCallResult] = []
    
    async def execute_single(self, request: ToolCallRequest) -> ToolCallResult:
        """执行单个工具调用"""
        start = time.time()
        tool_fn = self.tools.get(request.tool_name)
        
        if not tool_fn:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                result=f"Error: Tool '{request.tool_name}' not found",
                latency_ms=(time.time() - start) * 1000,
                status="error"
            )
        
        try:
            request.status = "running"
            result = tool_fn(**request.arguments)
            if asyncio.iscoroutine(result):
                result = await result
            
            latency = (time.time() - start) * 1000
            request.status = "completed"
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                result=result,
                latency_ms=latency,
                status="success"
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            request.status = "failed"
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                result=f"Error: {str(e)}",
                latency_ms=latency,
                status="error"
            )
    
    async def execute_parallel(self, requests: List[ToolCallRequest]) -> List[ToolCallResult]:
        """并行执行多个工具调用"""
        print(f"\n  [Parallel] 并行执行 {len(requests)} 个工具调用...")
        tasks = [self.execute_single(req) for req in requests]
        results = await asyncio.gather(*tasks)
        
        success = sum(1 for r in results if r.status == "success")
        failed = sum(1 for r in results if r.status == "error")
        max_latency = max(r.latency_ms for r in results)
        
        print(f"  [Parallel] 完成: {success} 成功, {failed} 失败, "
              f"最大延迟: {max_latency:.0f}ms")
        
        self.execution_log.extend(results)
        return list(results)
    
    async def execute_sequential(self, requests: List[ToolCallRequest]) -> List[ToolCallResult]:
        """顺序执行（对比用）"""
        print(f"\n  [Sequential] 顺序执行 {len(requests)} 个工具调用...")
        results = []
        for req in requests:
            result = await self.execute_single(req)
            results.append(result)
        
        total_latency = sum(r.latency_ms for r in results)
        print(f"  [Sequential] 完成: 总延迟 {total_latency:.0f}ms")
        
        self.execution_log.extend(results)
        return results


# ============================================================
# 第四部分：工具调用错误处理
# ============================================================

class ErrorRecoveryStrategy(Enum):
    """错误恢复策略"""
    RETRY = "retry"               # 重试
    FALLBACK = "fallback"         # 降级
    SKIP = "skip"                 # 跳过
    ABORT = "abort"               # 中止
    ASK_HUMAN = "ask_human"       # 请求人工干预


@dataclass
class ErrorRecoveryConfig:
    """错误恢复配置"""
    strategy: ErrorRecoveryStrategy
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_tool: Optional[str] = None
    error_message: str = ""


class ErrorHandler:
    """
    工具调用错误处理器
    
    【错误类型】
    - ToolNotFound: 工具不存在
    - InvalidArguments: 参数格式错误
    - TimeoutError: 工具执行超时
    - PermissionDenied: 权限不足
    - ExternalServiceError: 外部服务错误
    
    【恢复策略】
    - Retry: 自动重试（指数退避）
    - Fallback: 使用备选工具
    - Skip: 跳过该步骤继续执行
    - Abort: 中止并返回错误信息
    - AskHuman: 请求人工介入
    """
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.recovery_log: List[Dict] = []
    
    def classify_error(self, error: Exception) -> str:
        """分类错误类型"""
        error_str = str(error).lower()
        if "timeout" in error_str:
            return "timeout"
        elif "permission" in error_str or "auth" in error_str:
            return "permission"
        elif "not found" in error_str:
            return "not_found"
        elif "invalid" in error_str or "argument" in error_str:
            return "invalid_args"
        else:
            return "unknown"
    
    def get_recovery_strategy(self, error_type: str, 
                              config: ErrorRecoveryConfig) -> Dict:
        """获取恢复策略"""
        count = self.error_counts.get(error_type, 0)
        self.error_counts[error_type] = count + 1
        
        recovery = {
            "error_type": error_type,
            "occurrence": count + 1,
            "action": config.strategy.value,
            "message": config.error_message
        }
        
        if config.strategy == ErrorRecoveryStrategy.RETRY:
            if count < config.max_retries:
                recovery["action"] = "retry"
                recovery["retry_number"] = count + 1
                recovery["delay"] = config.retry_delay * (2 ** count)
            else:
                recovery["action"] = "abort"
                recovery["message"] = f"已达最大重试次数 ({config.max_retries})"
        
        elif config.strategy == ErrorRecoveryStrategy.FALLBACK:
            recovery["fallback_tool"] = config.fallback_tool
        
        self.recovery_log.append(recovery)
        return recovery
    
    def display_log(self):
        """展示错误恢复日志"""
        print("\n  -- 错误恢复日志 --")
        for entry in self.recovery_log:
            print(f"  [{entry['error_type']}] 第{entry['occurrence']}次 -> "
                  f"动作: {entry['action']}")
            if "message" in entry and entry["message"]:
                print(f"    信息: {entry['message']}")


# ============================================================
# 第五部分：Prompt Caching 提示缓存
# ============================================================

class PromptCache:
    """
    提示缓存系统
    
    【原理】
    在多轮对话中，System Prompt 和早期对话历史通常不变。
    通过缓存这些不变的部分，可以：
    1. 减少 API 调用成本（缓存的 token 更便宜）
    2. 降低延迟（不需要重新处理缓存部分）
    3. 提高吞吐量
    
    【缓存策略】
    - 精确匹配缓存: 完全相同的 prompt 直接返回缓存
    - 前缀缓存: 共享前缀的请求复用缓存
    - 语义缓存: 语义相似的请求复用缓存
    
    【Claude Prompt Caching】
    - 支持缓存 system prompt + 对话历史
    - 缓存最小 1024 tokens
    - 缓存有效期 5 分钟
    - 缓存读取费用为写入的 10%
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Dict] = {}
        self.stats = {"hits": 0, "misses": 0, "writes": 0}
    
    def _compute_key(self, prompt: str) -> str:
        """计算缓存键"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[str]:
        """查询缓存"""
        key = self._compute_key(prompt)
        entry = self.cache.get(key)
        
        if entry:
            # 检查是否过期（5分钟）
            age = (datetime.now() - entry["timestamp"]).total_seconds()
            if age < 300:  # 5分钟有效期
                self.stats["hits"] += 1
                return entry["response"]
            else:
                # 过期删除
                del self.cache[key]
        
        self.stats["misses"] += 1
        return None
    
    def put(self, prompt: str, response: str):
        """写入缓存"""
        key = self._compute_key(prompt)
        
        # 容量控制
        if len(self.cache) >= self.max_size:
            # 删除最旧的条目
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            "prompt": prompt[:100],
            "response": response,
            "timestamp": datetime.now()
        }
        self.stats["writes"] += 1
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "writes": self.stats["writes"],
            "hit_rate": f"{hit_rate:.1%}",
            "cache_size": len(self.cache)
        }


class CostOptimizer:
    """
    成本优化器
    
    【优化策略】
    1. Prompt Caching: 缓存不变的 system prompt
    2. Token 预算: 限制每次调用的最大 token
    3. 模型路由: 简单任务用小模型，复杂任务用大模型
    4. 请求合并: 批量处理多个相似请求
    """
    
    def __init__(self):
        self.total_cost = 0.0
        self.total_tokens = 0
        self.call_count = 0
        self.cache = PromptCache()
    
    def estimate_cost(self, input_tokens: int, output_tokens: int,
                      model: str) -> float:
        """估算调用成本"""
        pricing = {
            "claude-sonnet": {"input": 3.0, "output": 15.0},
            "claude-opus": {"input": 15.0, "output": 75.0},
            "gpt-4": {"input": 10.0, "output": 30.0},
            "gpt-3.5": {"input": 0.5, "output": 1.5},
        }
        
        model_key = model.lower()
        for key in pricing:
            if key in model_key:
                p = pricing[key]
                # 检查缓存命中（缓存 token 便宜 90%）
                cached_input = int(input_tokens * 0.1)  # 假设 90% 命中缓存
                cost = ((cached_input + input_tokens * 0.1) * p["input"] +
                        output_tokens * p["output"]) / 1_000_000
                return round(cost, 6)
        
        return 0.01  # 默认成本
    
    def optimize_prompt(self, system_prompt: str, messages: List[Dict],
                       task_complexity: str) -> Dict:
        """
        优化 prompt 以降低成本
        
        【策略】
        - 简单任务: 使用 GPT-3.5 级别模型
        - 中等任务: 使用 Claude Sonnet 级别
        - 复杂任务: 使用 Claude Opus 级别
        """
        model_selection = {
            "simple": "gpt-3.5-turbo",
            "medium": "claude-sonnet-4-20250514",
            "complex": "claude-opus-4-20250514"
        }
        
        selected_model = model_selection.get(task_complexity, "claude-sonnet-4-20250514")
        
        # 计算 token 数
        input_tokens = len(system_prompt) // 4 + sum(
            len(m.get("content", "")) // 4 for m in messages
        )
        
        # 检查缓存
        cache_key = system_prompt + str(messages[:2])  # 前2条消息作为缓存键
        cached = self.cache.get(cache_key)
        
        result = {
            "model": selected_model,
            "estimated_input_tokens": input_tokens,
            "cache_hit": cached is not None,
            "estimated_cost": self.estimate_cost(
                input_tokens, 500, selected_model
            )
        }
        
        self.total_cost += result["estimated_cost"]
        self.total_tokens += input_tokens
        self.call_count += 1
        
        return result


# ============================================================
# 第六部分：完整 Function Calling 流程演示
# ============================================================

class MockLLM:
    """模拟 LLM 的 Function Calling 行为"""
    
    def __init__(self, model_name: str = "claude-sonnet"):
        self.model_name = model_name
        self.call_count = 0
    
    def decide_tool_calls(self, user_input: str, 
                          available_tools: List[Dict]) -> List[Dict]:
        """
        模拟 LLM 决定调用哪些工具
        
        实际应用中，这一步由 LLM API 完成
        """
        self.call_count += 1
        tool_calls = []
        
        # 简单的关键词匹配模拟 LLM 决策
        if "天气" in user_input:
            tool_calls.append({
                "id": f"call_{self.call_count}_1",
                "name": "get_weather",
                "arguments": {"city": "北京", "unit": "celsius"}
            })
        
        if "计算" in user_input or any(c in user_input for c in "+-*/"):
            import re
            expr_match = re.search(r'[\d.]+\s*[*+\-/]\s*[\d.]+', user_input)
            expr = expr_match.group() if expr_match else "0"
            tool_calls.append({
                "id": f"call_{self.call_count}_2",
                "name": "calculator",
                "arguments": {"expression": expr}
            })
        
        if "搜索" in user_input or "查" in user_input:
            tool_calls.append({
                "id": f"call_{self.call_count}_3",
                "name": "web_search",
                "arguments": {"query": user_input, "max_results": 3}
            })
        
        return tool_calls


async def demo_parallel_execution():
    """演示并行 vs 顺序执行"""
    print("\n" + "=" * 60)
    print("演示2: 并行工具调用")
    print("=" * 60)
    
    # 定义模拟工具
    def get_weather(city: str, unit: str = "celsius") -> str:
        time.sleep(0.1)  # 模拟网络延迟
        return f"{city}: 25度{'C' if unit == 'celsius' else 'F'}"
    
    def calculator(expression: str) -> str:
        time.sleep(0.05)
        try:
            result = eval(expression)
            return f"{expression} = {result}"
        except:
            return "计算错误"
    
    def web_search(query: str, max_results: int = 5) -> str:
        time.sleep(0.15)
        return f"搜索 '{query}' 找到 {max_results} 条结果"
    
    tools = {
        "get_weather": get_weather,
        "calculator": calculator,
        "web_search": web_search
    }
    
    executor = ParallelExecutor(tools)
    
    # 模拟 LLM 决定调用多个工具
    requests = [
        ToolCallRequest("c1", "get_weather", {"city": "北京", "unit": "celsius"}),
        ToolCallRequest("c2", "calculator", {"expression": "100*15"}),
        ToolCallRequest("c3", "web_search", {"query": "AI最新进展", "max_results": 3}),
    ]
    
    print("\n  -- 并行执行 --")
    start = time.time()
    parallel_results = await executor.execute_parallel(requests)
    parallel_time = (time.time() - start) * 1000
    
    for r in parallel_results:
        print(f"    [{r.status}] {r.tool_name}: {r.result} ({r.latency_ms:.0f}ms)")
    
    # 重置执行器
    executor2 = ParallelExecutor(tools)
    requests2 = [
        ToolCallRequest("s1", "get_weather", {"city": "上海", "unit": "celsius"}),
        ToolCallRequest("s2", "calculator", {"expression": "200+300"}),
        ToolCallRequest("s3", "web_search", {"query": "Python教程", "max_results": 5}),
    ]
    
    print("\n  -- 顺序执行（对比）--")
    start = time.time()
    sequential_results = await executor2.execute_sequential(requests2)
    sequential_time = (time.time() - start) * 1000
    
    for r in sequential_results:
        print(f"    [{r.status}] {r.tool_name}: {r.result} ({r.latency_ms:.0f}ms)")
    
    print(f"\n  -- 性能对比 --")
    print(f"  并行总耗时: {parallel_time:.0f}ms")
    print(f"  顺序总耗时: {sequential_time:.0f}ms")
    print(f"  加速比: {sequential_time/parallel_time:.1f}x")


async def demo_error_handling():
    """演示错误处理策略"""
    print("\n" + "=" * 60)
    print("演示3: 工具调用错误处理")
    print("=" * 60)
    
    handler = ErrorHandler()
    
    # 模拟不同错误场景
    scenarios = [
        {
            "error": TimeoutError("工具执行超时 (>30s)"),
            "config": ErrorRecoveryConfig(
                strategy=ErrorRecoveryStrategy.RETRY,
                max_retries=3, retry_delay=1.0,
                error_message="超时重试"
            )
        },
        {
            "error": PermissionError("无权访问数据库"),
            "config": ErrorRecoveryConfig(
                strategy=ErrorRecoveryStrategy.FALLBACK,
                fallback_tool="read_only_query",
                error_message="降级为只读查询"
            )
        },
        {
            "error": ValueError("参数格式错误: date 应为 YYYY-MM-DD"),
            "config": ErrorRecoveryConfig(
                strategy=ErrorRecoveryStrategy.ASK_HUMAN,
                error_message="请人工修正参数格式"
            )
        },
        {
            "error": ConnectionError("外部API不可达"),
            "config": ErrorRecoveryConfig(
                strategy=ErrorRecoveryStrategy.SKIP,
                error_message="跳过该步骤，使用缓存数据"
            )
        },
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        error = scenario["error"]
        config = scenario["config"]
        
        error_type = handler.classify_error(error)
        recovery = handler.get_recovery_strategy(error_type, config)
        
        print(f"\n  -- 场景{i}: {type(error).__name__} --")
        print(f"  错误: {error}")
        print(f"  分类: {error_type}")
        print(f"  策略: {recovery['action']}")
        if "fallback_tool" in recovery:
            print(f"  降级工具: {recovery['fallback_tool']}")
        if "retry_number" in recovery:
            print(f"  重试次数: {recovery['retry_number']}, "
                  f"延迟: {recovery.get('delay', 0):.1f}s")
    
    handler.display_log()


def demo_prompt_caching():
    """演示 Prompt Caching"""
    print("\n" + "=" * 60)
    print("演示4: Prompt Caching 提示缓存")
    print("=" * 60)
    
    cache = PromptCache()
    
    # 模拟多轮对话（system prompt 不变）
    system_prompt = "你是一个专业的客服助手，负责回答产品相关问题..."
    
    conversations = [
        [{"role": "user", "content": "产品A的价格是多少？"}],
        [{"role": "user", "content": "产品A的价格是多少？"}],  # 重复请求
        [{"role": "user", "content": "产品B有什么功能？"}],
        [{"role": "user", "content": "产品A的价格是多少？"}],  # 再次重复
        [{"role": "user", "content": "如何联系售后？"}],
    ]
    
    print("\n  -- 模拟多轮对话（带缓存）--")
    for i, messages in enumerate(conversations, 1):
        cache_key = system_prompt + str(messages)
        
        # 先查缓存
        cached = cache.get(cache_key)
        if cached:
            print(f"  轮次{i}: CACHE HIT -> {cached[:40]}...")
        else:
            # 模拟 LLM 调用
            response = f"[LLM回复] 关于 {messages[0]['content'][:15]}... 的回答"
            cache.put(cache_key, response)
            print(f"  轮次{i}: CACHE MISS -> 调用LLM -> {response[:40]}...")
    
    stats = cache.get_stats()
    print(f"\n  -- 缓存统计 --")
    print(f"  命中: {stats['hits']}, 未命中: {stats['misses']}")
    print(f"  命中率: {stats['hit_rate']}")
    print(f"  写入次数: {stats['writes']}")


def demo_cost_optimization():
    """演示成本优化"""
    print("\n" + "=" * 60)
    print("演示5: 成本优化 - 模型路由")
    print("=" * 60)
    
    optimizer = CostOptimizer()
    
    system_prompt = "你是一个智能助手。" * 100  # 模拟长 system prompt
    messages = [{"role": "user", "content": "你好"}]
    
    tasks = [
        {"input": "1+1等于几？", "complexity": "simple"},
        {"input": "帮我分析这段代码的性能瓶颈", "complexity": "medium"},
        {"input": "设计一个分布式系统架构方案", "complexity": "complex"},
        {"input": "今天星期几？", "complexity": "simple"},
        {"input": "帮我写一首诗", "complexity": "medium"},
    ]
    
    print("\n  -- 模型路由决策 --")
    for task in tasks:
        msgs = [{"role": "user", "content": task["input"]}]
        result = optimizer.optimize_prompt(system_prompt, msgs, task["complexity"])
        
        print(f"\n  任务: {task['input']}")
        print(f"  复杂度: {task['complexity']}")
        print(f"  选择模型: {result['model']}")
        print(f"  缓存命中: {result['cache_hit']}")
        print(f"  预估成本: {result['estimated_cost']:.6f} 元")
    
    print(f"\n  -- 成本统计 --")
    print(f"  总调用次数: {optimizer.call_count}")
    print(f"  总Token数: {optimizer.total_tokens}")
    print(f"  总预估成本: {optimizer.total_cost:.6f} 元")


async def demo_full_pipeline():
    """演示完整 Function Calling 流程"""
    print("\n" + "=" * 60)
    print("演示6: 完整 Function Calling 流程")
    print("=" * 60)
    
    llm = MockLLM("claude-sonnet")
    
    # 可用工具列表
    tools = {
        "get_weather": lambda city, unit="celsius": f"{city}: 晴, 25度",
        "calculator": lambda expression: f"{expression} = {eval(expression)}",
        "web_search": lambda query, max_results=5: f"搜索'{query}' 找到{max_results}条结果"
    }
    
    executor = ParallelExecutor(tools)
    
    # 测试用例
    test_inputs = [
        "北京今天天气怎么样？顺便帮我算一下 256*128",
        "搜索最新的AI论文",
        "计算 99+1",
    ]
    
    for user_input in test_inputs:
        print(f"\n  -- 用户输入: {user_input} --")
        
        # 阶段1: LLM 决定调用哪些工具
        tool_calls = llm.decide_tool_calls(user_input, list(tools.keys()))
        print(f"  LLM决策: 调用 {len(tool_calls)} 个工具")
        
        if not tool_calls:
            print(f"  -> 直接回复（无需工具调用）")
            continue
        
        # 阶段2: 构建请求
        requests = [
            ToolCallRequest(
                call_id=tc["id"],
                tool_name=tc["name"],
                arguments=tc["arguments"]
            )
            for tc in tool_calls
        ]
        
        # 阶段3: 并行执行
        results = await executor.execute_parallel(requests)
        
        # 阶段4: 整合结果
        print(f"  工具结果:")
        for r in results:
            print(f"    [{r.tool_name}] {r.result}")
        
        print(f"  -> LLM 整合结果生成最终回复")


# ============================================================
# 总结
# ============================================================

async def main():
    """运行所有演示"""
    demo_schema_building()
    await demo_parallel_execution()
    await demo_error_handling()
    demo_prompt_caching()
    demo_cost_optimization()
    await demo_full_pipeline()
    
    print("\n" + "=" * 60)
    print("课程总结")
    print("=" * 60)
    
    print("""
  本课深入讲解了 Function Calling 的核心机制：

  核心知识点：
  1. Function Calling 原理: LLM 如何决定调用工具
  2. JSON Schema 定义: 精确描述工具参数的标准格式
  3. 并行工具调用: asyncio.gather 同时执行多个工具
  4. 错误处理策略: 重试/降级/跳过/人工介入
  5. Prompt Caching: 缓存不变部分降低成本和延迟
  6. 成本优化: 模型路由 + 缓存 + Token预算

  最佳实践：
  - 工具描述要清晰：LLM 靠描述选择工具
  - 参数说明要具体：包含示例和约束条件
  - 必填参数用 required 标注
  - 枚举值用 enum 限制选项
  - 并行调用无依赖的工具，降低延迟
  - 设置合理的超时和重试策略
  - 利用 Prompt Caching 降低重复调用成本
  - 简单任务路由到小模型，复杂任务用大模型

  不同模型的 Function Calling 差异：
  - Claude: 支持 tool_use，并行调用，缓存友好
  - GPT-4: 支持 function calling，parallel tool calls
  - 开源模型: 需要微调才能可靠使用 function calling

  生产部署建议：
  - 始终为工具调用设置超时
  - 实现断路器防止级联故障
  - 缓存高频查询结果
  - 监控工具调用成功率和延迟
  - 设置成本告警防止意外支出
""")


if __name__ == "__main__":
    asyncio.run(main())
