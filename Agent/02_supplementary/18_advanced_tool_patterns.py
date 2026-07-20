"""
==============================================================================
第十八课（补充）：Advanced Tool Patterns（高级工具模式）
==============================================================================

【为什么需要单独一课？】
现有课程只介绍了基础工具注册和调用，缺少高级工具使用模式。
生产级 Agent 需要更复杂的工具编排能力。

【学习目标】
- 掌握工具组合与链式调用
- 学会动态工具发现与加载
- 理解 MCP (Model Context Protocol) 协议
- 掌握工具权限分级与沙箱执行
- 学会工具缓存与性能优化

【核心概念】
- Tool Chaining（工具链）
- Dynamic Tool Discovery（动态工具发现）
- MCP Protocol（模型上下文协议）
- Tool Sandboxing（工具沙箱）
- Tool Caching（工具缓存）

==============================================================================
"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


# ============================================================================
# 第一部分：工具链（Tool Chaining）
# ============================================================================
# 【知识点】
# 工具链是将多个工具按顺序或条件组合执行的模式：
# - 顺序链：A → B → C
# - 条件链：A → (条件) → B 或 C
# - 并行链：A + B → 合并结果
# - 管道链：A 的输出作为 B 的输入


class ToolChain:
    """
    工具链：按顺序执行多个工具
    
    【使用场景】
    - 数据处理管道：获取数据 → 清洗 → 转换 → 存储
    - 多步查询：搜索 → 过滤 → 排序 → 格式化
    """

    def __init__(self, name: str):
        self.name = name
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, tool_name: str, params: Dict = None, 
                 input_from: str = None, transform: Callable = None):
        """
        添加工具链步骤
        
        Args:
            tool_name: 工具名称
            params: 固定参数
            input_from: 从上一步获取输入的字段名
            transform: 输入转换函数
        """
        self.steps.append({
            "tool": tool_name,
            "params": params or {},
            "input_from": input_from,
            "transform": transform
        })
        return self  # 支持链式调用

    def execute(self, tools_registry: Dict[str, Callable], 
                initial_input: Any = None) -> Dict[str, Any]:
        """执行工具链"""
        results = []
        current_input = initial_input

        print(f"\n  🔗 执行工具链: {self.name}")
        print(f"  {'─' * 40}")

        for i, step in enumerate(self.steps):
            # 准备输入
            if step["input_from"] and results:
                current_input = results[-1].get(step["input_from"])
            if step["transform"] and current_input is not None:
                current_input = step["transform"](current_input)

            # 合并参数
            params = {**step["params"]}
            if current_input is not None:
                params["input"] = current_input

            # 执行工具
            tool_fn = tools_registry.get(step["tool"])
            if not tool_fn:
                print(f"  ❌ 步骤 {i+1}: 工具 '{step['tool']}' 不存在")
                break

            print(f"  📌 步骤 {i+1}: [{step['tool']}] 输入: {str(current_input)[:40]}...")
            result = tool_fn(**params)
            results.append({"tool": step["tool"], "result": result})
            current_input = result
            print(f"     → 输出: {str(result)[:40]}...")

        return {"chain": self.name, "steps": results, "final": results[-1]["result"] if results else None}


# ============================================================================
# 第二部分：动态工具发现与加载
# ============================================================================
# 【知识点】
# 动态工具发现允许 Agent 在运行时发现和加载新工具：
# - 工具注册表扫描
# - 按需加载工具模块
# - 工具能力描述匹配
# - 热插拔工具


class ToolCapability(Enum):
    """工具能力标签"""
    SEARCH = "search"
    CALCULATION = "calculation"
    DATA_ACCESS = "data_access"
    COMMUNICATION = "communication"
    FILE_OPERATION = "file_operation"
    API_CALL = "api_call"


@dataclass
class ToolDescriptor:
    """工具描述符（用于动态发现）"""
    name: str
    description: str
    capabilities: Set[ToolCapability]
    parameters: Dict[str, str]
    version: str = "1.0.0"
    author: str = "system"
    is_loaded: bool = False
    module_path: Optional[str] = None


class DynamicToolRegistry:
    """
    动态工具注册中心
    
    【功能】
    - 工具描述符管理
    - 按能力搜索工具
    - 按需加载工具实现
    - 工具版本管理
    """

    def __init__(self):
        self.descriptors: Dict[str, ToolDescriptor] = {}
        self.implementations: Dict[str, Callable] = {}
        self._load_hooks: List[Callable] = []

    def register(self, descriptor: ToolDescriptor, implementation: Callable = None):
        """注册工具"""
        self.descriptors[descriptor.name] = descriptor
        if implementation:
            self.implementations[descriptor.name] = implementation
            descriptor.is_loaded = True
        print(f"  📦 注册工具: {descriptor.name} v{descriptor.version}")

    def discover(self, required_capabilities: Set[ToolCapability]) -> List[ToolDescriptor]:
        """
        按能力发现工具
        
        【原理】
        根据任务需要的能力，自动找到合适的工具
        """
        matches = []
        for desc in self.descriptors.values():
            if required_capabilities.issubset(desc.capabilities):
                matches.append(desc)
        return matches

    def search(self, query: str) -> List[ToolDescriptor]:
        """按关键词搜索工具"""
        results = []
        query_lower = query.lower()
        for desc in self.descriptors.values():
            if (query_lower in desc.name.lower() or 
                query_lower in desc.description.lower()):
                results.append(desc)
        return results

    def load_tool(self, name: str) -> bool:
        """按需加载工具实现"""
        desc = self.descriptors.get(name)
        if not desc:
            return False
        if desc.is_loaded:
            return True

        # 模拟动态加载（实际中可能是 importlib 动态导入）
        print(f"  ⏳ 加载工具: {name} from {desc.module_path}")
        desc.is_loaded = True

        # 触发加载钩子
        for hook in self._load_hooks:
            hook(name)

        return True

    def list_available(self) -> List[str]:
        """列出所有可用工具"""
        return [name for name, desc in self.descriptors.items() if desc.is_loaded]


# ============================================================================
# 第三部分：MCP (Model Context Protocol) 协议
# ============================================================================
# 【知识点】
# MCP 是 Anthropic 提出的标准协议，用于 LLM 与外部工具/数据的交互：
# - 标准化接口：统一工具描述格式
# - 服务器-客户端架构：工具作为服务提供
# - 上下文管理：动态注入上下文信息
# - 安全隔离：工具运行在独立沙箱


@dataclass
class MCPToolDefinition:
    """
    MCP 工具定义格式
    
    【协议规范】
    遵循 MCP 标准的工具定义，包含：
    - name: 工具唯一标识
    - description: 功能描述
    - inputSchema: JSON Schema 格式的输入定义
    """
    name: str
    description: str
    inputSchema: Dict[str, Any]


class MCPServer:
    """
    MCP 服务器模拟
    
    【概念】
    MCP Server 提供一组工具供 Agent（Client）调用：
    - 工具列表（tools/list）
    - 工具调用（tools/call）
    - 资源访问（resources/read）
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, MCPToolDefinition] = {}
        self.handlers: Dict[str, Callable] = {}
        self.resources: Dict[str, Any] = {}

    def add_tool(self, definition: MCPToolDefinition, handler: Callable):
        """注册工具到 MCP 服务器"""
        self.tools[definition.name] = definition
        self.handlers[definition.name] = handler

    def add_resource(self, uri: str, content: Any):
        """添加资源"""
        self.resources[uri] = content

    def handle_request(self, method: str, params: Dict = None) -> Dict[str, Any]:
        """处理 MCP 请求"""
        params = params or {}

        if method == "tools/list":
            return {
                "tools": [
                    {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
                    for t in self.tools.values()
                ]
            }
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            if tool_name in self.handlers:
                result = self.handlers[tool_name](**arguments)
                return {"content": [{"type": "text", "text": str(result)}]}
            return {"error": f"Tool '{tool_name}' not found"}
        elif method == "resources/read":
            uri = params.get("uri")
            if uri in self.resources:
                return {"content": self.resources[uri]}
            return {"error": f"Resource '{uri}' not found"}

        return {"error": f"Unknown method: {method}"}


class MCPClient:
    """
    MCP 客户端
    
    【功能】
    - 连接 MCP 服务器
    - 发现可用工具
    - 调用工具
    - 读取资源
    """

    def __init__(self):
        self.servers: List[MCPServer] = []

    def connect(self, server: MCPServer):
        """连接到 MCP 服务器"""
        self.servers.append(server)
        print(f"  🔌 已连接 MCP 服务器: {server.name} v{server.version}")

    def list_all_tools(self) -> List[Dict]:
        """列出所有服务器的工具"""
        all_tools = []
        for server in self.servers:
            response = server.handle_request("tools/list")
            for tool in response.get("tools", []):
                tool["server"] = server.name
            all_tools.extend(response.get("tools", []))
        return all_tools

    def call_tool(self, server_name: str, tool_name: str, arguments: Dict) -> Dict:
        """调用指定服务器的工具"""
        for server in self.servers:
            if server.name == server_name:
                return server.handle_request("tools/call", {
                    "name": tool_name,
                    "arguments": arguments
                })
        return {"error": f"Server '{server_name}' not connected"}


# ============================================================================
# 第四部分：工具沙箱与安全执行
# ============================================================================
# 【知识点】
# 工具执行需要安全隔离：
# - 资源限制：CPU、内存、时间
# - 权限控制：文件访问、网络访问
# - 沙箱环境：隔离执行
# - 输入验证：防止注入攻击


class SandboxConfig:
    """沙箱配置"""
    def __init__(self):
        self.max_execution_time: float = 30.0     # 最大执行时间（秒）
        self.max_memory_mb: int = 256              # 最大内存（MB）
        self.allowed_domains: Set[str] = set()     # 允许访问的域名
        self.allowed_paths: Set[str] = set()       # 允许访问的文件路径
        self.blocked_operations: Set[str] = {"eval", "exec", "__import__"}


class SandboxedToolExecutor:
    """
    沙箱工具执行器
    
    【安全机制】
    1. 输入验证：检查参数合法性
    2. 超时控制：防止无限执行
    3. 权限检查：验证资源访问权限
    4. 输出过滤：过滤敏感信息
    """

    def __init__(self, config: SandboxConfig = None):
        self.config = config or SandboxConfig()
        self.execution_log: List[Dict] = []

    def validate_input(self, tool_name: str, params: Dict) -> bool:
        """验证输入参数"""
        # 检查是否包含危险操作
        params_str = json.dumps(params)
        for blocked in self.config.blocked_operations:
            if blocked in params_str:
                print(f"  ⚠️ 拦截危险输入: '{blocked}' in params")
                return False
        return True

    def execute(self, tool_name: str, tool_fn: Callable, params: Dict) -> Dict[str, Any]:
        """在沙箱中执行工具"""
        start_time = time.time()
        
        # 1. 输入验证
        if not self.validate_input(tool_name, params):
            return {"success": False, "error": "Input validation failed"}

        # 2. 执行（带超时）
        try:
            result = tool_fn(**params)
            elapsed = time.time() - start_time

            # 3. 超时检查
            if elapsed > self.config.max_execution_time:
                return {"success": False, "error": "Execution timeout"}

            # 4. 记录日志
            self.execution_log.append({
                "tool": tool_name,
                "success": True,
                "elapsed": elapsed,
                "timestamp": datetime.now().isoformat()
            })

            return {"success": True, "result": result, "elapsed": elapsed}

        except Exception as e:
            self.execution_log.append({
                "tool": tool_name,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return {"success": False, "error": str(e)}


# ============================================================================
# 第五部分：工具缓存
# ============================================================================
# 【知识点】
# 工具缓存可以减少重复调用：
# - 相同输入返回缓存结果
# - TTL（过期时间）控制
# - LRU（最近最少使用）淘汰策略
# - 缓存预热和失效


class ToolCache:
    """
    工具调用缓存
    
    【策略】
    - 基于输入参数的哈希作为缓存键
    - TTL 过期机制
    - 最大容量限制
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.stats = {"hits": 0, "misses": 0}

    def _make_key(self, tool_name: str, params: Dict) -> str:
        """生成缓存键"""
        param_str = json.dumps(params, sort_keys=True)
        hash_val = hashlib.md5(f"{tool_name}:{param_str}".encode()).hexdigest()
        return f"{tool_name}:{hash_val}"

    def get(self, tool_name: str, params: Dict) -> Optional[Any]:
        """获取缓存"""
        key = self._make_key(tool_name, params)
        
        if key in self.cache:
            entry = self.cache[key]
            # 检查是否过期
            if datetime.now() - entry["time"] < timedelta(seconds=self.ttl_seconds):
                self.stats["hits"] += 1
                print(f"  💾 缓存命中: {tool_name}")
                return entry["result"]
            else:
                # 过期删除
                del self.cache[key]

        self.stats["misses"] += 1
        return None

    def put(self, tool_name: str, params: Dict, result: Any):
        """存入缓存"""
        key = self._make_key(tool_name, params)
        
        # LRU 淘汰
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]["time"])
            del self.cache[oldest_key]

        self.cache[key] = {
            "result": result,
            "time": datetime.now()
        }

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0}

    def display_stats(self):
        """显示缓存统计"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total * 100 if total > 0 else 0
        print(f"\n  📊 缓存统计:")
        print(f"     命中: {self.stats['hits']} | 未命中: {self.stats['misses']}")
        print(f"     命中率: {hit_rate:.1f}% | 缓存条目: {len(self.cache)}/{self.max_size}")


# ============================================================================
# 第六部分：工具组合模式
# ============================================================================

class ToolComposer:
    """
    工具组合器
    
    【模式】
    支持多种工具组合方式：
    - Sequential: 顺序执行
    - Parallel: 并行执行
    - Conditional: 条件分支
    - MapReduce: 映射-归约
    """

    def __init__(self, tools: Dict[str, Callable]):
        self.tools = tools

    def sequential(self, tool_names: List[str], initial_input: Any) -> Any:
        """顺序执行"""
        result = initial_input
        print(f"\n  📋 顺序执行: {' → '.join(tool_names)}")
        for name in tool_names:
            fn = self.tools.get(name)
            if fn:
                result = fn(input=result)
                print(f"    [{name}] → {str(result)[:30]}...")
        return result

    def parallel(self, tool_names: List[str], input_data: Any) -> List[Any]:
        """并行执行（模拟）"""
        print(f"\n  ⚡ 并行执行: {tool_names}")
        results = []
        for name in tool_names:
            fn = self.tools.get(name)
            if fn:
                result = fn(input=input_data)
                results.append({"tool": name, "result": result})
                print(f"    [{name}] → {str(result)[:30]}...")
        return results

    def conditional(self, condition_fn: Callable, 
                    true_tool: str, false_tool: str, input_data: Any) -> Any:
        """条件分支"""
        condition_result = condition_fn(input_data)
        chosen = true_tool if condition_result else false_tool
        print(f"\n  🔀 条件分支: 条件={condition_result} → [{chosen}]")
        fn = self.tools.get(chosen)
        return fn(input=input_data) if fn else None

    def map_reduce(self, map_tool: str, reduce_tool: str, 
                   input_list: List[Any]) -> Any:
        """Map-Reduce 模式"""
        print(f"\n  🗺️ Map-Reduce: map=[{map_tool}] reduce=[{reduce_tool}]")
        
        # Map 阶段
        mapped = []
        map_fn = self.tools.get(map_tool)
        for item in input_list:
            result = map_fn(input=item) if map_fn else item
            mapped.append(result)
            print(f"    Map: {str(item)[:20]} → {str(result)[:20]}")
        
        # Reduce 阶段
        reduce_fn = self.tools.get(reduce_tool)
        final = reduce_fn(input=mapped) if reduce_fn else mapped
        print(f"    Reduce: {len(mapped)} items → {str(final)[:30]}")
        
        return final


# ============================================================================
# 演示运行
# ============================================================================

def demo_tool_chain():
    """演示工具链"""
    print("\n" + "🔧" * 30)
    print("第十八课：Advanced Tool Patterns（高级工具模式）")
    print("🔧" * 30)

    print("\n" + "=" * 60)
    print("演示1：工具链（Tool Chaining）")
    print("=" * 60)

    # 定义工具
    def fetch_data(input=None, **kwargs):
        return {"records": [1, 2, 3, 4, 5], "source": "database"}

    def filter_data(input=None, **kwargs):
        records = input.get("records", []) if isinstance(input, dict) else []
        return {"records": [r for r in records if r > 2], "filtered": True}

    def transform_data(input=None, **kwargs):
        records = input.get("records", []) if isinstance(input, dict) else []
        return {"records": [r * 10 for r in records], "transformed": True}

    def format_output(input=None, **kwargs):
        records = input.get("records", []) if isinstance(input, dict) else []
        return f"结果: {', '.join(map(str, records))}"

    tools = {
        "fetch": fetch_data,
        "filter": filter_data,
        "transform": transform_data,
        "format": format_output
    }

    # 构建工具链
    chain = ToolChain("数据处理管道")
    chain.add_step("fetch") \
         .add_step("filter", input_from="result") \
         .add_step("transform", input_from="result") \
         .add_step("format", input_from="result")

    result = chain.execute(tools)
    print(f"\n  ✅ 最终结果: {result['final']}")


def demo_dynamic_discovery():
    """演示动态工具发现"""
    print("\n" + "=" * 60)
    print("演示2：动态工具发现")
    print("=" * 60)

    registry = DynamicToolRegistry()

    # 注册工具描述符
    registry.register(ToolDescriptor(
        name="web_search",
        description="搜索互联网获取信息",
        capabilities={ToolCapability.SEARCH, ToolCapability.API_CALL},
        parameters={"query": "搜索关键词"}
    ), implementation=lambda **kw: f"搜索结果: {kw}")

    registry.register(ToolDescriptor(
        name="calculator",
        description="数学计算工具",
        capabilities={ToolCapability.CALCULATION},
        parameters={"expression": "数学表达式"}
    ), implementation=lambda **kw: eval(kw.get("expression", "0")))

    registry.register(ToolDescriptor(
        name="file_reader",
        description="读取文件内容",
        capabilities={ToolCapability.FILE_OPERATION, ToolCapability.DATA_ACCESS},
        parameters={"path": "文件路径"}
    ))

    registry.register(ToolDescriptor(
        name="email_sender",
        description="发送电子邮件",
        capabilities={ToolCapability.COMMUNICATION},
        parameters={"to": "收件人", "subject": "主题", "body": "内容"}
    ))

    # 按能力发现
    print("\n  🔍 需要搜索+计算能力的工具:")
    search_tools = registry.discover({ToolCapability.SEARCH})
    for t in search_tools:
        print(f"    → {t.name}: {t.description}")

    print("\n  🔍 搜索 '文件' 相关工具:")
    file_tools = registry.search("文件")
    for t in file_tools:
        print(f"    → {t.name}: {t.description}")

    print(f"\n  📋 已加载工具: {registry.list_available()}")


def demo_mcp():
    """演示 MCP 协议"""
    print("\n" + "=" * 60)
    print("演示3：MCP 协议")
    print("=" * 60)

    # 创建 MCP 服务器
    weather_server = MCPServer("weather-service", "2.0.0")
    weather_server.add_tool(
        MCPToolDefinition(
            name="get_weather",
            description="获取指定城市天气",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        ),
        handler=lambda city: f"{city}：晴天，28°C"
    )
    weather_server.add_resource("weather://beijing", {"temp": 28, "condition": "晴"})

    db_server = MCPServer("database-service", "1.5.0")
    db_server.add_tool(
        MCPToolDefinition(
            name="query",
            description="执行数据库查询",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL 查询语句"}
                },
                "required": ["sql"]
            }
        ),
        handler=lambda sql: f"查询结果: [{sql}]"
    )

    # 创建客户端
    client = MCPClient()
    client.connect(weather_server)
    client.connect(db_server)

    # 发现工具
    print("\n  📋 所有可用工具:")
    for tool in client.list_all_tools():
        print(f"    [{tool.get('server')}] {tool['name']}: {tool['description']}")

    # 调用工具
    print("\n  🔄 调用工具:")
    result = client.call_tool("weather-service", "get_weather", {"city": "北京"})
    print(f"    天气查询: {result}")

    result = client.call_tool("database-service", "query", {"sql": "SELECT * FROM users"})
    print(f"    数据库查询: {result}")


def demo_sandbox():
    """演示沙箱执行"""
    print("\n" + "=" * 60)
    print("演示4：沙箱安全执行")
    print("=" * 60)

    config = SandboxConfig()
    config.max_execution_time = 5.0
    config.blocked_operations = {"eval", "exec", "__import__", "rm -rf"}

    executor = SandboxedToolExecutor(config)

    # 安全工具
    def safe_tool(input="hello", **kwargs):
        return f"处理: {input}"

    # 危险工具（模拟）
    def dangerous_tool(input="eval('malicious')", **kwargs):
        return input

    print("\n  ✅ 安全调用:")
    result = executor.execute("safe_tool", safe_tool, {"input": "test data"})
    print(f"    结果: {result}")

    print("\n  ⚠️ 危险调用拦截:")
    result = executor.execute("dangerous_tool", dangerous_tool, {"input": "eval('hack')"})
    print(f"    结果: {result}")

    print(f"\n  📊 执行日志: {len(executor.execution_log)} 条")


def demo_caching():
    """演示工具缓存"""
    print("\n" + "=" * 60)
    print("演示5：工具缓存")
    print("=" * 60)

    cache = ToolCache(max_size=50, ttl_seconds=60)

    # 模拟慢速工具
    call_count = 0
    def slow_weather_tool(city=""):
        nonlocal call_count
        call_count += 1
        time.sleep(0.1)  # 模拟网络延迟
        return f"{city}: 28°C, 晴天"

    # 多次调用（相同参数）
    cities = ["北京", "上海", "北京", "广州", "上海", "北京"]
    for city in cities:
        # 先查缓存
        cached = cache.get("weather", {"city": city})
        if cached is None:
            # 缓存未命中，实际调用
            result = slow_weather_tool(city=city)
            cache.put("weather", {"city": city}, result)
        print(f"    查询 {city}")

    cache.display_stats()
    print(f"\n  💡 实际调用次数: {call_count} (缓存节省了 {len(cities) - call_count} 次)")


def demo_composition():
    """演示工具组合"""
    print("\n" + "=" * 60)
    print("演示6：工具组合模式")
    print("=" * 60)

    def uppercase_tool(input="", **kw):
        return str(input).upper()

    def reverse_tool(input="", **kw):
        return str(input)[::-1]

    def add_prefix_tool(input="", **kw):
        return f"[RESULT] {input}"

    def length_tool(input="", **kw):
        return f"长度: {len(str(input))}"

    composer = ToolComposer({
        "upper": uppercase_tool,
        "reverse": reverse_tool,
        "prefix": add_prefix_tool,
        "length": length_tool
    })

    # 顺序组合
    result = composer.sequential(["upper", "reverse", "prefix"], "hello world")
    print(f"  最终结果: {result}")

    # 并行组合
    results = composer.parallel(["upper", "reverse", "length"], "test")
    print(f"  并行结果: {[r['result'] for r in results]}")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    demo_tool_chain()
    demo_dynamic_discovery()
    demo_mcp()
    demo_sandbox()
    demo_caching()
    demo_composition()

    print("\n" + "=" * 60)
    print("✅ 第十八课学习完成！")
    print("=" * 60)
    print("""
  本课要点总结：
1. 工具链（Tool Chaining）将多个工具按管道组合执行
2. 动态工具发现允许按能力搜索和按需加载工具
3. MCP 协议提供标准化的工具交互接口
4. 沙箱执行保障工具调用安全
5. 工具缓存减少重复调用，提升性能
6. 工具组合模式：顺序、并行、条件、Map-Reduce

  进阶方向：
- 实现真实的 MCP Server/Client
- 工具版本管理与灰度发布
- 分布式工具执行
- 工具编排 DSL（领域特定语言）
    """)
