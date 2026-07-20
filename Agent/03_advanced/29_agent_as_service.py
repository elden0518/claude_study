"""
第29课：Agent即服务（Agent-as-a-Service, AaaS）

核心知识点：
1. 多租户架构 - 租户隔离、资源共享、配置管理
2. API网关 - 请求路由、认证鉴权、限流熔断
3. SLA管理 - 服务等级目标、性能监控、违约检测
4. 服务编排 - 多Agent协调、服务发现、负载均衡
5. 计费与配额 - 使用量追踪、配额管理、计费策略

架构要点：
- 租户隔离：每个租户有独立的配置、会话、记忆空间
- API网关：统一入口，处理认证、限流、路由
- SLA管理：定义服务等级目标，监控响应时间和可用性
- 服务编排：根据请求类型分发到不同Agent实例
- 计费系统：按Token/请求次数/时间计费

实际应用场景：
- SaaS平台为多个企业提供Agent服务
- 不同租户有不同的模型配置和权限
- 需要保证SLA（如99.9%可用性、<2s响应时间）
- 需要按使用量计费和配额管理
"""

import asyncio
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict


# ============================================================
# 第一部分：多租户架构
# ============================================================

print("\n" + "=" * 60)
print("第29课：Agent即服务（Agent-as-a-Service）")
print("=" * 60)

print("""
-- 什么是 Agent-as-a-Service --

Agent-as-a-Service 是将 Agent 能力以云服务形式提供给多个租户：

  租户A（企业客服）  --\\
  租户B（数据分析）  ----  AaaS Platform  -->  Agent实例池
  租户C（代码审查）  --/        |
                               API网关
                               
核心挑战：
  1. 租户隔离：数据、配置、资源互不干扰
  2. 弹性伸缩：根据负载动态分配资源
  3. 服务质量：保证SLA（可用性、响应时间）
  4. 成本控制：按使用量计费，防止资源滥用

-- 多租户隔离级别 --

  Level 1: 共享一切（成本低，隔离弱）
    所有租户共用同一Agent实例和数据库
    
  Level 2: 数据隔离（推荐）
    共享Agent实例，但数据（记忆/会话）按租户隔离
    
  Level 3: 完全隔离（成本高，安全性强）
    每个租户独立Agent实例、独立数据库、独立配置
""")


@dataclass
class TenantConfig:
    """租户配置"""
    tenant_id: str
    name: str
    plan: str  # free, basic, pro, enterprise
    max_tokens_per_day: int = 100000
    max_concurrent_requests: int = 5
    allowed_models: List[str] = field(default_factory=lambda: ["gpt-3.5"])
    custom_system_prompt: str = ""
    rate_limit_per_minute: int = 30
    enabled_features: List[str] = field(default_factory=lambda: ["chat"])
    created_at: datetime = field(default_factory=datetime.now)


class TenantManager:
    """租户管理器 - 处理租户生命周期和配置"""
    
    def __init__(self):
        self.tenants: Dict[str, TenantConfig] = {}
        self.tenant_usage: Dict[str, Dict] = defaultdict(
            lambda: {"tokens_today": 0, "requests_today": 0, "last_reset": datetime.now()}
        )
    
    def register_tenant(self, config: TenantConfig) -> str:
        """注册新租户"""
        self.tenants[config.tenant_id] = config
        print(f"  [TenantMgr] 注册租户: {config.name} (plan: {config.plan})")
        return config.tenant_id
    
    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """获取租户配置"""
        return self.tenants.get(tenant_id)
    
    def check_quota(self, tenant_id: str) -> Dict[str, Any]:
        """检查租户配额"""
        config = self.tenants.get(tenant_id)
        usage = self.tenant_usage[tenant_id]
        
        if not config:
            return {"allowed": False, "reason": "租户不存在"}
        
        # 检查日重置
        now = datetime.now()
        if (now - usage["last_reset"]).days >= 1:
            usage["tokens_today"] = 0
            usage["requests_today"] = 0
            usage["last_reset"] = now
        
        tokens_remaining = config.max_tokens_per_day - usage["tokens_today"]
        requests_remaining = config.rate_limit_per_minute - usage["requests_today"]
        
        allowed = tokens_remaining > 0 and requests_remaining > 0
        return {
            "allowed": allowed,
            "tokens_remaining": max(0, tokens_remaining),
            "requests_remaining": max(0, requests_remaining),
            "plan": config.plan,
            "reason": None if allowed else "配额已用尽"
        }
    
    def record_usage(self, tenant_id: str, tokens: int):
        """记录使用量"""
        self.tenant_usage[tenant_id]["tokens_today"] += tokens
        self.tenant_usage[tenant_id]["requests_today"] += 1
    
    def list_tenants(self) -> List[Dict]:
        """列出所有租户"""
        result = []
        for tid, config in self.tenants.items():
            usage = self.tenant_usage[tid]
            result.append({
                "tenant_id": tid,
                "name": config.name,
                "plan": config.plan,
                "tokens_used": usage["tokens_today"],
                "requests": usage["requests_today"]
            })
        return result


# ============================================================
# 第二部分：API网关
# ============================================================

@dataclass
class APIRequest:
    """API请求模型"""
    request_id: str
    tenant_id: str
    api_key: str
    method: str
    path: str
    body: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class APIResponse:
    """API响应模型"""
    request_id: str
    status_code: int
    body: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0
    tenant_id: str = ""


class RateLimiter:
    """限流器 - 滑动窗口算法"""
    
    def __init__(self):
        self.windows: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        """检查是否允许请求"""
        now = time.time()
        # 清理过期记录
        self.windows[key] = [
            t for t in self.windows[key]
            if now - t < window_seconds
        ]
        if len(self.windows[key]) >= limit:
            return False
        self.windows[key].append(now)
        return True
    
    def get_remaining(self, key: str, limit: int, window_seconds: int = 60) -> int:
        """获取剩余配额"""
        now = time.time()
        active = [t for t in self.windows[key] if now - t < window_seconds]
        return max(0, limit - len(active))


class APIGateway:
    """API网关 - 统一入口管理"""
    
    def __init__(self, tenant_manager: TenantManager):
        self.tenant_manager = tenant_manager
        self.rate_limiter = RateLimiter()
        self.api_keys: Dict[str, str] = {}  # api_key -> tenant_id
        self.request_log: List[APIResponse] = []
        self.middleware_chain: List[callable] = []
    
    def register_api_key(self, api_key: str, tenant_id: str):
        """注册API Key"""
        self.api_keys[api_key] = tenant_id
    
    def add_middleware(self, middleware_fn):
        """添加中间件"""
        self.middleware_chain.append(middleware_fn)
    
    def authenticate(self, request: APIRequest) -> Optional[str]:
        """认证请求，返回tenant_id"""
        api_key = request.api_key or request.headers.get("X-API-Key", "")
        tenant_id = self.api_keys.get(api_key)
        if not tenant_id:
            return None
        return tenant_id
    
    async def handle_request(self, request: APIRequest) -> APIResponse:
        """处理请求的完整流程"""
        start_time = time.time()
        
        # 1. 认证
        tenant_id = self.authenticate(request)
        if not tenant_id:
            return APIResponse(
                request_id=request.request_id,
                status_code=401,
                body={"error": "Invalid API Key"}
            )
        request.tenant_id = tenant_id
        
        # 2. 限流检查
        config = self.tenant_manager.get_tenant(tenant_id)
        rate_key = f"rate:{tenant_id}"
        if not self.rate_limiter.is_allowed(rate_key, config.rate_limit_per_minute):
            return APIResponse(
                request_id=request.request_id,
                status_code=429,
                body={"error": "Rate limit exceeded", "retry_after": 60}
            )
        
        # 3. 配额检查
        quota = self.tenant_manager.check_quota(tenant_id)
        if not quota["allowed"]:
            return APIResponse(
                request_id=request.request_id,
                status_code=403,
                body={"error": "Quota exceeded", "detail": quota["reason"]}
            )
        
        # 4. 执行中间件
        for middleware in self.middleware_chain:
            result = await middleware(request) if asyncio.iscoroutinefunction(middleware) else middleware(request)
            if result is not None:
                return result
        
        # 5. 路由到对应处理器
        response = await self._route_and_process(request)
        
        # 6. 记录使用量
        tokens_used = response.body.get("tokens_used", 0)
        self.tenant_manager.record_usage(tenant_id, tokens_used)
        
        # 7. 计算延迟
        response.latency_ms = (time.time() - start_time) * 1000
        response.tenant_id = tenant_id
        
        # 8. 记录日志
        self.request_log.append(response)
        
        return response
    
    async def _route_and_process(self, request: APIRequest) -> APIResponse:
        """路由并处理请求"""
        # 模拟Agent处理
        path = request.path
        
        if path == "/v1/chat":
            # 模拟聊天处理
            message = request.body.get("message", "")
            response_text = f"[Agent响应] 收到消息: {message[:50]}"
            tokens = len(message) + len(response_text)
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                body={
                    "response": response_text,
                    "tokens_used": tokens,
                    "model": "gpt-4"
                }
            )
        elif path == "/v1/status":
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                body={"status": "healthy", "version": "1.0.0"}
            )
        elif path == "/v1/usage":
            quota = self.tenant_manager.check_quota(request.tenant_id)
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                body=quota
            )
        else:
            return APIResponse(
                request_id=request.request_id,
                status_code=404,
                body={"error": "Endpoint not found"}
            )


# ============================================================
# 第三部分：SLA管理
# ============================================================

@dataclass
class SLATarget:
    """SLA目标定义"""
    name: str
    metric: str  # latency_p99, availability, error_rate
    target_value: float  # 目标值（ms或百分比）
    operator: str  # "<" means metric must be < target, ">" means must be >
    plan_tier: str  # 适用的计划等级


class SLAMonitor:
    """SLA监控器"""
    
    def __init__(self):
        self.targets: List[SLATarget] = []
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.violations: List[Dict] = []
        self.incidents: List[Dict] = []
    
    def add_target(self, target: SLATarget):
        """添加SLA目标"""
        self.targets.append(target)
    
    def record_metric(self, metric_name: str, value: float):
        """记录指标值"""
        self.metrics[metric_name].append(value)
        # 只保留最近1000条
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name] = self.metrics[metric_name][-1000:]
    
    def check_violations(self) -> List[Dict]:
        """检查是否有SLA违约"""
        violations = []
        
        for target in self.targets:
            values = self.metrics.get(target.metric, [])
            if not values:
                continue
            
            # 计算当前值
            if "p99" in target.metric:
                current = sorted(values)[int(len(values) * 0.99)] if values else 0
            elif "avg" in target.metric:
                current = sum(values) / len(values) if values else 0
            else:
                current = values[-1] if values else 0
            
            # 检查是否违约
            violated = False
            if target.operator == "<" and current >= target.target_value:
                violated = True
            elif target.operator == ">" and current <= target.target_value:
                violated = True
            
            if violated:
                violation = {
                    "target": target.name,
                    "metric": target.metric,
                    "current_value": current,
                    "target_value": target.target_value,
                    "timestamp": datetime.now(),
                    "severity": "critical" if target.plan_tier == "enterprise" else "warning"
                }
                violations.append(violation)
                self.violations.append(violation)
        
        return violations
    
    def get_report(self) -> Dict:
        """生成SLA报告"""
        report = {
            "timestamp": datetime.now(),
            "targets": [],
            "total_violations": len(self.violations),
            "health_status": "healthy"
        }
        
        for target in self.targets:
            values = self.metrics.get(target.metric, [])
            if not values:
                continue
            
            current = sum(values) / len(values)
            target_met = True
            if target.operator == "<" and current >= target.target_value:
                target_met = False
            elif target.operator == ">" and current <= target.target_value:
                target_met = False
            
            report["targets"].append({
                "name": target.name,
                "metric": target.metric,
                "current": round(current, 2),
                "target": target.target_value,
                "met": target_met
            })
        
        if any(not t["met"] for t in report["targets"]):
            report["health_status"] = "degraded"
        
        return report


# ============================================================
# 第四部分：服务编排与负载均衡
# ============================================================

@dataclass
class AgentInstance:
    """Agent服务实例"""
    instance_id: str
    host: str
    port: int
    status: str  # healthy, degraded, down
    load: float  # 0.0 ~ 1.0
    capabilities: List[str]
    last_heartbeat: datetime = field(default_factory=datetime.now)
    active_requests: int = 0


class ServiceDiscovery:
    """服务发现 - 管理Agent实例注册与发现"""
    
    def __init__(self):
        self.instances: Dict[str, AgentInstance] = {}
        self.health_checks: Dict[str, List[bool]] = defaultdict(list)
    
    def register(self, instance: AgentInstance):
        """注册Agent实例"""
        self.instances[instance.instance_id] = instance
        print(f"  [Discovery] 注册实例: {instance.instance_id} @ {instance.host}:{instance.port}")
    
    def deregister(self, instance_id: str):
        """注销实例"""
        if instance_id in self.instances:
            del self.instances[instance_id]
            print(f"  [Discovery] 注销实例: {instance_id}")
    
    def discover(self, capability: str = None) -> List[AgentInstance]:
        """发现可用实例"""
        healthy = [
            inst for inst in self.instances.values()
            if inst.status == "healthy"
        ]
        if capability:
            healthy = [
                inst for inst in healthy
                if capability in inst.capabilities
            ]
        return healthy
    
    def report_health(self, instance_id: str, is_healthy: bool):
        """上报健康状态"""
        if instance_id in self.instances:
            self.instances[instance_id].last_heartbeat = datetime.now()
            self.health_checks[instance_id].append(is_healthy)
            # 最近3次都不健康则标记为down
            recent = self.health_checks[instance_id][-3:]
            if len(recent) >= 3 and all(not h for h in recent):
                self.instances[instance_id].status = "down"
                print(f"  [Discovery] 实例 {instance_id} 标记为 DOWN")


class LoadBalancer:
    """负载均衡器 - 多种策略"""
    
    def __init__(self, discovery: ServiceDiscovery):
        self.discovery = discovery
        self.strategy = "least_load"  # round_robin, least_load, weighted
    
    def select_instance(self, capability: str = None) -> Optional[AgentInstance]:
        """根据策略选择实例"""
        available = self.discovery.discover(capability)
        if not available:
            return None
        
        if self.strategy == "round_robin":
            # 简单轮询
            idx = hash(time.time()) % len(available)
            return available[idx]
        elif self.strategy == "least_load":
            # 最小负载
            return min(available, key=lambda x: x.load)
        elif self.strategy == "weighted":
            # 加权随机（按剩余容量）
            weights = [1.0 - inst.load for inst in available]
            total = sum(weights)
            if total == 0:
                return available[0]
            # 简化版：选权重最大的
            best_idx = weights.index(max(weights))
            return available[best_idx]
        
        return available[0] if available else None


class ServiceOrchestrator:
    """服务编排器 - 协调多Agent完成复杂任务"""
    
    def __init__(self, discovery: ServiceDiscovery, lb: LoadBalancer):
        self.discovery = discovery
        self.lb = lb
        self.execution_log: List[Dict] = []
    
    async def execute_pipeline(self, task: str, steps: List[Dict]) -> Dict:
        """执行Agent管道 - 串联多个Agent"""
        print(f"\n  [Pipeline] 开始执行: {task}")
        results = []
        context = {"original_task": task}
        
        for i, step in enumerate(steps):
            step_name = step.get("name", f"step_{i}")
            capability = step.get("capability", "general")
            
            # 选择合适的Agent实例
            instance = self.lb.select_instance(capability)
            if not instance:
                print(f"  [Pipeline] 步骤 {step_name}: 无可用实例!")
                return {"status": "failed", "error": f"No instance for {capability}"}
            
            # 模拟执行
            print(f"  [Pipeline] 步骤 {step_name} -> 实例 {instance.instance_id} (load: {instance.load:.2f})")
            step_result = {
                "step": step_name,
                "instance": instance.instance_id,
                "result": f"完成: {step.get('description', '')}",
                "latency_ms": 50 + i * 10
            }
            results.append(step_result)
            context[f"step_{i}_result"] = step_result["result"]
            
            # 更新实例负载
            instance.active_requests += 1
            instance.load = min(1.0, instance.load + 0.1)
        
        # 释放负载
        for inst in self.discovery.instances.values():
            inst.active_requests = max(0, inst.active_requests - 1)
            inst.load = max(0.0, inst.load - 0.05)
        
        return {
            "status": "completed",
            "task": task,
            "steps": results,
            "total_latency_ms": sum(r["latency_ms"] for r in results)
        }


# ============================================================
# 第五部分：计费系统
# ============================================================

@dataclass
class BillingRecord:
    """计费记录"""
    tenant_id: str
    timestamp: datetime
    tokens_used: int
    requests_count: int
    model_used: str
    cost: float


class BillingEngine:
    """计费引擎"""
    
    # 模型单价（每1000 token的价格，单位：元）
    PRICING = {
        "gpt-3.5": 0.002,
        "gpt-4": 0.03,
        "gpt-4-turbo": 0.01,
        "claude-3": 0.025,
    }
    
    # 套餐折扣
    PLAN_DISCOUNTS = {
        "free": 1.0,
        "basic": 0.9,
        "pro": 0.75,
        "enterprise": 0.6
    }
    
    def __init__(self):
        self.records: List[BillingRecord] = []
        self.invoices: Dict[str, List[BillingRecord]] = defaultdict(list)
    
    def calculate_cost(self, tenant_id: str, tokens: int, model: str, plan: str) -> float:
        """计算费用"""
        price_per_1k = self.PRICING.get(model, 0.01)
        base_cost = (tokens / 1000) * price_per_1k
        discount = self.PLAN_DISCOUNTS.get(plan, 1.0)
        return round(base_cost * discount, 6)
    
    def record_usage(self, tenant_id: str, tokens: int, model: str, plan: str):
        """记录使用并计费"""
        cost = self.calculate_cost(tenant_id, tokens, model, plan)
        record = BillingRecord(
            tenant_id=tenant_id,
            timestamp=datetime.now(),
            tokens_used=tokens,
            requests_count=1,
            model_used=model,
            cost=cost
        )
        self.records.append(record)
        self.invoices[tenant_id].append(record)
        return cost
    
    def get_invoice(self, tenant_id: str, period_days: int = 30) -> Dict:
        """生成账单"""
        cutoff = datetime.now() - timedelta(days=period_days)
        records = [r for r in self.invoices.get(tenant_id, []) if r.timestamp > cutoff]
        
        total_tokens = sum(r.tokens_used for r in records)
        total_cost = sum(r.cost for r in records)
        total_requests = sum(r.requests_count for r in records)
        
        # 按模型分组
        by_model = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "requests": 0})
        for r in records:
            by_model[r.model_used]["tokens"] += r.tokens_used
            by_model[r.model_used]["cost"] += r.cost
            by_model[r.model_used]["requests"] += r.requests_count
        
        return {
            "tenant_id": tenant_id,
            "period_days": period_days,
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "total_cost": round(total_cost, 4),
            "by_model": dict(by_model),
            "records_count": len(records)
        }


# ============================================================
# 演示与总结
# ============================================================

async def demo_multi_tenant():
    """演示多租户管理"""
    print("\n" + "=" * 60)
    print("演示1: 多租户注册与配额管理")
    print("=" * 60)
    
    mgr = TenantManager()
    
    # 注册租户
    mgr.register_tenant(TenantConfig(
        tenant_id="tenant_001", name="企业A-客服",
        plan="enterprise", max_tokens_per_day=1000000,
        rate_limit_per_minute=100, allowed_models=["gpt-4", "gpt-3.5"]
    ))
    mgr.register_tenant(TenantConfig(
        tenant_id="tenant_002", name="开发者B",
        plan="pro", max_tokens_per_day=500000,
        rate_limit_per_minute=60, allowed_models=["gpt-4", "gpt-3.5"]
    ))
    mgr.register_tenant(TenantConfig(
        tenant_id="tenant_003", name="试用者C",
        plan="free", max_tokens_per_day=10000,
        rate_limit_per_minute=10, allowed_models=["gpt-3.5"]
    ))
    
    # 检查配额
    print("\n  -- 配额检查 --")
    for tid in ["tenant_001", "tenant_002", "tenant_003"]:
        quota = mgr.check_quota(tid)
        config = mgr.get_tenant(tid)
        print(f"  {config.name} ({config.plan}): "
              f"tokens余量={quota['tokens_remaining']}, "
              f"请求余量={quota['requests_remaining']}")
    
    # 模拟使用
    print("\n  -- 模拟使用 --")
    mgr.record_usage("tenant_003", 5000)
    mgr.record_usage("tenant_003", 4000)
    quota = mgr.check_quota("tenant_003")
    print(f"  试用者C 使用9000 tokens后: 剩余={quota['tokens_remaining']}")
    mgr.record_usage("tenant_003", 2000)
    quota = mgr.check_quota("tenant_003")
    print(f"  试用者C 再使用2000后: 剩余={quota['tokens_remaining']}, 允许={quota['allowed']}")


async def demo_api_gateway():
    """演示API网关"""
    print("\n" + "=" * 60)
    print("演示2: API网关 - 认证/限流/路由")
    print("=" * 60)
    
    # 初始化
    mgr = TenantManager()
    mgr.register_tenant(TenantConfig(
        tenant_id="t1", name="测试租户", plan="pro",
        rate_limit_per_minute=5
    ))
    
    gateway = APIGateway(mgr)
    gateway.register_api_key("sk-test-123", "t1")
    gateway.register_api_key("sk-invalid", "unknown")
    
    # 测试1: 正常请求
    print("\n  -- 测试1: 正常请求 --")
    req = APIRequest(
        request_id="req_001", tenant_id="", api_key="sk-test-123",
        method="POST", path="/v1/chat",
        body={"message": "你好，请帮我分析数据"}
    )
    resp = await gateway.handle_request(req)
    print(f"  状态码: {resp.status_code}")
    print(f"  响应: {resp.body.get('response', '')[:60]}...")
    print(f"  延迟: {resp.latency_ms:.1f}ms")
    
    # 测试2: 无效API Key
    print("\n  -- 测试2: 无效API Key --")
    req2 = APIRequest(
        request_id="req_002", tenant_id="", api_key="sk-bad-key",
        method="POST", path="/v1/chat", body={"message": "test"}
    )
    resp2 = await gateway.handle_request(req2)
    print(f"  状态码: {resp2.status_code}")
    print(f"  错误: {resp2.body.get('error', '')}")
    
    # 测试3: 查看使用量
    print("\n  -- 测试3: 查看使用量 --")
    req3 = APIRequest(
        request_id="req_003", tenant_id="", api_key="sk-test-123",
        method="GET", path="/v1/usage", body={}
    )
    resp3 = await gateway.handle_request(req3)
    print(f"  状态码: {resp3.status_code}")
    print(f"  配额: {resp3.body}")
    
    # 测试4: 404路径
    print("\n  -- 测试4: 不存在的端点 --")
    req4 = APIRequest(
        request_id="req_004", tenant_id="", api_key="sk-test-123",
        method="GET", path="/v1/nonexistent", body={}
    )
    resp4 = await gateway.handle_request(req4)
    print(f"  状态码: {resp4.status_code}")
    print(f"  错误: {resp4.body.get('error', '')}")


async def demo_sla():
    """演示SLA监控"""
    print("\n" + "=" * 60)
    print("演示3: SLA监控与违约检测")
    print("=" * 60)
    
    monitor = SLAMonitor()
    
    # 设置SLA目标
    monitor.add_target(SLATarget(
        name="P99延迟", metric="latency_p99",
        target_value=2000, operator="<", plan_tier="enterprise"
    ))
    monitor.add_target(SLATarget(
        name="可用性", metric="availability",
        target_value=99.9, operator=">", plan_tier="enterprise"
    ))
    monitor.add_target(SLATarget(
        name="错误率", metric="error_rate",
        target_value=1.0, operator="<", plan_tier="pro"
    ))
    
    # 模拟正常指标
    print("\n  -- 模拟正常指标 --")
    for i in range(20):
        monitor.record_metric("latency_p99", 800 + i * 50)
        monitor.record_metric("availability", 99.95)
        monitor.record_metric("error_rate", 0.1 + i * 0.02)
    
    violations = monitor.check_violations()
    report = monitor.get_report()
    print(f"  健康状态: {report['health_status']}")
    for t in report["targets"]:
        status = "OK" if t["met"] else "VIOLATED"
        print(f"  [{status}] {t['name']}: 当前={t['current']}, 目标={t['target']}")
    
    # 模拟异常（延迟飙升）
    print("\n  -- 模拟异常（延迟飙升）--")
    for i in range(10):
        monitor.record_metric("latency_p99", 3000 + i * 200)
    
    violations = monitor.check_violations()
    report = monitor.get_report()
    print(f"  健康状态: {report['health_status']}")
    for t in report["targets"]:
        status = "OK" if t["met"] else "VIOLATED"
        print(f"  [{status}] {t['name']}: 当前={t['current']}, 目标={t['target']}")
    if violations:
        print(f"  违约次数: {len(violations)}")
        for v in violations[-2:]:
            print(f"    - {v['target']}: {v['current_value']:.0f} (目标: {v['target_value']}) [{v['severity']}]")


async def demo_service_orchestration():
    """演示服务编排"""
    print("\n" + "=" * 60)
    print("演示4: 服务发现与负载均衡")
    print("=" * 60)
    
    discovery = ServiceDiscovery()
    
    # 注册Agent实例
    instances = [
        AgentInstance("agent-1", "10.0.1.1", 8001, "healthy", 0.3, ["chat", "rag"]),
        AgentInstance("agent-2", "10.0.1.2", 8002, "healthy", 0.5, ["chat", "code"]),
        AgentInstance("agent-3", "10.0.1.3", 8003, "healthy", 0.2, ["rag", "code"]),
        AgentInstance("agent-4", "10.0.1.4", 8004, "degraded", 0.8, ["chat"]),
    ]
    for inst in instances:
        discovery.register(inst)
    
    # 服务发现
    print("\n  -- 服务发现 --")
    chat_instances = discovery.discover("chat")
    print(f"  支持chat的实例: {[i.instance_id for i in chat_instances]}")
    rag_instances = discovery.discover("rag")
    print(f"  支持rag的实例: {[i.instance_id for i in rag_instances]}")
    
    # 负载均衡
    lb = LoadBalancer(discovery)
    print("\n  -- 负载均衡（最小负载策略）--")
    for _ in range(5):
        selected = lb.select_instance("chat")
        if selected:
            print(f"  选中: {selected.instance_id} (load: {selected.load:.2f})")
    
    # 健康检查
    print("\n  -- 健康检查 --")
    discovery.report_health("agent-1", True)
    discovery.report_health("agent-2", False)
    discovery.report_health("agent-2", False)
    discovery.report_health("agent-2", False)
    print(f"  agent-2 连续3次不健康 -> 状态: {discovery.instances['agent-2'].status}")


async def demo_pipeline():
    """演示多Agent管道编排"""
    print("\n" + "=" * 60)
    print("演示5: 多Agent管道编排")
    print("=" * 60)
    
    discovery = ServiceDiscovery()
    # 注册专用Agent
    for i, (caps, load) in enumerate([
        (["nlp"], 0.2), (["rag"], 0.3), (["analysis"], 0.4),
        (["summary"], 0.1), (["general"], 0.5)
    ]):
        discovery.register(AgentInstance(
            f"svc-{i+1}", f"10.0.2.{i+1}", 9000+i+1,
            "healthy", load, caps
        ))
    
    lb = LoadBalancer(discovery)
    orchestrator = ServiceOrchestrator(discovery, lb)
    
    # 定义管道
    pipeline_steps = [
        {"name": "意图识别", "capability": "nlp", "description": "识别用户意图"},
        {"name": "知识检索", "capability": "rag", "description": "检索相关知识"},
        {"name": "数据分析", "capability": "analysis", "description": "分析数据内容"},
        {"name": "生成摘要", "capability": "summary", "description": "生成响应摘要"},
    ]
    
    result = await orchestrator.execute_pipeline("复杂数据分析任务", pipeline_steps)
    print(f"\n  -- 执行结果 --")
    print(f"  状态: {result['status']}")
    print(f"  总延迟: {result['total_latency_ms']}ms")
    print(f"  步骤数: {len(result['steps'])}")


async def demo_billing():
    """演示计费系统"""
    print("\n" + "=" * 60)
    print("演示6: 计费系统")
    print("=" * 60)
    
    billing = BillingEngine()
    
    # 模拟不同租户的使用
    usage_data = [
        ("tenant_A", 15000, "gpt-4", "enterprise"),
        ("tenant_A", 8000, "gpt-3.5", "enterprise"),
        ("tenant_B", 50000, "gpt-4", "pro"),
        ("tenant_B", 20000, "gpt-4-turbo", "pro"),
        ("tenant_C", 3000, "gpt-3.5", "free"),
    ]
    
    print("\n  -- 记录使用 --")
    for tenant_id, tokens, model, plan in usage_data:
        cost = billing.record_usage(tenant_id, tokens, model, plan)
        print(f"  {tenant_id}: {tokens} tokens ({model}) -> 费用: {cost:.4f} 元")
    
    # 生成账单
    print("\n  -- 生成账单 --")
    for tid in ["tenant_A", "tenant_B", "tenant_C"]:
        invoice = billing.get_invoice(tid)
        print(f"\n  [{tid}]")
        print(f"    总tokens: {invoice['total_tokens']}")
        print(f"    总请求: {invoice['total_requests']}")
        print(f"    总费用: {invoice['total_cost']:.4f} 元")
        print(f"    模型分布:")
        for model, data in invoice["by_model"].items():
            print(f"      {model}: {data['tokens']} tokens, {data['cost']:.4f} 元")


async def main():
    """运行所有演示"""
    await demo_multi_tenant()
    await demo_api_gateway()
    await demo_sla()
    await demo_service_orchestration()
    await demo_pipeline()
    await demo_billing()
    
    # 总结
    print("\n" + "=" * 60)
    print("课程总结")
    print("=" * 60)
    
    print("""
  本课介绍了 Agent-as-a-Service (AaaS) 的完整架构：

  核心组件：
  1. 多租户管理: 租户注册、配置隔离、配额控制
  2. API网关: 统一入口、认证鉴权、限流熔断、请求路由
  3. SLA监控: 服务等级目标定义、指标采集、违约检测
  4. 服务编排: 服务发现、负载均衡、多Agent管道协调
  5. 计费系统: 按量计费、套餐折扣、账单生成

  架构模式：
  - 多租户隔离: 共享实例 + 数据隔离（推荐Level 2）
  - API网关模式: 统一入口 + 中间件链 + 后端路由
  - 微服务模式: 服务注册/发现 + 健康检查 + 负载均衡
  - 管道编排: 多步骤串行/并行 + 上下文传递

  生产部署要点：
  - 租户数据严格隔离（会话、记忆、配置分开存储）
  - API Key + OAuth2 双重认证
  - 滑动窗口限流 + Token配额双保险
  - P99延迟/可用性/错误率 三个SLA核心指标
  - 按模型分级计费，企业套餐给折扣
  - 健康检查失败自动摘除实例

  扩展方向：
  - 支持WebSocket长连接的Agent服务
  - Agent市场（Marketplace）- 第三方Agent上架
  - 多区域部署 + 就近路由
  - Agent版本管理 + 灰度发布
  - 自定义Agent模板 - 租户自助配置
""")


if __name__ == "__main__":
    asyncio.run(main())
