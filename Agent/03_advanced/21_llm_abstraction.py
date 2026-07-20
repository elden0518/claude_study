"""
==============================================================================
第二十一课（补充）：LLM 抽象层与 Token 管理
==============================================================================

【为什么需要单独一课？】
现有课程直接使用了特定 LLM API，没有讲解如何设计抽象层。
生产级 Agent 需要支持多模型切换、Token 预算管理、成本控制。

【学习目标】
- 掌握 LLM 抽象层设计（策略模式）
- 学会多模型路由（按能力/复杂度选择）
- 掌握 Token 预算管理
- 学会成本追踪和优化
- 理解语义缓存技术

【核心概念】
- LLM Provider Abstraction（提供商抽象）
- Model Router（模型路由）
- Token Budget（Token 预算）
- Cost Tracking（成本追踪）
- Semantic Cache（语义缓存）

==============================================================================
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================================
# 第一部分：LLM 抽象层设计
# ============================================================================

@dataclass
class LLMResponse:
    """LLM 响应模型"""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.usage.get("input_tokens", 0) + self.usage.get("output_tokens", 0)


class BaseLLMProvider(ABC):
    """
    LLM 提供商抽象基类

    【设计原则】
    策略模式：不同的 LLM 提供商实现相同接口
    - 可以轻松切换模型（Claude/GPT/本地模型）
    - 便于测试（Mock 实现）
    - 统一错误处理
    """

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """生成补全"""
        pass

    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """对话补全"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称"""
        pass

    @property
    @abstractmethod
    def pricing(self) -> Dict[str, float]:
        """定价信息（每百万 Token 的美元价格）"""
        pass


class ClaudeProvider(BaseLLMProvider):
    """Claude 提供商实现"""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self._model = model
        self._pricing = {
            "input_per_million": 3.0,
            "output_per_million": 15.0,
        }

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """模拟 Claude API 调用"""
        # 实际应用中这里调用 Anthropic API
        time.sleep(0.1)  # 模拟延迟
        return LLMResponse(
            content=f"[Claude 回复] 基于输入: {prompt[:30]}...",
            model=self._model,
            usage={"input_tokens": len(prompt) // 4, "output_tokens": 100},
            latency_ms=100,
            cost_usd=self._calculate_cost(len(prompt) // 4, 100)
        )

    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """模拟 Claude 对话"""
        total_input = sum(len(m.get("content", "")) for m in messages) // 4
        time.sleep(0.1)
        return LLMResponse(
            content="[Claude 回复] 对话响应",
            model=self._model,
            usage={"input_tokens": total_input, "output_tokens": 150},
            latency_ms=100,
            cost_usd=self._calculate_cost(total_input, 150)
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def pricing(self) -> Dict[str, float]:
        return self._pricing

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self._pricing["input_per_million"] +
                output_tokens * self._pricing["output_per_million"]) / 1_000_000


class OpenAIProvider(BaseLLMProvider):
    """OpenAI 提供商实现"""

    def __init__(self, model: str = "gpt-4"):
        self._model = model
        self._pricing = {
            "input_per_million": 10.0,
            "output_per_million": 30.0,
        }

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        time.sleep(0.1)
        return LLMResponse(
            content=f"[GPT-4 回复] 基于输入: {prompt[:30]}...",
            model=self._model,
            usage={"input_tokens": len(prompt) // 4, "output_tokens": 120},
            latency_ms=120,
            cost_usd=self._calculate_cost(len(prompt) // 4, 120)
        )

    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        total_input = sum(len(m.get("content", "")) for m in messages) // 4
        time.sleep(0.1)
        return LLMResponse(
            content="[GPT-4 回复] 对话响应",
            model=self._model,
            usage={"input_tokens": total_input, "output_tokens": 180},
            latency_ms=120,
            cost_usd=self._calculate_cost(total_input, 180)
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def pricing(self) -> Dict[str, float]:
        return self._pricing

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self._pricing["input_per_million"] +
                output_tokens * self._pricing["output_per_million"]) / 1_000_000


class LocalModelProvider(BaseLLMProvider):
    """本地模型提供商"""

    def __init__(self, model: str = "llama-2-7b"):
        self._model = model

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        time.sleep(0.05)
        return LLMResponse(
            content=f"[本地模型回复] {prompt[:30]}...",
            model=self._model,
            usage={"input_tokens": len(prompt) // 4, "output_tokens": 80},
            latency_ms=50,
            cost_usd=0.0  # 本地模型无 API 费用
        )

    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        time.sleep(0.05)
        return LLMResponse(
            content="[本地模型回复] 对话响应",
            model=self._model,
            usage={"input_tokens": 200, "output_tokens": 100},
            latency_ms=50,
            cost_usd=0.0
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def pricing(self) -> Dict[str, float]:
        return {"input_per_million": 0.0, "output_per_million": 0.0}


# ============================================================================
# 第二部分：模型路由器
# ============================================================================

class ModelCapability(Enum):
    """模型能力"""
    FAST = "fast"               # 快速响应
    CHEAP = "cheap"             # 低成本
    HIGH_QUALITY = "high_quality"  # 高质量
    CODING = "coding"           # 编程能力
    REASONING = "reasoning"     # 推理能力
    MULTILINGUAL = "multilingual"  # 多语言


class ModelRouter:
    """
    模型路由器

    【原理】
    根据任务需求自动选择最合适的模型：
    - 简单任务 -> 快速/便宜模型
    - 复杂任务 -> 高质量模型
    - 编程任务 -> 编程专用模型
    """

    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.capability_map: Dict[ModelCapability, str] = {}

    def register_provider(self, provider: BaseLLMProvider,
                          capabilities: List[ModelCapability]):
        """注册提供商"""
        self.providers[provider.model_name] = provider
        for cap in capabilities:
            self.capability_map[cap] = provider.model_name
        print(f"  [Router] 注册模型: {provider.model_name}")

    def route(self, task: str, required_capabilities: List[ModelCapability] = None) -> BaseLLMProvider:
        """
        根据任务路由到合适的模型

        【策略】
        1. 如果有指定能力要求，选择匹配的模型
        2. 否则根据任务复杂度自动判断
        """
        if required_capabilities:
            for cap in required_capabilities:
                if cap in self.capability_map:
                    model_name = self.capability_map[cap]
                    return self.providers[model_name]

        # 自动判断复杂度
        complexity = self._estimate_complexity(task)
        if complexity == "high":
            return self.providers.get(
                self.capability_map.get(ModelCapability.HIGH_QUALITY, ""),
                list(self.providers.values())[0]
            )
        else:
            return self.providers.get(
                self.capability_map.get(ModelCapability.FAST, ""),
                list(self.providers.values())[0]
            )

    def _estimate_complexity(self, task: str) -> str:
        """估算任务复杂度"""
        complex_keywords = ["分析", "推理", "证明", "设计", "架构", "复杂"]
        if any(kw in task for kw in complex_keywords):
            return "high"
        return "low"


# ============================================================================
# 第三部分：Token 预算管理
# ============================================================================

class TokenBudget:
    """
    Token 预算管理器

    【功能】
    - 设置日/月 Token 预算
    - 跟踪已使用量
    - 超预算时拒绝请求或降级
    """

    def __init__(self, daily_limit: int = 1_000_000, monthly_limit: int = 30_000_000):
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self._daily_used = 0
        self._monthly_used = 0
        self._last_reset = datetime.now().date()
        self._monthly_reset = datetime.now().month

    def check_budget(self, estimated_tokens: int) -> bool:
        """检查是否有足够预算"""
        self._reset_if_needed()
        return (self._daily_used + estimated_tokens <= self.daily_limit and
                self._monthly_used + estimated_tokens <= self.monthly_limit)

    def consume(self, tokens: int):
        """消耗 Token"""
        self._reset_if_needed()
        self._daily_used += tokens
        self._monthly_used += tokens

    def get_usage(self) -> Dict[str, Any]:
        """获取使用情况"""
        self._reset_if_needed()
        return {
            "daily": {
                "used": self._daily_used,
                "limit": self.daily_limit,
                "remaining": self.daily_limit - self._daily_used,
                "percentage": self._daily_used / self.daily_limit * 100
            },
            "monthly": {
                "used": self._monthly_used,
                "limit": self.monthly_limit,
                "remaining": self.monthly_limit - self._monthly_used,
                "percentage": self._monthly_used / self.monthly_limit * 100
            }
        }

    def _reset_if_needed(self):
        """重置计数器"""
        today = datetime.now().date()
        if today > self._last_reset:
            self._daily_used = 0
            self._last_reset = today

        current_month = datetime.now().month
        if current_month > self._monthly_reset:
            self._monthly_used = 0
            self._monthly_reset = current_month


# ============================================================================
# 第四部分：语义缓存
# ============================================================================

class SemanticCache:
    """
    语义缓存

    【原理】
    缓存相似问题的回答，避免重复调用 LLM
    - 使用文本哈希做快速匹配
    - 使用向量相似度做语义匹配
    """

    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._hit_count = 0
        self._miss_count = 0

    def get(self, query: str) -> Optional[str]:
        """查询缓存"""
        # 精确匹配
        cache_key = self._normalize(query)
        if cache_key in self.cache:
            self._hit_count += 1
            return self.cache[cache_key]["response"]

        # 语义匹配（简化版）
        for key, value in self.cache.items():
            if self._text_similarity(query, key) >= self.similarity_threshold:
                self._hit_count += 1
                return value["response"]

        self._miss_count += 1
        return None

    def put(self, query: str, response: str):
        """写入缓存"""
        cache_key = self._normalize(query)
        self.cache[cache_key] = {
            "response": response,
            "timestamp": datetime.now(),
            "query": query
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total * 100 if total > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": f"{hit_rate:.1f}%"
        }

    @staticmethod
    def _normalize(text: str) -> str:
        """标准化文本"""
        return text.lower().strip()

    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """简化的文本相似度计算"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)


# ============================================================================
# 第五部分：成本追踪器
# ============================================================================

class CostTracker:
    """
    成本追踪器

    【功能】
    - 记录每次 LLM 调用的成本
    - 按模型/时间统计成本
    - 成本告警
    """

    def __init__(self, budget_limit_usd: float = 100.0):
        self.budget_limit_usd = budget_limit_usd
        self.records: List[Dict[str, Any]] = []
        self._total_cost = 0.0

    def record(self, model: str, usage: Dict[str, int], cost_usd: float):
        """记录一次调用"""
        self.records.append({
            "model": model,
            "usage": usage,
            "cost_usd": cost_usd,
            "timestamp": datetime.now()
        })
        self._total_cost += cost_usd

        if self._total_cost > self.budget_limit_usd * 0.8:
            print(f"  [!] 成本告警: 已使用 ${self._total_cost:.4f} / ${self.budget_limit_usd}")

    def get_total_cost(self) -> float:
        return self._total_cost

    def get_cost_by_model(self) -> Dict[str, float]:
        """按模型统计成本"""
        costs: Dict[str, float] = {}
        for record in self.records:
            model = record["model"]
            costs[model] = costs.get(model, 0) + record["cost_usd"]
        return costs

    def get_summary(self) -> Dict[str, Any]:
        """获取成本摘要"""
        return {
            "total_cost_usd": round(self._total_cost, 6),
            "total_calls": len(self.records),
            "avg_cost_per_call": round(self._total_cost / len(self.records), 6) if self.records else 0,
            "budget_remaining": round(self.budget_limit_usd - self._total_cost, 6),
            "cost_by_model": self.get_cost_by_model()
        }


# ============================================================================
# 第六部分：智能 LLM 客户端
# ============================================================================

class SmartLLMClient:
    """
    智能 LLM 客户端

    【整合功能】
    - 模型路由
    - Token 预算
    - 语义缓存
    - 成本追踪
    """

    def __init__(self, router: ModelRouter, budget: TokenBudget,
                 cache: SemanticCache, cost_tracker: CostTracker):
        self.router = router
        self.budget = budget
        self.cache = cache
        self.cost_tracker = cost_tracker

    def chat(self, messages: List[Dict], capabilities: List[ModelCapability] = None) -> LLMResponse:
        """智能对话"""
        # 1. 检查缓存
        query = messages[-1].get("content", "") if messages else ""
        cached = self.cache.get(query)
        if cached:
            print(f"  [Cache Hit] 使用缓存回答")
            return LLMResponse(
                content=cached,
                model="cache",
                usage={"input_tokens": 0, "output_tokens": 0},
                cost_usd=0.0
            )

        # 2. 检查预算
        estimated_tokens = sum(len(m.get("content", "")) for m in messages) // 4 + 200
        if not self.budget.check_budget(estimated_tokens):
            print(f"  [Budget] Token 预算不足，降级到本地模型")
            provider = self.router.providers.get("llama-2-7b",
                                                   list(self.router.providers.values())[0])
        else:
            # 3. 路由到合适模型
            provider = self.router.route(query, capabilities)

        # 4. 调用 LLM
        print(f"  [LLM Call] 使用模型: {provider.model_name}")
        response = provider.chat(messages)

        # 5. 记录
        self.budget.consume(response.total_tokens)
        self.cost_tracker.record(provider.model_name, response.usage, response.cost_usd)
        self.cache.put(query, response.content)

        return response


# ============================================================================
# 第七部分：完整示例
# ============================================================================

def demo_llm_abstraction():
    """演示 LLM 抽象层"""

    print("=" * 60)
    print("LLM 抽象层完整演示")
    print("=" * 60)

    # 1. 创建提供商
    print("\n[Step 1] 注册 LLM 提供商...")
    claude = ClaudeProvider()
    gpt4 = OpenAIProvider()
    local = LocalModelProvider()

    # 2. 配置路由器
    print("\n[Step 2] 配置模型路由器...")
    router = ModelRouter()
    router.register_provider(local, [ModelCapability.FAST, ModelCapability.CHEAP])
    router.register_provider(claude, [ModelCapability.HIGH_QUALITY, ModelCapability.REASONING])
    router.register_provider(gpt4, [ModelCapability.CODING, ModelCapability.MULTILINGUAL])

    # 3. 创建辅助组件
    print("\n[Step 3] 创建辅助组件...")
    budget = TokenBudget(daily_limit=1_000_000, monthly_limit=30_000_000)
    cache = SemanticCache(similarity_threshold=0.85)
    cost_tracker = CostTracker(budget_limit_usd=50.0)

    # 4. 创建智能客户端
    print("\n[Step 4] 创建智能 LLM 客户端...")
    client = SmartLLMClient(router, budget, cache, cost_tracker)

    # 5. 测试不同场景
    print("\n[Step 5] 测试不同场景...")

    # 简单问题 -> 快速模型
    print("\n  --- 场景1: 简单问题 ---")
    messages = [{"role": "user", "content": "你好"}]
    response = client.chat(messages, [ModelCapability.FAST])
    print(f"  回复: {response.content}")
    print(f"  模型: {response.model}, 成本: ${response.cost_usd:.6f}")

    # 复杂问题 -> 高质量模型
    print("\n  --- 场景2: 复杂推理 ---")
    messages = [{"role": "user", "content": "请分析这个复杂的架构设计问题"}]
    response = client.chat(messages, [ModelCapability.HIGH_QUALITY])
    print(f"  回复: {response.content}")
    print(f"  模型: {response.model}, 成本: ${response.cost_usd:.6f}")

    # 编程问题 -> 编程模型
    print("\n  --- 场景3: 编程任务 ---")
    messages = [{"role": "user", "content": "写一个排序算法"}]
    response = client.chat(messages, [ModelCapability.CODING])
    print(f"  回复: {response.content}")
    print(f"  模型: {response.model}, 成本: ${response.cost_usd:.6f}")

    # 重复问题 -> 缓存命中
    print("\n  --- 场景4: 重复问题（缓存）---")
    messages = [{"role": "user", "content": "你好"}]
    response = client.chat(messages)
    print(f"  回复: {response.content}")
    print(f"  模型: {response.model}, 成本: ${response.cost_usd:.6f}")

    # 6. 查看统计
    print("\n[Step 6] 查看统计信息...")
    print(f"\n  Token 预算使用:")
    usage = budget.get_usage()
    print(f"    日预算: {usage['daily']['used']}/{usage['daily']['limit']} ({usage['daily']['percentage']:.1f}%)")
    print(f"    月预算: {usage['monthly']['used']}/{usage['monthly']['limit']} ({usage['monthly']['percentage']:.1f}%)")

    print(f"\n  成本统计:")
    cost_summary = cost_tracker.get_summary()
    print(f"    总成本: ${cost_summary['total_cost_usd']:.6f}")
    print(f"    调用次数: {cost_summary['total_calls']}")
    print(f"    按模型: {cost_summary['cost_by_model']}")

    print(f"\n  缓存统计:")
    cache_stats = cache.get_stats()
    print(f"    缓存大小: {cache_stats['size']}")
    print(f"    命中率: {cache_stats['hit_rate']}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("LLM 抽象层与 Token 管理")
    print("=" * 60)

    demo_llm_abstraction()

    print("\n" + "=" * 60)
    print("课程总结")
    print("=" * 60)
    print("""
  本课介绍了 LLM 抽象层设计：

  核心知识点：
  1. LLM Provider 抽象：策略模式支持多模型
  2. Model Router：按能力/复杂度自动选择模型
  3. Token Budget：日/月预算管理
  4. Semantic Cache：语义缓存减少重复调用
  5. Cost Tracker：成本追踪和告警
  6. SmartLLMClient：整合所有功能的智能客户端

  设计模式：
  - 策略模式：不同 LLM 提供商
  - 路由模式：智能模型选择
  - 缓存模式：语义缓存
  - 观察者模式：成本告警
    """)
