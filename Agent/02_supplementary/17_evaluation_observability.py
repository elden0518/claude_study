"""
==============================================================================
第十七课（补充）：Evaluation & Observability（评估与可观测性）
==============================================================================

【为什么需要单独一课？】
现有课程没有系统讲解如何评估 Agent 的质量和运行时可观测性。
生产级 Agent 需要完善的评估体系和运行时监控。

【学习目标】
- 掌握 Agent 评估的维度和方法
- 学会构建评估数据集和基准测试
- 掌握 A/B 测试和人工评估
- 理解 Tracing（链路追踪）的实现
- 学会构建可观测性体系（日志、指标、追踪）

【核心概念】
- Evaluation Metrics（评估指标）
- Benchmark Suite（基准测试套件）
- A/B Testing（A/B 测试）
- Distributed Tracing（分布式追踪）
- Observability（可观测性）

==============================================================================
"""

import time
import uuid
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# 第一部分：评估指标体系
# ============================================================================
# 【知识点】Agent 评估需要多维度指标
# - 任务完成率：是否成功完成任务
# - 效率：步骤数、Token 消耗、耗时
# - 质量：输出质量评分
# - 安全性：是否触发安全规则
# - 用户满意度：人工评分


class MetricType(Enum):
    """评估指标类型"""
    ACCURACY = "accuracy"           # 准确率
    EFFICIENCY = "efficiency"       # 效率
    SAFETY = "safety"               # 安全性
    QUALITY = "quality"             # 质量
    LATENCY = "latency"             # 延迟
    COST = "cost"                   # 成本


@dataclass
class EvaluationMetric:
    """单个评估指标"""
    name: str
    metric_type: MetricType
    score: float                    # 0.0 ~ 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self):
        bar_len = int(self.score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        return f"  {self.name:<20} [{bar}] {self.score:.2f}"


class EvaluationResult:
    """评估结果集合"""

    def __init__(self, run_id: str, config_name: str):
        self.run_id = run_id
        self.config_name = config_name
        self.metrics: List[EvaluationMetric] = []
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.total_tokens = 0
        self.total_steps = 0

    def add_metric(self, metric: EvaluationMetric):
        self.metrics.append(metric)

    def finish(self):
        self.end_time = datetime.now()

    @property
    def duration(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0

    def get_summary(self) -> Dict[str, float]:
        """获取评估摘要"""
        summary = {}
        for m in self.metrics:
            if m.name in summary:
                summary[m.name] = (summary[m.name] + m.score) / 2
            else:
                summary[m.name] = m.score
        return summary

    def display(self):
        """展示评估结果"""
        print(f"\n{'='*60}")
        print(f"  评估结果: {self.config_name}")
        print(f"  Run ID: {self.run_id}")
        print(f"  耗时: {self.duration:.2f}s | 步骤: {self.total_steps} | Tokens: {self.total_tokens}")
        print(f"{'='*60}")
        for m in self.metrics:
            print(m)
        print(f"{'='*60}")


# ============================================================================
# 第二部分：评估器（Evaluators）
# ============================================================================
# 【知识点】不同类型的评估器
# - 规则评估器：基于预定义规则
# - LLM 评估器：使用 LLM 评判输出质量
# - 人工评估器：收集人工反馈
# - 自动评估器：自动化指标计算


class BaseEvaluator(ABC):
    """评估器基类"""

    @abstractmethod
    def evaluate(self, input_text: str, output: str, expected: Optional[str] = None) -> EvaluationMetric:
        pass


class RuleBasedEvaluator(BaseEvaluator):
    """
    基于规则的评估器
    
    【原理】
    检查输出是否符合预定义规则：
    - 长度限制
    - 禁止词汇
    - 格式要求
    - 关键词包含
    """

    def __init__(self, name: str, rules: Dict[str, Any]):
        self.name = name
        self.rules = rules

    def evaluate(self, input_text: str, output: str, expected: Optional[str] = None) -> EvaluationMetric:
        score = 1.0
        details = {}

        # 规则1：长度检查
        if "max_length" in self.rules:
            max_len = self.rules["max_length"]
            if len(output) > max_len:
                score -= 0.2
                details["max_length_violated"] = True

        # 规则2：禁止词汇检查
        if "forbidden_words" in self.rules:
            for word in self.rules["forbidden_words"]:
                if word.lower() in output.lower():
                    score -= 0.3
                    details["forbidden_word_found"] = word
                    break

        # 规则3：必须包含关键词
        if "required_keywords" in self.rules:
            missing = []
            for kw in self.rules["required_keywords"]:
                if kw.lower() not in output.lower():
                    missing.append(kw)
            if missing:
                score -= 0.1 * len(missing)
                details["missing_keywords"] = missing

        # 规则4：格式检查（如 JSON 格式）
        if "format" in self.rules:
            if self.rules["format"] == "json":
                import json
                try:
                    json.loads(output)
                except json.JSONDecodeError:
                    score -= 0.5
                    details["invalid_json"] = True

        return EvaluationMetric(
            name=self.name,
            metric_type=MetricType.QUALITY,
            score=max(0.0, min(1.0, score)),
            details=details
        )


class ExactMatchEvaluator(BaseEvaluator):
    """
    精确匹配评估器
    
    【原理】
    将输出与期望答案进行精确/模糊匹配
    适用于有明确正确答案的场景
    """

    def evaluate(self, input_text: str, output: str, expected: Optional[str] = None) -> EvaluationMetric:
        if expected is None:
            return EvaluationMetric(
                name="exact_match",
                metric_type=MetricType.ACCURACY,
                score=0.0,
                details={"error": "No expected output provided"}
            )

        # 精确匹配
        exact = output.strip() == expected.strip()

        # 模糊匹配（包含）
        contains = expected.strip().lower() in output.strip().lower()

        score = 1.0 if exact else (0.7 if contains else 0.0)

        return EvaluationMetric(
            name="exact_match",
            metric_type=MetricType.ACCURACY,
            score=score,
            details={"exact": exact, "contains": contains}
        )


class LLMEvaluator(BaseEvaluator):
    """
    LLM-as-Judge 评估器
    
    【原理】
    使用另一个 LLM 来评判输出质量
    这是目前评估开放式输出的主流方法
    
    【评估维度】
    - 相关性：回答是否与问题相关
    - 准确性：信息是否准确
    - 完整性：是否覆盖了所有要点
    - 流畅性：语言是否流畅自然
    """

    def __init__(self):
        # 模拟 LLM 评判（实际使用时调用真实 LLM）
        self.criteria = ["相关性", "准确性", "完整性", "流畅性"]

    def evaluate(self, input_text: str, output: str, expected: Optional[str] = None) -> EvaluationMetric:
        # 模拟 LLM 评分（实际中会构建评判 prompt 发给 LLM）
        # 评判 Prompt 示例：
        judging_prompt = f"""
请作为评判专家，对以下回答进行评分（1-5分）：

问题：{input_text}
回答：{output}
{f'参考答案：{expected}' if expected else ''}

请从以下维度评分：
1. 相关性：回答是否切题
2. 准确性：信息是否正确
3. 完整性：是否完整回答
4. 流畅性：表达是否清晰

输出格式：{{"relevance": 4, "accuracy": 5, "completeness": 3, "fluency": 4}}
"""
        # 模拟评分结果
        import hashlib
        hash_val = int(hashlib.md5((input_text + output).encode()).hexdigest()[:8], 16)
        scores = {
            "relevance": (hash_val % 3 + 3) / 5,
            "accuracy": ((hash_val >> 4) % 3 + 3) / 5,
            "completeness": ((hash_val >> 8) % 3 + 3) / 5,
            "fluency": ((hash_val >> 12) % 3 + 3) / 5,
        }
        avg_score = statistics.mean(scores.values())

        return EvaluationMetric(
            name="llm_judge",
            metric_type=MetricType.QUALITY,
            score=avg_score,
            details={"scores": scores, "prompt_template": "llm_as_judge_v1"}
        )


# ============================================================================
# 第三部分：基准测试套件（Benchmark Suite）
# ============================================================================
# 【知识点】
# - 标准化测试用例集
# - 自动化运行和评分
# - 结果对比和回归检测


@dataclass
class TestCase:
    """测试用例"""
    id: str
    input_text: str
    expected_output: Optional[str] = None
    category: str = "general"
    difficulty: str = "easy"        # easy / medium / hard
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkSuite:
    """
    基准测试套件
    
    【功能】
    - 管理测试用例集
    - 批量运行评估
    - 生成对比报告
    - 回归检测
    """

    def __init__(self, name: str):
        self.name = name
        self.test_cases: List[TestCase] = []
        self.evaluators: List[BaseEvaluator] = []

    def add_test_case(self, test_case: TestCase):
        self.test_cases.append(test_case)

    def add_evaluator(self, evaluator: BaseEvaluator):
        self.evaluators.append(evaluator)

    def run(self, agent_fn: Callable[[str], str], config_name: str = "default") -> EvaluationResult:
        """
        运行基准测试
        
        Args:
            agent_fn: Agent 函数，接受输入返回输出
            config_name: 配置名称（用于对比）
        """
        result = EvaluationResult(
            run_id=uuid.uuid4().hex[:8],
            config_name=config_name
        )

        print(f"\n🔄 运行基准测试: {self.name} ({len(self.test_cases)} 个用例)")
        print("-" * 50)

        all_scores = []
        for i, tc in enumerate(self.test_cases):
            # 运行 Agent
            start = time.time()
            output = agent_fn(tc.input_text)
            elapsed = time.time() - start

            print(f"  [{i+1}/{len(self.test_cases)}] {tc.category}/{tc.difficulty}: "
                  f"耗时 {elapsed:.2f}s, 输出 {len(output)} 字符")

            # 评估
            for evaluator in self.evaluators:
                metric = evaluator.evaluate(tc.input_text, output, tc.expected_output)
                result.add_metric(metric)
                all_scores.append(metric.score)

            result.total_steps += 1

        result.finish()

        # 汇总
        if all_scores:
            overall = EvaluationMetric(
                name="overall_score",
                metric_type=MetricType.QUALITY,
                score=statistics.mean(all_scores),
                details={
                    "min": min(all_scores),
                    "max": max(all_scores),
                    "stddev": statistics.stdev(all_scores) if len(all_scores) > 1 else 0
                }
            )
            result.add_metric(overall)

        return result


# ============================================================================
# 第四部分：链路追踪（Tracing）
# ============================================================================
# 【知识点】
# - Agent 运行需要完整的链路追踪
# - 记录每一步的输入/输出/耗时
# - 支持调试和性能分析
# - 类似 OpenTelemetry 的 Span 模型


@dataclass
class Span:
    """
    追踪 Span（类似 OpenTelemetry）
    
    【概念】
    Span 是追踪的基本单元，代表一次操作：
    - 有开始和结束时间
    - 可以有子 Span（嵌套）
    - 携带属性和事件
    """
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: Optional[str] = None
    operation: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"            # ok / error

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

    def add_event(self, name: str, data: Dict[str, Any] = None):
        self.events.append({
            "timestamp": time.time(),
            "name": name,
            "data": data or {}
        })

    def finish(self, status: str = "ok"):
        self.end_time = time.time()
        self.status = status

    def display(self, indent: int = 0):
        prefix = "  " * indent
        status_icon = "✅" if self.status == "ok" else "❌"
        print(f"{prefix}{status_icon} [{self.operation}] {self.duration_ms:.1f}ms")
        for event in self.events:
            print(f"{prefix}  📌 {event['name']}: {event['data']}")


class TraceCollector:
    """
    追踪收集器
    
    【功能】
    - 收集所有 Span
    - 构建调用树
    - 导出追踪数据
    """

    def __init__(self):
        self.spans: List[Span] = []
        self._current_trace_id = uuid.uuid4().hex[:8]

    def start_span(self, operation: str, parent_id: Optional[str] = None) -> Span:
        span = Span(
            parent_id=parent_id,
            operation=operation
        )
        self.spans.append(span)
        return span

    def display_trace(self):
        """显示追踪链路"""
        print(f"\n🔍 Trace ID: {self._trace_id}")
        print("=" * 50)

        # 构建树结构
        root_spans = [s for s in self.spans if s.parent_id is None]
        for root in root_spans:
            root.display(indent=0)
            children = [s for s in self.spans if s.parent_id == root.span_id]
            for child in children:
                child.display(indent=1)
                grandchildren = [s for s in self.spans if s.parent_id == child.span_id]
                for gc in grandchildren:
                    gc.display(indent=2)

        # 汇总
        total_time = sum(s.duration_ms for s in self.spans if s.parent_id is None)
        print(f"\n  总耗时: {total_time:.1f}ms | Span 数: {len(self.spans)}")

    @property
    def _trace_id(self):
        return self._current_trace_id


# ============================================================================
# 第五部分：可观测性仪表盘
# ============================================================================
# 【知识点】
# 可观测性三大支柱：
# 1. Logs（日志）：离散事件记录
# 2. Metrics（指标）：可聚合的数值数据
# 3. Traces（追踪）：请求的完整链路


class ObservabilityDashboard:
    """
    可观测性仪表盘
    
    【功能】
    - 聚合日志、指标、追踪数据
    - 提供统一视图
    - 告警规则
    """

    def __init__(self):
        self.metrics_history: Dict[str, List[float]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.alert_rules: List[Dict[str, Any]] = []

    def record_metric(self, name: str, value: float):
        """记录指标"""
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        self.metrics_history[name].append(value)

        # 检查告警规则
        for rule in self.alert_rules:
            if rule["metric"] == name:
                if rule["condition"] == "above" and value > rule["threshold"]:
                    self.alerts.append({
                        "time": datetime.now(),
                        "rule": rule["name"],
                        "value": value,
                        "threshold": rule["threshold"]
                    })

    def add_alert_rule(self, name: str, metric: str, condition: str, threshold: float):
        self.alert_rules.append({
            "name": name,
            "metric": metric,
            "condition": condition,
            "threshold": threshold
        })

    def display(self):
        """显示仪表盘"""
        print("\n" + "=" * 60)
        print("  📊 Agent 可观测性仪表盘")
        print("=" * 60)

        # 指标汇总
        print("\n  📈 关键指标:")
        for name, values in self.metrics_history.items():
            if values:
                avg = statistics.mean(values)
                current = values[-1]
                trend = "↑" if len(values) > 1 and values[-1] > values[-2] else "↓"
                print(f"    {name:<25} 当前: {current:.2f} | 平均: {avg:.2f} | 趋势: {trend}")

        # 告警
        if self.alerts:
            print(f"\n  ⚠️  告警 ({len(self.alerts)} 条):")
            for alert in self.alerts[-5:]:  # 最近5条
                print(f"    [{alert['time'].strftime('%H:%M:%S')}] "
                      f"{alert['rule']}: {alert['value']:.2f} > {alert['threshold']:.2f}")
        else:
            print("\n  ✅ 无告警")

        print("=" * 60)


# ============================================================================
# 第六部分：A/B 测试框架
# ============================================================================

class ABTestRunner:
    """
    A/B 测试运行器
    
    【原理】
    - 将用户请求随机分配到不同配置
    - 收集各配置的评估指标
    - 统计显著性检验
    """

    def __init__(self):
        self.configs: Dict[str, Callable] = {}
        self.results: Dict[str, List[float]] = {}

    def add_config(self, name: str, agent_fn: Callable):
        self.configs[name] = agent_fn
        self.results[name] = []

    def run_comparison(self, test_cases: List[TestCase], evaluator: BaseEvaluator):
        """运行 A/B 对比"""
        print(f"\n🔬 A/B 测试: {list(self.configs.keys())}")
        print("=" * 55)

        for tc in test_cases:
            for name, fn in self.configs.items():
                output = fn(tc.input_text)
                metric = evaluator.evaluate(tc.input_text, output, tc.expected_output)
                self.results[name].append(metric.score)

        # 结果对比
        print(f"\n  {'配置':<20} {'平均分':<10} {'最低':<10} {'最高':<10} {'用例数':<8}")
        print("  " + "-" * 55)
        for name, scores in self.results.items():
            if scores:
                avg = statistics.mean(scores)
                print(f"  {name:<20} {avg:<10.3f} {min(scores):<10.3f} "
                      f"{max(scores):<10.3f} {len(scores):<8}")

        # 简单对比
        if len(self.configs) == 2:
            names = list(self.results.keys())
            avg_a = statistics.mean(self.results[names[0]])
            avg_b = statistics.mean(self.results[names[1]])
            diff = avg_b - avg_a
            winner = names[1] if diff > 0 else names[0]
            print(f"\n  🏆 优胜者: {winner} (领先 {abs(diff):.3f})")


# ============================================================================
# 演示运行
# ============================================================================

def demo_evaluation():
    """演示评估流程"""
    print("\n" + "🤖" * 30)
    print("第十七课：Evaluation & Observability（评估与可观测性）")
    print("🤖" * 30)

    # ---- 演示1：规则评估 ----
    print("\n" + "=" * 60)
    print("演示1：规则评估器")
    print("=" * 60)

    evaluator = RuleBasedEvaluator(
        name="output_quality",
        rules={
            "max_length": 500,
            "forbidden_words": ["不知道", "无法回答"],
            "required_keywords": ["分析", "建议"]
        }
    )

    test_outputs = [
        ("什么是Agent？", "Agent是一个智能分析系统，可以自主完成任务。我有以下建议..."),
        ("帮我写代码", "不知道，无法回答这个问题。"),
        ("分析数据", "数据分析结果显示趋势向上，建议关注增长率。"),
    ]

    for input_text, output in test_outputs:
        metric = evaluator.evaluate(input_text, output)
        print(f"\n  输入: {input_text}")
        print(f"  输出: {output[:50]}...")
        print(f"  评分: {metric.score:.2f} | 详情: {metric.details}")

    # ---- 演示2：基准测试 ----
    print("\n" + "=" * 60)
    print("演示2：基准测试套件")
    print("=" * 60)

    suite = BenchmarkSuite("Agent 基础能力测试")
    suite.add_evaluator(ExactMatchEvaluator())
    suite.add_evaluator(LLMEvaluator())

    # 添加测试用例
    suite.add_test_case(TestCase("tc1", "1+1等于几？", "2", "math", "easy"))
    suite.add_test_case(TestCase("tc2", "中国的首都是？", "北京", "knowledge", "easy"))
    suite.add_test_case(TestCase("tc3", "Python是什么语言？", "编程语言", "knowledge", "medium"))

    # 模拟 Agent（返回不同质量的回答）
    def mock_agent_v1(input_text: str) -> str:
        responses = {
            "1+1等于几？": "2",
            "中国的首都是？": "北京",
            "Python是什么语言？": "Python是一种高级编程语言"
        }
        return responses.get(input_text, "我不确定")

    result = suite.run(mock_agent_v1, "mock_agent_v1")
    result.display()

    return suite, result


def demo_tracing():
    """演示链路追踪"""
    print("\n" + "=" * 60)
    print("演示3：链路追踪")
    print("=" * 60)

    collector = TraceCollector()

    # 模拟 Agent 运行追踪
    root_span = collector.start_span("agent.run")
    root_span.attributes["input"] = "查天气并计算出行建议"

    # LLM 调用
    llm_span = collector.start_span("llm.call", parent_id=root_span.span_id)
    llm_span.add_event("prompt_sent", {"tokens": 150})
    time.sleep(0.05)  # 模拟延迟
    llm_span.add_event("response_received", {"tokens": 80})
    llm_span.finish()

    # 工具调用
    tool_span = collector.start_span("tool.execute", parent_id=root_span.span_id)
    tool_span.add_event("tool_selected", {"name": "get_weather"})
    time.sleep(0.03)
    tool_span.add_event("tool_result", {"temp": "28°C"})
    tool_span.finish()

    # 第二次 LLM 调用
    llm_span2 = collector.start_span("llm.call", parent_id=root_span.span_id)
    llm_span2.add_event("prompt_sent", {"tokens": 200})
    time.sleep(0.04)
    llm_span2.add_event("response_received", {"tokens": 120})
    llm_span2.finish()

    root_span.finish()

    collector.display_trace()


def demo_observability():
    """演示可观测性"""
    print("\n" + "=" * 60)
    print("演示4：可观测性仪表盘")
    print("=" * 60)

    dashboard = ObservabilityDashboard()

    # 设置告警规则
    dashboard.add_alert_rule("延迟告警", "avg_latency_ms", "above", 5000)
    dashboard.add_alert_rule("错误率告警", "error_rate", "above", 0.1)

    # 模拟指标数据
    import random
    for i in range(10):
        dashboard.record_metric("avg_latency_ms", 2000 + random.randint(-500, 1500))
        dashboard.record_metric("success_rate", 0.85 + random.uniform(-0.1, 0.15))
        dashboard.record_metric("error_rate", 0.05 + random.uniform(-0.03, 0.08))
        dashboard.record_metric("tokens_per_request", 500 + random.randint(-100, 300))

    dashboard.display()


def demo_ab_testing():
    """演示 A/B 测试"""
    print("\n" + "=" * 60)
    print("演示5：A/B 测试对比")
    print("=" * 60)

    runner = ABTestRunner()

    # 配置 A：简洁回答
    def agent_config_a(input_text: str) -> str:
        return "2"

    # 配置 B：详细回答
    def agent_config_b(input_text: str) -> str:
        responses = {
            "1+1等于几？": "答案是2。这是基础数学运算。",
            "中国的首都是？": "中国的首都是北京，位于华北平原。",
            "Python是什么语言？": "Python是一种广泛使用的高级编程语言。"
        }
        return responses.get(input_text, "我不确定答案。")

    runner.add_config("简洁版", agent_config_a)
    runner.add_config("详细版", agent_config_b)

    test_cases = [
        TestCase("1", "1+1等于几？", "2"),
        TestCase("2", "中国的首都是？", "北京"),
        TestCase("3", "Python是什么语言？", "编程语言"),
    ]

    runner.run_comparison(test_cases, ExactMatchEvaluator())


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    demo_evaluation()
    demo_tracing()
    demo_observability()
    demo_ab_testing()

    print("\n" + "=" * 60)
    print("✅ 第十七课学习完成！")
    print("=" * 60)
    print("""
  本课要点总结：
1. Agent 评估需要多维度指标（准确率、效率、安全性、质量）
2. 规则评估器适合有明确规则的场景
3. LLM-as-Judge 适合开放式输出评估
4. 基准测试套件提供标准化测试和回归检测
5. 链路追踪（Tracing）帮助调试和性能分析
6. 可观测性 = 日志 + 指标 + 追踪
7. A/B 测试帮助对比不同配置的效果

  进阶方向：
- LangSmith / LangFuse 等专用追踪平台
- 构建领域特定的评估数据集
- 自动化回归测试流水线
- 成本效益分析（Cost-Benefit Analysis）
    """)
