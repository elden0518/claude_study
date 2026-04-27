"""
主题：评估与测试 —— RAGAS 评估框架、单元测试、A/B 测试

学习目标：
  1. 理解为什么需要系统化评估（避免主观判断）
  2. 掌握 RAGAS 评估框架（RAG 系统的黄金标准）
  3. 学会编写 LangGraph 单元测试
  4. 掌握 A/B 测试方法（对比不同策略效果）
  5. 建立持续评估和监控体系

核心概念：
  评估维度：
  - Faithfulness（忠实度）：答案是否基于检索内容
  - Answer Relevance（相关性）：答案是否回答问题
  - Context Precision（上下文精确度）：检索文档的质量
  - Context Recall（上下文召回率）：是否找到所有相关文档
  
  测试类型：
  - 单元测试：单个节点/函数的正确性
  - 集成测试：完整工作流的端到端测试
  - A/B 测试：线上对比不同版本

前置知识：已完成 02_lc_advanced/10_rag_chain.py
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# =============================================================================
# Part 1：RAGAS 评估框架
# =============================================================================

def demo_ragas_evaluation():
    """
    RAGAS（Retrieval Augmented Generation Assessment）是专门评估 RAG 系统的框架。
    
    核心指标：
    1. Faithfulness（忠实度）：答案中的事实是否都来自上下文
    2. Answer Relevance（答案相关性）：答案是否与问题相关
    3. Context Precision（上下文精确度）：检索到的文档中有多少是相关的
    4. Context Recall（上下文召回率）：是否检索到了所有相关文档
    5. Answer Semantic Similarity（语义相似度）：与标准答案的相似程度
    
    评分范围：0-1（越高越好）
    """
    print("=" * 60)
    print("Part 1: RAGAS Evaluation Framework")
    print("=" * 60)
    
    print("""
  📊 RAGAS 评估流程：
  ──────────────────────────────────────────────
  
  准备测试集：
  ├─ 问题（questions）
  ├─ 标准答案（ground_truths）
  └─ 检索上下文（contexts）
       ↓
  生成系统答案：
  └─ answers（你的 RAG 系统输出）
       ↓
  RAGAS 评估：
  ├─ Faithfulness
  ├─ Answer Relevance
  ├─ Context Precision
  ├─ Context Recall
  └─ Answer Similarity
       ↓
  综合报告：
  └─ 各指标平均分 + 详细分析
    """)
    
    # 安装说明
    print("\n【安装 RAGAS】")
    print("-" * 60)
    print("pip install ragas datasets")
    print("-" * 60)
    
    # 使用示例
    example_code = '''
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity
)
from datasets import Dataset

# 1. 准备测试数据
test_data = {
    "question": [
        "LangChain 是什么？",
        "如何实现 RAG？",
        "什么是 Agent？"
    ],
    "answer": [
        "LangChain 是构建 LLM 应用的框架...",  # 你的系统生成的答案
        "RAG 通过检索相关文档增强生成...",
        "Agent 是能自主调用工具的智能体..."
    ],
    "contexts": [
        ["LangChain是一个用于构建基于大型语言模型应用的开源框架..."],  # 检索到的文档
        ["RAG（检索增强生成）是LangChain的重要应用场景..."],
        ["Agent是LangChain中能够自主调用工具、规划步骤的智能体组件..."]
    ],
    "ground_truth": [
        "LangChain 是一个用于构建基于大型语言模型（LLM）应用的开源框架。",  # 标准答案
        "RAG 通过检索相关文档来增强 LLM 的回答能力。",
        "Agent 是能够自主调用工具、规划步骤来完成复杂任务的智能体。"
    ]
}

# 2. 转换为 Dataset
dataset = Dataset.from_dict(test_data)

# 3. 执行评估
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity
    ]
)

# 4. 查看结果
print(result)
# 输出示例：
# {
#   'faithfulness': 0.85,
#   'answer_relevancy': 0.92,
#   'context_precision': 0.78,
#   'context_recall': 0.81,
#   'answer_similarity': 0.88
# }

# 5. 详细分析
df = result.to_pandas()
print(df.head())

# 6. 可视化
import matplotlib.pyplot as plt

metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 
           'context_recall', 'answer_similarity']
scores = [result[m] for m in metrics]

plt.bar(metrics, scores)
plt.ylim(0, 1)
plt.title('RAG System Evaluation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ragas_evaluation.png')
    '''
    
    print("\n【完整评估代码示例】")
    print("-" * 60)
    print(example_code)
    print("-" * 60)
    
    # 指标解释
    print("\n【指标详解】")
    print("-" * 60)
    metrics_explanation = """
  1. Faithfulness（忠实度）
  ──────────────────────────────────────────────
  定义：答案中的所有事实是否都能从上下文中推断出来
  目的：检测幻觉（hallucination）
  
  示例：
  上下文："Python 由 Guido van Rossum 于 1991 年创建"
  答案："Python 由 Guido 在 1990 年代初期创建" → ✓ 忠实
  答案："Python 由 Facebook 创建" → ✗ 不忠实（幻觉）
  
  评分标准：
  • 0.9-1.0：优秀，几乎没有幻觉
  • 0.7-0.9：良好，偶有小错误
  • < 0.7：需改进，存在明显幻觉
  
  
  2. Answer Relevance（答案相关性）
  ──────────────────────────────────────────────
  定义：答案是否直接回答了问题
  目的：确保答案不偏离主题
  
  示例：
  问题："Python 的优点是什么？"
  答案："Python 易学易用，库丰富..." → ✓ 相关
  答案："Python 是一种编程语言" → ✗ 不相关（未回答优点）
  
  
  3. Context Precision（上下文精确度）
  ──────────────────────────────────────────────
  定义：检索到的文档中有多少是真正相关的
  目的：评估检索器质量
  
  计算：相关文档数 / 总检索文档数
  
  示例：
  检索 5 个文档，其中 4 个相关 → Precision = 0.8
  
  
  4. Context Recall（上下文召回率）
  ──────────────────────────────────────────────
  定义：是否检索到了所有应该找到的相关文档
  目的：评估检索覆盖率
  
  计算：检索到的相关文档数 / 所有相关文档总数
  
  示例：
  应该有 10 个相关文档，只找到 7 个 → Recall = 0.7
  
  
  5. Answer Similarity（答案相似度）
  ──────────────────────────────────────────────
  定义：生成答案与标准答案的语义相似度
  目的：评估答案质量
  
  方法：使用 Embedding 模型计算余弦相似度
  
  注意：仅作为参考，因为相同意思可以有不同表述
    """
    print(metrics_explanation)
    print("-" * 60)
    
    # 最佳实践
    print("\n💡 RAGAS 最佳实践：")
    print("  • 测试集大小：至少 50-100 个样本")
    print("  • 覆盖多样性：包含简单、中等、困难问题")
    print("  • 定期更新：随着系统迭代更新测试集")
    print("  • 结合人工评估：自动化指标不能完全替代人工")
    print("  • 设定阈值：如 Faithfulness > 0.85 才上线")


# =============================================================================
# Part 2：LangGraph 单元测试
# =============================================================================

def demo_unit_testing():
    """
    为 LangGraph 应用编写单元测试。
    
    测试层次：
    1. 节点级别：测试单个节点的输入输出
    2. 图级别：测试完整工作流
    3. 状态管理：测试 checkpoint 和恢复
    """
    print("\n" + "=" * 60)
    print("Part 2: Unit Testing for LangGraph")
    print("=" * 60)
    
    test_code = '''
# file: test_agent.py
"""
LangGraph 单元测试示例
运行：pytest test_agent.py -v
"""

import pytest
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from unittest.mock import Mock, patch


# ── 被测代码 ──────────────────────────────────────────────

class CalculatorState(TypedDict):
    a: int
    b: int
    result: int
    operation: str


def add_node(state: CalculatorState) -> dict:
    """加法节点"""
    return {"result": state["a"] + state["b"]}


def multiply_node(state: CalculatorState) -> dict:
    """乘法节点"""
    return {"result": state["a"] * state["b"]}


def decide_operation(state: CalculatorState) -> str:
    """条件路由"""
    return state["operation"]


def build_calculator_graph():
    """构建计算器图"""
    graph = StateGraph(CalculatorState)
    
    graph.add_node("add", add_node)
    graph.add_node("multiply", multiply_node)
    
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", decide_operation, {
        "add": "add",
        "multiply": "multiply"
    })
    graph.add_edge("add", END)
    graph.add_edge("multiply", END)
    
    return graph.compile()


# ── 测试用例 ──────────────────────────────────────────────

class TestCalculatorNodes:
    """节点级别测试"""
    
    def test_add_node(self):
        """测试加法节点"""
        state = {"a": 3, "b": 5, "result": 0, "operation": "add"}
        result = add_node(state)
        
        assert result["result"] == 8
        assert isinstance(result["result"], int)
    
    def test_multiply_node(self):
        """测试乘法节点"""
        state = {"a": 4, "b": 7, "result": 0, "operation": "multiply"}
        result = multiply_node(state)
        
        assert result["result"] == 28
    
    def test_add_node_with_negative(self):
        """测试负数加法"""
        state = {"a": -5, "b": 3, "result": 0, "operation": "add"}
        result = add_node(state)
        
        assert result["result"] == -2


class TestCalculatorGraph:
    """图级别测试"""
    
    @pytest.fixture
    def calculator_app(self):
        """测试夹具：创建计算器应用"""
        return build_calculator_graph()
    
    def test_addition_workflow(self, calculator_app):
        """测试加法工作流"""
        initial_state = {
            "a": 10,
            "b": 20,
            "result": 0,
            "operation": "add"
        }
        
        result = calculator_app.invoke(initial_state)
        
        assert result["result"] == 30
        assert result["operation"] == "add"
    
    def test_multiplication_workflow(self, calculator_app):
        """测试乘法工作流"""
        initial_state = {
            "a": 6,
            "b": 7,
            "result": 0,
            "operation": "multiply"
        }
        
        result = calculator_app.invoke(initial_state)
        
        assert result["result"] == 42
    
    def test_invalid_operation(self, calculator_app):
        """测试无效操作"""
        initial_state = {
            "a": 1,
            "b": 2,
            "result": 0,
            "operation": "divide"  # 不支持的操作
        }
        
        # 应该抛出异常或返回默认行为
        with pytest.raises(Exception):
            calculator_app.invoke(initial_state)


class TestCheckpointing:
    """状态管理测试"""
    
    def test_state_persistence(self):
        """测试状态持久化"""
        memory = MemorySaver()
        graph = build_calculator_graph()
        app = graph.compile(checkpointer=memory)
        
        config = {"configurable": {"thread_id": "test-1"}}
        
        # 第一次执行
        result1 = app.invoke({
            "a": 5,
            "b": 10,
            "result": 0,
            "operation": "add"
        }, config)
        
        # 获取状态快照
        snapshot = app.get_state(config)
        
        assert snapshot.values["result"] == 15
        assert snapshot.config["configurable"]["thread_id"] == "test-1"
    
    def test_state_history(self):
        """测试状态历史"""
        memory = MemorySaver()
        graph = build_calculator_graph()
        app = graph.compile(checkpointer=memory)
        
        config = {"configurable": {"thread_id": "test-2"}}
        
        # 多次执行
        app.invoke({"a": 1, "b": 2, "result": 0, "operation": "add"}, config)
        app.invoke({"a": 3, "b": 4, "result": 0, "operation": "add"}, config)
        
        # 检查历史记录
        history = list(app.get_state_history(config))
        
        assert len(history) >= 2


class TestWithMockedLLM:
    """使用 Mock LLM 的测试（避免真实 API 调用）"""
    
    def test_llm_node_with_mock(self):
        """测试包含 LLM 调用的节点"""
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Mocked response")
        
        # 测试使用 mock LLM 的节点
        def llm_node(state):
            response = mock_llm.invoke("Test question")
            return {"response": response.content}
        
        result = llm_node({})
        
        assert result["response"] == "Mocked response"
        mock_llm.invoke.assert_called_once_with("Test question")


# ── 性能测试 ──────────────────────────────────────────────

class TestPerformance:
    """性能测试"""
    
    def test_response_time(self):
        """测试响应时间"""
        import time
        
        app = build_calculator_graph()
        
        start = time.time()
        app.invoke({"a": 100, "b": 200, "result": 0, "operation": "add"})
        elapsed = time.time() - start
        
        # 应该在 1 秒内完成
        assert elapsed < 1.0
    
    def test_concurrent_execution(self):
        """测试并发执行"""
        import asyncio
        
        app = build_calculator_graph()
        
        async def run_test():
            tasks = [
                app.ainvoke({"a": i, "b": i+1, "result": 0, "operation": "add"})
                for i in range(10)
            ]
            results = await asyncio.gather(*tasks)
            return results
        
        results = asyncio.run(run_test())
        
        assert len(results) == 10
        assert all(r["result"] == r["a"] + r["b"] for r in results)
    '''
    
    print(test_code)
    
    print("\n💡 测试最佳实践：")
    print("  • 每个节点独立测试（隔离依赖）")
    print("  • 使用 Mock 避免真实 API 调用")
    print("  • 覆盖边界情况（空输入、异常值）")
    print("  • 测试状态持久化和恢复")
    print("  • 设置性能基线（防止回归）")


# =============================================================================
# Part 3：A/B 测试
# =============================================================================

def demo_ab_testing():
    """
    A/B 测试：在线上环境对比不同版本的性能。
    
    测试场景：
    1. 不同 Prompt 模板的效果
    2. 不同检索策略的准确率
    3. 不同模型的性价比
    4. 不同 Agent 模式的用户满意度
    """
    print("\n" + "=" * 60)
    print("Part 3: A/B Testing")
    print("=" * 60)
    
    ab_guide = """
  🧪 A/B 测试实施指南：
  ──────────────────────────────────────────────
  
  步骤 1：定义假设和目标
  ──────────────────────────────────────────────
  
  示例：
  假设："使用查询重写的 RAG 比基础 RAG 准确率高 15%"
  
  目标指标：
  • 主要：Answer Relevance（答案相关性）
  • 次要：用户满意度评分、点击率
  • 护栏：延迟增加不超过 20%
  
  
  步骤 2：设计实验组
  ──────────────────────────────────────────────
  
  对照组（Control, A 组）：
  • 基础 RAG（无查询重写）
  • 流量分配：50%
  
  实验组（Treatment, B 组）：
  • 高级 RAG（带查询重写）
  • 流量分配：50%
  
  
  步骤 3：实现分流逻辑
  ──────────────────────────────────────────────
  
  分流策略：
  1. 随机分配（最简单）
     user_id % 2 == 0 → A 组
     user_id % 2 == 1 → B 组
  
  2. 哈希分配（一致性更好）
     hash(user_id) % 100 < 50 → A 组
     else → B 组
  
  3. 分层实验（同时测试多个特性）
     Layer 1: Prompt 模板（A1 vs B1）
     Layer 2: 检索策略（A2 vs B2）
  
  
  步骤 4：收集数据
  ──────────────────────────────────────────────
  
  记录每次请求：
  {
    "timestamp": "2024-01-01T12:00:00Z",
    "user_id": "user_123",
    "group": "B",  // A 或 B
    "question": "...",
    "answer": "...",
    "metrics": {
      "latency_ms": 2345,
      "tokens_used": 567,
      "relevance_score": 0.85
    },
    "feedback": {
      "thumbs_up": true,
      "rating": 4
    }
  }
  
  
  步骤 5：统计分析
  ──────────────────────────────────────────────
  
  显著性检验：
  • t-test：比较两组均值差异
  • p-value < 0.05：差异显著
  
  效应量：
  • Cohen's d：衡量差异大小
  • d > 0.8：大效应
  
  置信区间：
  • 95% CI：估计真实差异范围
  
  
  步骤 6：决策
  ──────────────────────────────────────────────
  
  决策矩阵：
  ┌──────────┬──────────┬──────────┐
  │ 指标改善  │ 统计显著  │ 决策      │
  ├──────────┼──────────┼──────────┤
  │ ✓        │ ✓        │ 全量发布  │
  │ ✓        │ ✗        │ 继续测试  │
  │ ✗        │ ✓        │ 放弃方案  │
  │ ✗        │ ✗        │ 重新设计  │
  └──────────┴──────────┴──────────┘
    """
    print(ab_guide)
    
    # 代码示例
    print("\n【A/B 测试代码示例】")
    print("-" * 60)
    
    ab_code = '''
# file: ab_testing.py
"""
简单的 A/B 测试框架
"""

import hashlib
import random
from typing import Dict, List
from dataclasses import dataclass, field
import json


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    groups: Dict[str, float]  # 组名 -> 流量比例
    metric_to_track: List[str]


class ABTestManager:
    """A/B 测试管理器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {group: [] for group in config.groups.keys()}
    
    def assign_group(self, user_id: str) -> str:
        """
        根据用户 ID 分配实验组
        
        使用一致性哈希，确保同一用户始终分配到同一组
        """
        # 计算哈希值
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        normalized = hash_value % 10000 / 10000  # 归一化到 [0, 1]
        
        # 根据累积概率分配
        cumulative = 0
        for group, proportion in self.config.groups.items():
            cumulative += proportion
            if normalized < cumulative:
                return group
        
        # 兜底：返回最后一组
        return list(self.config.groups.keys())[-1]
    
    def record_result(self, user_id: str, group: str, metrics: Dict):
        """记录实验结果"""
        self.results[group].append({
            "user_id": user_id,
            "metrics": metrics
        })
    
    def analyze_results(self) -> Dict:
        """分析实验结果"""
        analysis = {}
        
        for group, results in self.results.items():
            if not results:
                continue
            
            # 计算每个指标的平均值
            avg_metrics = {}
            for metric in self.config.metric_to_track:
                values = [r["metrics"].get(metric, 0) for r in results]
                avg_metrics[metric] = sum(values) / len(values)
            
            analysis[group] = {
                "sample_size": len(results),
                "avg_metrics": avg_metrics
            }
        
        return analysis
    
    def export_results(self, filepath: str):
        """导出结果到文件"""
        with open(filepath, "w") as f:
            json.dump({
                "config": self.config.__dict__,
                "results": self.results
            }, f, indent=2)


# ── 使用示例 ──────────────────────────────────────────────

if __name__ == "__main__":
    # 1. 配置实验
    config = ExperimentConfig(
        name="rag_query_rewriting_test",
        groups={"control": 0.5, "treatment": 0.5},
        metric_to_track=["relevance_score", "latency_ms", "user_rating"]
    )
    
    # 2. 创建管理器
    ab_manager = ABTestManager(config)
    
    # 3. 模拟用户请求
    for i in range(1000):
        user_id = f"user_{i}"
        
        # 分配组
        group = ab_manager.assign_group(user_id)
        
        # 模拟指标（实际应从系统获取）
        if group == "control":
            metrics = {
                "relevance_score": random.gauss(0.75, 0.1),
                "latency_ms": random.gauss(2000, 300),
                "user_rating": random.gauss(3.5, 0.8)
            }
        else:  # treatment
            metrics = {
                "relevance_score": random.gauss(0.85, 0.1),  # 提升
                "latency_ms": random.gauss(2500, 300),       # 略慢
                "user_rating": random.gauss(4.2, 0.7)        # 提升
            }
        
        # 记录结果
        ab_manager.record_result(user_id, group, metrics)
    
    # 4. 分析结果
    analysis = ab_manager.analyze_results()
    
    print("实验结果分析：")
    for group, stats in analysis.items():
        print(f"\\n{group.upper()} 组:")
        print(f"  样本数: {stats['sample_size']}")
        for metric, value in stats['avg_metrics'].items():
            print(f"  {metric}: {value:.3f}")
    
    # 5. 导出
    ab_manager.export_results("ab_test_results.json")
    '''
    
    print(ab_code)
    print("-" * 60)


# =============================================================================
# Part 4：持续评估体系
# =============================================================================

def demo_continuous_evaluation():
    """
    建立持续评估和监控体系。
    """
    print("\n" + "=" * 60)
    print("Part 4: Continuous Evaluation System")
    print("=" * 60)
    
    evaluation_system = """
  🔄 持续评估体系架构：
  ──────────────────────────────────────────────
  
  离线评估（开发阶段）：
  ├─ RAGAS 批量测试
  ├─ 人工审核抽样
  └─ 回归测试套件
  
  在线评估（生产阶段）：
  ├─ 实时指标监控
  ├─ 用户反馈收集
  └─ A/B 测试平台
  
  自动化流程：
  ┌─────────────┐
  │ 代码提交     │
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │ 运行单元测试  │ ← CI/CD Pipeline
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │ RAGAS 评估   │ ← 必须通过阈值
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │ 部署到 Staging│
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │ A/B 测试     │ ← 小流量验证
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │ 全量发布     │
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │ 持续监控     │ ← 告警 + 自动回滚
  └─────────────┘
  
  
  📊 关键仪表盘：
  ──────────────────────────────────────────────
  
  1. 质量仪表盘
  • Faithfulness 趋势图
  • Answer Relevance 分布
  • 低分案例抽样
  
  2. 性能仪表盘
  • P50/P95/P99 延迟
  • Token 消耗速率
  • 错误率热力图
  
  3. 业务仪表盘
  • 活跃用户数
  • 会话时长
  • 用户满意度（ thumbs up/down ）
  
  4. 成本仪表盘
  • 每日 Token 费用
  • 各模型占比
  • 预算使用进度
  
  
  ⚠️  告警规则：
  ──────────────────────────────────────────────
  
  紧急告警（立即处理）：
  • 错误率 > 10% 持续 5 分钟
  • P95 延迟 > 30 秒
  • Faithfulness < 0.7
  
  警告告警（当天处理）：
  • 错误率 > 5% 持续 1 小时
  • Token 成本超出日预算 80%
  • 用户满意度下降 10%
  
  信息提示（本周关注）：
  • 流量增长 > 50%
  • 新出现的高频问题
  • 模型配额即将耗尽
  
  
  🛠️  工具推荐：
  ──────────────────────────────────────────────
  
  监控：
  • Prometheus + Grafana（指标）
  • ELK Stack（日志）
  • LangSmith（LangChain 专用）
  
  评估：
  • RAGAS（RAG 质量）
  • Ragas + TruLens（综合评估）
  • Custom Metrics（业务指标）
  
  实验：
  • MLflow（实验追踪）
  • Weights & Biases（可视化）
  • 自建 A/B 测试平台
    """
    print(evaluation_system)


def main():
    print("=" * 60)
    print("评估与测试详解")
    print("=" * 60)
    
    demo_ragas_evaluation()
    demo_unit_testing()
    demo_ab_testing()
    demo_continuous_evaluation()
    
    print("\n" + "=" * 60)
    print("学习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
