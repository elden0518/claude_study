# 🆕 新增内容说明

本文档说明为 `langchain_langgraph` 学习项目补充的完整知识体系。

---

## 📋 补充概览

原有课程（01-05）已经覆盖了 LangChain 和 LangGraph 的核心基础，但缺少**生产级应用**所需的关键知识。本次补充新增了 **6 个模块**，共计 **8 个新文件**，涵盖：

1. ✅ 生产就绪（缓存、重试、调试）
2. ✅ 高级 RAG 技巧
3. ✅ 高级 Agent 模式
4. ✅ 部署与监控
5. ✅ 评估与测试
6. ✅ 实战项目

---

## 📁 新增目录结构

```
langchain_langgraph/
├── 01_lc_basics/              # 原有 - LangChain 基础
├── 02_lc_advanced/            # 原有 - LangChain 进阶
├── 03_lg_basics/              # 原有 - LangGraph 基础
├── 04_lg_advanced/            # 原有 - LangGraph 进阶
├── 05_multi_agent/            # 原有 - 多 Agent 系统
│
├── 06_production_ready/       # ⭐ 新增 - 生产就绪
│   ├── 27_caching_layer.py
│   ├── 28_retry_and_fallback.py
│   └── 29_debugging_and_visualization.py
│
├── 07_advanced_rag/           # ⭐ 新增 - 高级 RAG
│   └── 30_advanced_rag_techniques.py
│
├── 08_advanced_agents/        # ⭐ 新增 - 高级 Agent
│   └── 31_advanced_agent_patterns.py
│
├── 09_deployment/             # ⭐ 新增 - 部署监控
│   └── 32_deployment_and_monitoring.py
│
├── 10_evaluation/             # ⭐ 新增 - 评估测试
│   └── 33_evaluation_and_testing.py
│
├── 11_projects/               # ⭐ 新增 - 实战项目
│   └── 34_project_customer_service_bot.py
│
├── LEARNING_PATH.md           # ⭐ 新增 - 完整学习路线
├── NEW_CONTENTS.md            # ⭐ 新增 - 本文件
├── requirements.txt           # 已更新 - 新增依赖
└── .env.example
```

---

## 📚 新增内容详解

### 模块 6：生产就绪（06_production_ready）

#### 27_caching_layer.py - 缓存层
**学习目标**：
- 理解为什么需要缓存（降低成本、减少延迟）
- 掌握 InMemoryCache、SQLiteCache
- 学会自定义缓存策略

**核心内容**：
```python
# 内存缓存（开发环境）
from langchain_core.caches import InMemoryCache
set_llm_cache(InMemoryCache())

# SQLite 缓存（持久化）
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))
```

**适用场景**：
- FAQ 问答系统（大量重复问题）
- 代码补全（常见模式复用）
- 批量数据处理（失败重试）

---

#### 28_retry_and_fallback.py - 重试与降级
**学习目标**：
- 掌握指数退避重试
- 学会配置 Fallback 链
- 设计优雅降级策略

**核心内容**：
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def call_llm_with_retry(question: str):
    # 带重试的 LLM 调用
    pass

# Fallback 链
llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])
```

**生产价值**：
- 提高系统可用性（从 95% → 99.9%）
- 降低 API 错误影响
- 保证基本功能可用

---

#### 29_debugging_and_visualization.py - 调试与可视化
**学习目标**：
- 导出 Mermaid 图
- 使用 LangGraph Studio
- 状态快照和历史回放

**核心内容**：
```python
# 导出 Mermaid 图
mermaid_code = app.get_graph().draw_mermaid()

# 状态快照
snapshot = app.get_state(config)
print(snapshot.values)
```

**工具推荐**：
- Mermaid：静态文档
- LangGraph Studio：交互式调试
- LangSmith：生产监控

---

### 模块 7：高级 RAG（07_advanced_rag）

#### 30_advanced_rag_techniques.py - RAG 高级技巧
**学习目标**：
- 查询重写（Query Rewriting）
- 混合检索（Hybrid Search）
- 重排序（Re-ranking）
- 多跳推理（Multi-hop）

**核心技巧**：

1. **查询重写**
```python
# LLM 优化用户问题
rewrite_prompt = "将问题转换为更适合检索的形式..."
rewritten_query = llm.invoke(original_question)
```

2. **混合检索**
```python
from langchain.retrievers import EnsembleRetriever

ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)
```

3. **重排序**
```python
from langchain_cohere import CohereRerank

reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)
```

4. **多跳推理**
```python
# 分解复杂问题 → 逐步检索 → 整合答案
sub_questions = decompose(complex_question)
results = [retrieve(q) for q in sub_questions]
answer = synthesize(results)
```

**性能提升**：
- 检索准确率：+15-20%
- 回答质量：显著提升
- 成本增加：约 2-3 倍（权衡使用）

---

### 模块 8：高级 Agent（08_advanced_agents）

#### 31_advanced_agent_patterns.py - Agent 高级模式
**学习目标**：
- Plan-and-Execute（先规划后执行）
- Self-Reflection（自我反思）
- ReWOO（减少 LLM 调用）

**模式对比**：

| 模式 | LLM 调用次数 | 灵活性 | 适用场景 |
|------|------------|--------|---------|
| ReAct | 高（每步调用） | ⭐⭐⭐⭐⭐ | 探索性任务 |
| Plan-and-Execute | 中（规划+执行） | ⭐⭐⭐ | 结构化任务 |
| Self-Reflection | 高（迭代优化） | ⭐⭐⭐⭐ | 高质量要求 |
| ReWOO | 低（批量执行） | ⭐⭐ | 效率优先 |

**选择指南**：
- 简单任务 → ReAct
- 中等复杂 → Plan-and-Execute
- 需要优化 → Self-Reflection
- 追求效率 → ReWOO

---

### 模块 9：部署监控（09_deployment）

#### 32_deployment_and_monitoring.py - 部署与监控
**学习目标**：
- FastAPI 封装 LangGraph
- Docker 容器化部署
- 水平扩展与负载均衡
- 生产环境配置

**核心内容**：

1. **FastAPI 封装**
```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest):
    result = agent_app.invoke(request.message)
    return {"response": result}
```

2. **Docker 部署**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0"]
```

3. **Kubernetes 扩展**
```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: langgraph-api:latest
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
```

**生产配置清单**：
- ✅ 超时控制（30-60s）
- ✅ 重试机制（3 次以内）
- ✅ 资源限制（CPU/内存）
- ✅ 健康检查
- ✅ 日志轮转
- ✅ 监控告警

---

### 模块 10：评估测试（10_evaluation）

#### 33_evaluation_and_testing.py - 评估与测试
**学习目标**：
- RAGAS 评估框架
- LangGraph 单元测试
- A/B 测试方法
- 持续评估体系

**RAGAS 核心指标**：

| 指标 | 含义 | 目标值 |
|------|------|--------|
| Faithfulness | 忠实度（无幻觉） | > 0.85 |
| Answer Relevance | 答案相关性 | > 0.90 |
| Context Precision | 上下文精确度 | > 0.75 |
| Context Recall | 上下文召回率 | > 0.80 |

**测试示例**：
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
print(result)
# {'faithfulness': 0.85, 'answer_relevancy': 0.92}
```

**A/B 测试流程**：
1. 定义假设和目标
2. 设计实验组（A/B）
3. 实现分流逻辑
4. 收集数据
5. 统计分析（t-test）
6. 决策（发布/放弃）

---

### 模块 11：实战项目（11_projects）

#### 34_project_customer_service_bot.py - 智能客服机器人
**项目目标**：
构建生产级客服系统，支持：
- 意图识别
- 知识检索（RAG）
- 答案生成
- 人工升级
- 反馈收集

**架构图**：
```
用户问题
  ↓
意图分类节点
  ↓
知识检索节点（RAG）
  ↓
人工升级判断 ──→ 是 ──→ 人工升级节点
  ↓ 否
答案生成节点
  ↓
响应格式化节点
  ↓
反馈收集节点（Human-in-the-Loop）
  ↓
结束
```

**技术栈**：
- LangGraph：工作流编排
- RAG：知识检索
- FastAPI：API 服务
- PostgreSQL：持久化

**学习价值**：
- 综合运用所有知识点
- 理解真实业务场景
- 掌握端到端开发流程

---

## 🎯 学习建议

### 顺序学习（推荐）

按照编号顺序学习，每个模块都依赖前面的知识：

```
27 → 28 → 29 → 30 → 31 → 32 → 33 → 34
```

### 按需学习

根据当前需求选择模块：

**想优化现有 RAG 系统**：
- 27（缓存）
- 30（高级 RAG）
- 33（评估）

**想部署到生产环境**：
- 28（重试降级）
- 32（部署监控）
- 33（测试）

**想构建复杂 Agent**：
- 29（调试）
- 31（高级模式）
- 34（项目实战）

---

## 📊 与原课程的对应关系

| 新增模块 | 前置知识 | 补充内容 |
|---------|---------|---------|
| 27 缓存 | 03 LCEL Chains | 性能优化 |
| 28 重试降级 | 05 Error Handling | 可靠性提升 |
| 29 调试可视化 | 13 Hello LangGraph | 开发效率 |
| 30 高级 RAG | 10 RAG Chain | 质量提升 |
| 31 高级 Agent | 11 Tools & Agents | 模式扩展 |
| 32 部署监控 | 26 Production Patterns | 运维能力 |
| 33 评估测试 | - | 质量保障 |
| 34 实战项目 | 全部 | 综合应用 |

---

## 🔧 安装新增依赖

```bash
# 进入项目目录
cd langchain_langgraph

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装新增依赖
pip install -r requirements.txt
```

**主要新增包**：
- `tenacity`：重试机制
- `fastapi` + `uvicorn`：API 服务
- `ragas`：RAG 评估
- `pytest`：单元测试
- `redis`：缓存
- `psycopg2-binary`：PostgreSQL

---

## 📖 相关文档

- [LEARNING_PATH.md](./LEARNING_PATH.md) - 完整学习路线图
- [README.md](../README.md) - 项目总览

---

## 💡 下一步行动

1. **阅读 LEARNING_PATH.md**：了解完整学习路线
2. **选择学习路径**：快速上手 / 系统学习 / 生产级应用
3. **动手实践**：运行每个示例文件
4. **构建项目**：基于 34 号文件创建自己的应用
5. **分享反馈**：遇到问题或有建议，欢迎提出

---

**更新时间**：2026-04-27  
**维护者**：Lingma AI Assistant
