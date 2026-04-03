# LangChain + LangGraph 学习课程设计文档

**日期：** 2026-04-03  
**目标用户：** 完全零基础，最终目标是构建多 Agent 协作系统  
**课程规模：** 完整版，共 26 个 demo  
**课程结构：** 线性渐进  
**LLM：** Claude 为主（`ppio/pa/claude-sonnet-4-6`），部分 demo 对比 OpenAI  
**位置：** `langchain_langgraph/`（当前仓库子目录，独立可运行）

---

## 整体结构

```
langchain_langgraph/
├── 01_lc_basics/          # LangChain 基础（6 个）
├── 02_lc_advanced/        # LangChain 进阶（6 个）
├── 03_lg_basics/          # LangGraph 基础（5 个）
├── 04_lg_advanced/        # LangGraph 进阶（6 个）
├── 05_multi_agent/        # 综合项目（3 个）
├── requirements.txt
└── README.md
```

**学习路径：**
```
01_lc_basics (6) → 02_lc_advanced (6) → 03_lg_basics (5) → 04_lg_advanced (6) → 05_multi_agent (3)
    LCEL/Chain/       RAG/Tool/              Graph/State/       Supervisor/            3 个完整
    Memory/Parser     Callback               Loop/Persist       Network/Stream         生产项目
```

---

## 模块一：`01_lc_basics/` — LangChain 基础（6 个 Demo）

| # | 文件 | 核心知识点 |
|---|------|-----------|
| 01 | `01_hello_langchain.py` | `ChatAnthropic` 初始化；与 Anthropic SDK 直接调用对比；`invoke` / `stream` / `batch` |
| 02 | `02_prompt_templates.py` | `PromptTemplate` / `ChatPromptTemplate` / `MessagesPlaceholder`；变量注入；few-shot |
| 03 | `03_lcel_chains.py` | LCEL 管道操作符 `\|`；`RunnableSequence` / `RunnableParallel`；链的组合与复用 |
| 04 | `04_output_parsers.py` | `StrOutputParser` / `JsonOutputParser` / `PydanticOutputParser`；结构化输出 |
| 05 | `05_memory.py` | `ConversationBufferMemory` / `ConversationSummaryMemory`；多轮对话管理 |
| 06 | `06_multi_model.py` | `ChatOpenAI` vs `ChatAnthropic` 无缝切换；展示 LangChain 模型无关性 |

**模块目标：** 掌握 LangChain 的核心抽象层（Model、Prompt、Chain、Output），为进阶 RAG 和 Agent 打基础。

---

## 模块二：`02_lc_advanced/` — LangChain 进阶（6 个 Demo）

| # | 文件 | 核心知识点 |
|---|------|-----------|
| 07 | `07_document_loaders.py` | `TextLoader` / `PyPDFLoader` / `WebBaseLoader` / `DirectoryLoader`；批量加载 |
| 08 | `08_text_splitters.py` | `RecursiveCharacterTextSplitter`；chunk_size / overlap；分块策略对比 |
| 09 | `09_embeddings_vectorstore.py` | `HuggingFaceEmbeddings` / `Chroma` / `FAISS`；相似度检索 |
| 10 | `10_rag_chain.py` | 完整 RAG pipeline：加载 → 分块 → 嵌入 → 检索 → 生成；`create_retrieval_chain` |
| 11 | `11_tools_agents.py` | `@tool` 装饰器；`create_react_agent`；ReAct 推理循环；工具调用追踪 |
| 12 | `12_callbacks.py` | `StdOutCallbackHandler`；自定义 Callback；追踪完整调用链；LangSmith 简介 |

**模块目标：** 掌握 RAG 全流程和 LangChain Agent，直接为 LangGraph 的 ToolNode 做铺垫。

---

## 模块三：`03_lg_basics/` — LangGraph 基础（5 个 Demo）

| # | 文件 | 核心知识点 |
|---|------|-----------|
| 13 | `13_hello_langgraph.py` | `StateGraph` + `TypedDict` State；节点函数；`add_edge`；`compile()` + `invoke()` |
| 14 | `14_conditional_edges.py` | `add_conditional_edges`；路由函数；根据 State 动态决定下一个节点 |
| 15 | `15_cycles_loops.py` | 循环图；`END` 退出条件；迭代自我改进（写作→评分→重写） |
| 16 | `16_human_in_the_loop.py` | `interrupt_before` / `interrupt_after`；暂停等待人工审核；恢复执行 |
| 17 | `17_persistence.py` | `MemorySaver` / `SqliteSaver`；`thread_id` 跨会话恢复；Checkpoint 机制 |

**模块目标：** 掌握 LangGraph 的核心概念（图结构、State、节点、边），理解它与普通 Chain 的本质区别：**可循环、可持久、可中断**。

---

## 模块四：`04_lg_advanced/` — LangGraph 进阶（6 个 Demo）

| # | 文件 | 核心知识点 |
|---|------|-----------|
| 18 | `18_subgraphs.py` | 子图嵌套；父图调用子图；模块化复杂工作流；State 传递机制 |
| 19 | `19_parallel_nodes.py` | `Send` API；扇出（fan-out）/ 扇入（fan-in）；多节点并行执行后汇总 |
| 20 | `20_tool_node.py` | `ToolNode`；`create_react_agent`（LangGraph 版）；完整 Agentic Loop |
| 21 | `21_streaming.py` | `graph.stream()` / `graph.astream_events()`；实时追踪每个节点的输出 |
| 22 | `22_supervisor_pattern.py` | **Supervisor 模式**：协调 Agent 动态调度多个专职 Agent；集中式控制 |
| 23 | `23_network_pattern.py` | **Network 模式**：Agent 之间直接传递消息；去中心化协作；Handoff 机制 |

**模块目标：** 掌握 LangGraph 的生产级特性，重点理解 Supervisor 和 Network 两种多 Agent 架构范式。

---

## 模块五：`05_multi_agent/` — 综合项目（3 个 Demo）

| # | 文件 | 项目描述 |
|---|------|---------|
| 24 | `24_research_team.py` | **研究团队**：搜索员（网络搜索）+ 分析师（信息提炼）+ 写作者（报告生成），Supervisor 调度，输出完整研究报告 |
| 25 | `25_code_review_team.py` | **代码审查团队**：开发者（写代码）+ 测试员（写测试）+ 审核者（评审反馈），循环迭代直到通过 |
| 26 | `26_production_patterns.py` | **生产级模式**：错误重试、节点超时、结构化日志、LangSmith 追踪接入 |

**模块目标：** 将前四个模块的知识整合为真实可运行的多 Agent 系统，每个项目独立完整。

---

## 技术约定

### 依赖（`requirements.txt`）

```
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-openai>=0.3.0
langchain-community>=0.3.0
langgraph>=0.3.0
chromadb>=0.5.0
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
python-dotenv>=1.0.0
```

### 代码风格（与现有 claude_study 一致）

- 每文件独立可运行，顶部有完整中文注释（学习目标、核心概念、前置知识）
- Windows UTF-8 编码修复块（`sys.stdout` 重定向）
- 模型统一使用 `"ppio/pa/claude-sonnet-4-6"`
- 每个 demo 内部分 Part 1 / Part 2 / Part 3 递进展示知识点
- 每节以 `print("=" * 60)` 分隔，方便运行时阅读输出

### 环境变量（`.env`）

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxx        # 仅 06/多模型对比 demo 需要
LANGCHAIN_API_KEY=xxxxxxxx        # 可选，接入 LangSmith 追踪
LANGCHAIN_TRACING_V2=true         # 可选
```

---

## 范围边界

**包含：**
- LangChain LCEL、RAG、Tool、Callback 核心 API
- LangGraph StateGraph、条件边、循环、持久化、子图、并行、流式
- 两种多 Agent 架构范式（Supervisor / Network）
- 3 个可运行的完整项目

**不包含：**
- LangChain v0.1/v0.2 旧版 API（直接使用 v0.3+）
- 向量数据库深度调优（Chroma/FAISS 仅示范基本用法）
- LangServe / LangGraph Platform 部署
- 与现有 `01_basics/` 至 `05_production/` 模块的代码复用（保持独立）
