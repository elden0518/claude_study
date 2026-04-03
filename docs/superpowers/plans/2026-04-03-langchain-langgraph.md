# LangChain + LangGraph 学习课程 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `langchain_langgraph/` 目录下创建 26 个带详细中文注解的学习 demo，覆盖 LangChain 基础/进阶和 LangGraph 基础/进阶，最终构建三个完整多 Agent 项目。

**Architecture:** 五个独立子模块线性递进（01→02→03→04→05），每个 demo 文件完全独立可运行，无跨文件依赖。所有 demo 顶部有完整中文学习目标注释，内部按 Part 1/2/3 分节展示知识点。

**Tech Stack:** `langchain>=0.3`, `langchain-anthropic>=0.3`, `langchain-openai>=0.3`, `langgraph>=0.3`, `chromadb`, `faiss-cpu`, `sentence-transformers`, `python-dotenv`

---

## 前置：目录结构与依赖

### Task 0: 搭建目录和依赖

**Files:**
- Create: `langchain_langgraph/requirements.txt`
- Create: `langchain_langgraph/.env.example`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p langchain_langgraph/01_lc_basics
mkdir -p langchain_langgraph/02_lc_advanced
mkdir -p langchain_langgraph/03_lg_basics
mkdir -p langchain_langgraph/04_lg_advanced
mkdir -p langchain_langgraph/05_multi_agent
```

- [ ] **Step 2: 写 requirements.txt**

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

保存到 `langchain_langgraph/requirements.txt`

- [ ] **Step 3: 写 .env.example**

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxx
LANGCHAIN_API_KEY=xxxxxxxx
LANGCHAIN_TRACING_V2=true
```

保存到 `langchain_langgraph/.env.example`

- [ ] **Step 4: 安装依赖**

```bash
pip install -r langchain_langgraph/requirements.txt
```

预期：所有包安装成功，无报错

- [ ] **Step 5: Commit**

```bash
git add langchain_langgraph/
git commit -m "chore: scaffold langchain_langgraph directory structure"
```

---

## Module 1: LangChain 基础

### Task 1: `01_hello_langchain.py`

**Files:**
- Create: `langchain_langgraph/01_lc_basics/01_hello_langchain.py`

- [ ] **Step 1: 写文件**

```python
"""
主题：Hello LangChain —— 第一次调用，理解它与直接用 SDK 的区别

学习目标：
  1. 理解 LangChain 是什么：统一接口层，不是 AI 模型本身
  2. 掌握 ChatAnthropic 的初始化和基本调用
  3. 对比 LangChain 与直接用 anthropic SDK 的写法差异
  4. 掌握三种调用方式：invoke / stream / batch
  5. 理解 AIMessage 响应对象的结构

核心概念：
  LangChain 不是模型，是框架。它把不同模型（Claude、GPT、Gemini）
  包装成统一接口，让你的代码可以无缝切换模型。
  
  ChatAnthropic  ←→  anthropic.Anthropic().messages.create()
  chain.invoke() ←→  client.messages.create()
  
前置知识：已完成 01_basics/01_hello_claude.py
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
from dotenv import load_dotenv
load_dotenv()

# ── LangChain 的 Anthropic 集成包 ──────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

MODEL = "ppio/pa/claude-sonnet-4-6"


# =============================================================================
# Part 1：直接用 anthropic SDK vs 用 LangChain（对比）
# =============================================================================

def demo_direct_sdk():
    """原生 SDK 调用方式（参照系）"""
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": "用一句话介绍Python"}]
    )
    print(f"[原生SDK] {resp.content[0].text}")


def demo_langchain_invoke():
    """LangChain 调用方式 —— invoke（同步，返回 AIMessage）"""
    llm = ChatAnthropic(model=MODEL, max_tokens=128)
    
    # invoke 接受字符串或消息列表
    response = llm.invoke("用一句话介绍Python")
    
    # response 是 AIMessage 对象
    print(f"[LangChain invoke] {response.content}")
    print(f"  类型: {type(response).__name__}")
    print(f"  usage: {response.usage_metadata}")


# =============================================================================
# Part 2：传入消息列表（System + Human）
# =============================================================================

def demo_messages():
    """使用消息列表，等价于设置 system prompt"""
    llm = ChatAnthropic(model=MODEL, max_tokens=256)
    
    messages = [
        SystemMessage(content="你是一位资深 Python 工程师，回答简洁。"),
        HumanMessage(content="什么是列表推导式？给一个例子。"),
    ]
    
    response = llm.invoke(messages)
    print(f"[消息列表调用]\n{response.content}")


# =============================================================================
# Part 3：流式输出
# =============================================================================

def demo_streaming():
    """stream() 方法 —— 实时输出每个 token"""
    llm = ChatAnthropic(model=MODEL, max_tokens=256)
    
    print("[流式输出] ", end="", flush=True)
    for chunk in llm.stream("请列举 Python 3 的三个新特性"):
        print(chunk.content, end="", flush=True)
    print()


# =============================================================================
# Part 4：批量调用（batch）
# =============================================================================

def demo_batch():
    """batch() 方法 —— 并发处理多个输入"""
    llm = ChatAnthropic(model=MODEL, max_tokens=64)
    
    questions = [
        "什么是 Python？一句话。",
        "什么是 JavaScript？一句话。",
        "什么是 Rust？一句话。",
    ]
    
    # batch 并发调用，返回列表
    responses = llm.batch(questions)
    print("[批量调用]")
    for q, r in zip(questions, responses):
        print(f"  Q: {q}")
        print(f"  A: {r.content}\n")


def main():
    print("=" * 60)
    print("Part 1：原生 SDK vs LangChain 对比")
    print("=" * 60)
    demo_direct_sdk()
    demo_langchain_invoke()
    
    print("\n" + "=" * 60)
    print("Part 2：消息列表（System + Human）")
    print("=" * 60)
    demo_messages()
    
    print("\n" + "=" * 60)
    print("Part 3：流式输出")
    print("=" * 60)
    demo_streaming()
    
    print("\n" + "=" * 60)
    print("Part 4：批量调用")
    print("=" * 60)
    demo_batch()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证运行**

```bash
cd langchain_langgraph && python 01_lc_basics/01_hello_langchain.py
```

预期：四个 Part 顺序输出，无报错，流式部分可见逐字打印

- [ ] **Step 3: Commit**

```bash
git add langchain_langgraph/01_lc_basics/01_hello_langchain.py
git commit -m "feat(lc-basics): 01 hello langchain with invoke/stream/batch"
```

---

### Task 2: `02_prompt_templates.py`

**Files:**
- Create: `langchain_langgraph/01_lc_basics/02_prompt_templates.py`

- [ ] **Step 1: 写文件**

```python
"""
主题：Prompt Templates —— 把提示词变成可复用的模板

学习目标：
  1. 理解为什么需要 Prompt Template（复用、版本管理、动态注入）
  2. 掌握 PromptTemplate（纯文本）的创建和使用
  3. 掌握 ChatPromptTemplate（对话格式）的创建和使用
  4. 学会 MessagesPlaceholder（插入动态消息列表）
  5. 掌握 few-shot prompt（示例驱动提示）

核心概念：
  Prompt Template = 带变量的字符串模板
  {变量名} 占位符在 invoke 时被替换为实际值
  
  PromptTemplate       → 单一字符串，适合补全模型
  ChatPromptTemplate   → 消息列表，适合对话模型（Claude、GPT）
  MessagesPlaceholder  → 在模板中插入动态的消息历史
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=256)


# =============================================================================
# Part 1：PromptTemplate（纯文本）
# =============================================================================

def demo_prompt_template():
    """基础字符串模板，用 {变量} 占位"""
    template = PromptTemplate.from_template(
        "请用{language}写一个函数，功能是{task}。只输出代码，不要解释。"
    )
    
    # format_prompt() → 查看渲染后的完整 prompt
    rendered = template.format(language="Python", task="计算斐波那契数列")
    print(f"[渲染后的 Prompt]\n{rendered}\n")
    
    # 与 LLM 串联（| 操作符，LCEL 基础）
    from langchain_core.output_parsers import StrOutputParser
    chain = template | llm | StrOutputParser()
    result = chain.invoke({"language": "Python", "task": "反转字符串"})
    print(f"[模型输出]\n{result}")


# =============================================================================
# Part 2：ChatPromptTemplate（对话格式）
# =============================================================================

def demo_chat_prompt_template():
    """对话格式模板，包含 system/human/ai 角色"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位{expertise}专家，用{style}风格回答。"),
        ("human", "{question}"),
    ])
    
    # format_messages() → 查看渲染后的消息列表
    messages = prompt.format_messages(
        expertise="Python",
        style="简洁",
        question="什么是装饰器？"
    )
    print(f"[渲染后的消息列表]")
    for msg in messages:
        print(f"  {msg.__class__.__name__}: {msg.content[:50]}")
    
    # 与 LLM 串联
    from langchain_core.output_parsers import StrOutputParser
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "expertise": "数据库",
        "style": "举例子",
        "question": "什么是索引？"
    })
    print(f"\n[模型输出]\n{result}")


# =============================================================================
# Part 3：MessagesPlaceholder（插入动态历史）
# =============================================================================

def demo_messages_placeholder():
    """在模板中插入动态的对话历史，用于多轮对话"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位耐心的编程助手。"),
        MessagesPlaceholder(variable_name="history"),  # 动态插入历史消息
        ("human", "{input}"),
    ])
    
    # 模拟已有对话历史
    history = [
        HumanMessage(content="我在学Python"),
        AIMessage(content="很好！Python 是很棒的入门语言。"),
    ]
    
    from langchain_core.output_parsers import StrOutputParser
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "history": history,
        "input": "我应该先学哪个部分？"
    })
    print(f"[含历史的对话]\n{result}")


# =============================================================================
# Part 4：Few-shot Prompt（示例驱动）
# =============================================================================

def demo_few_shot():
    """通过示例告诉模型期望的输入输出格式"""
    examples = [
        {"input": "快乐", "output": "悲伤"},
        {"input": "黑暗", "output": "光明"},
        {"input": "寒冷", "output": "温暖"},
    ]
    
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "请给出输入词的反义词，只输出一个词。"),
        few_shot_prompt,
        ("human", "{input}"),
    ])
    
    from langchain_core.output_parsers import StrOutputParser
    chain = final_prompt | llm | StrOutputParser()
    result = chain.invoke({"input": "成功"})
    print(f"[Few-shot 反义词] 成功 → {result}")


def main():
    print("=" * 60)
    print("Part 1：PromptTemplate（纯文本）")
    print("=" * 60)
    demo_prompt_template()
    
    print("\n" + "=" * 60)
    print("Part 2：ChatPromptTemplate（对话格式）")
    print("=" * 60)
    demo_chat_prompt_template()
    
    print("\n" + "=" * 60)
    print("Part 3：MessagesPlaceholder（动态历史）")
    print("=" * 60)
    demo_messages_placeholder()
    
    print("\n" + "=" * 60)
    print("Part 4：Few-shot Prompt")
    print("=" * 60)
    demo_few_shot()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证运行**

```bash
python 01_lc_basics/02_prompt_templates.py
```

预期：四个 Part 顺序输出，few-shot 返回一个反义词

- [ ] **Step 3: Commit**

```bash
git add langchain_langgraph/01_lc_basics/02_prompt_templates.py
git commit -m "feat(lc-basics): 02 prompt templates - PromptTemplate/ChatPromptTemplate/few-shot"
```

---

### Task 3: `03_lcel_chains.py`

**Files:**
- Create: `langchain_langgraph/01_lc_basics/03_lcel_chains.py`

- [ ] **Step 1: 写文件**

```python
"""
主题：LCEL 链（LangChain Expression Language）—— 用 | 把组件串联

学习目标：
  1. 理解 LCEL 是什么：用 | 操作符把 Runnable 对象串联成流水线
  2. 掌握基础链：prompt | llm | parser
  3. 掌握 RunnableParallel（并行执行多个分支）
  4. 掌握 RunnablePassthrough（透传输入值）
  5. 掌握链的组合（链套链）

核心概念：
  LCEL（LangChain Expression Language）
  每个 Runnable 都有 invoke / stream / batch 方法
  | 操作符等价于 RunnableSequence(a, b, c)
  
  a | b | c  ≡  RunnableSequence(steps=[a, b, c])
  调用时：输入 → a → b → c → 输出
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=256)


# =============================================================================
# Part 1：基础链 prompt | llm | parser
# =============================================================================

def demo_basic_chain():
    """最经典的三段式链"""
    prompt = ChatPromptTemplate.from_template("用一句话解释：{concept}")
    parser = StrOutputParser()
    
    # | 把三个 Runnable 串联
    chain = prompt | llm | parser
    
    result = chain.invoke({"concept": "递归"})
    print(f"[基础链] {result}")
    
    # 查看链的结构
    print(f"\n链的类型: {type(chain).__name__}")
    print(f"链的步骤数: {len(chain.steps)}")


# =============================================================================
# Part 2：RunnableParallel（并行分支）
# =============================================================================

def demo_parallel():
    """同时运行多个分支，结果合并为字典"""
    prompt = ChatPromptTemplate.from_template("{topic}")
    
    # 两个不同的后续处理分支
    pros_chain = (
        ChatPromptTemplate.from_template("列举{topic}的3个优点，每条一句话")
        | llm | StrOutputParser()
    )
    cons_chain = (
        ChatPromptTemplate.from_template("列举{topic}的3个缺点，每条一句话")
        | llm | StrOutputParser()
    )
    
    # RunnableParallel 同时调用两个链
    parallel = RunnableParallel(
        pros=pros_chain,
        cons=cons_chain,
    )
    
    result = parallel.invoke({"topic": "Python"})
    print(f"[并行链] Python 分析:")
    print(f"\n优点:\n{result['pros']}")
    print(f"\n缺点:\n{result['cons']}")


# =============================================================================
# Part 3：RunnablePassthrough（透传）
# =============================================================================

def demo_passthrough():
    """把原始输入与处理结果同时传递给下一步"""
    def format_docs(docs):
        return "\n".join(docs)
    
    # 模拟检索到的文档
    fake_docs = ["Python是解释型语言", "Python支持面向对象", "Python有丰富的库"]
    
    chain = (
        {
            "context": RunnableLambda(lambda _: format_docs(fake_docs)),
            "question": RunnablePassthrough(),   # 原样透传输入
        }
        | ChatPromptTemplate.from_template(
            "根据以下背景回答问题:\n\n背景:{context}\n\n问题:{question}"
        )
        | llm
        | StrOutputParser()
    )
    
    result = chain.invoke("Python适合做什么？")
    print(f"[透传链] {result}")


# =============================================================================
# Part 4：链的组合（链套链）
# =============================================================================

def demo_chain_composition():
    """把两个链组合成更长的链"""
    # 第一条链：生成文章大纲
    outline_chain = (
        ChatPromptTemplate.from_template("为主题「{topic}」生成一个3点文章大纲，每点10字内")
        | llm
        | StrOutputParser()
    )
    
    # 第二条链：根据大纲写摘要
    summary_chain = (
        ChatPromptTemplate.from_template("根据以下大纲写一句话摘要:\n{outline}")
        | llm
        | StrOutputParser()
    )
    
    # 串联：第一条链的输出作为第二条链的输入
    full_chain = outline_chain | RunnableLambda(lambda x: {"outline": x}) | summary_chain
    
    result = full_chain.invoke({"topic": "机器学习入门"})
    print(f"[组合链] {result}")


def main():
    print("=" * 60)
    print("Part 1：基础链 prompt | llm | parser")
    print("=" * 60)
    demo_basic_chain()
    
    print("\n" + "=" * 60)
    print("Part 2：RunnableParallel（并行分支）")
    print("=" * 60)
    demo_parallel()
    
    print("\n" + "=" * 60)
    print("Part 3：RunnablePassthrough（透传）")
    print("=" * 60)
    demo_passthrough()
    
    print("\n" + "=" * 60)
    print("Part 4：链的组合（链套链）")
    print("=" * 60)
    demo_chain_composition()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证运行**

```bash
python 01_lc_basics/03_lcel_chains.py
```

预期：四个 Part 输出，并行链同时返回优点和缺点两部分

- [ ] **Step 3: Commit**

```bash
git add langchain_langgraph/01_lc_basics/03_lcel_chains.py
git commit -m "feat(lc-basics): 03 LCEL chains - pipe operator, parallel, passthrough"
```

---

### Task 4: `04_output_parsers.py`

**Files:**
- Create: `langchain_langgraph/01_lc_basics/04_output_parsers.py`

- [ ] **Step 1: 写文件**

```python
"""
主题：Output Parsers —— 把模型输出变成结构化数据

学习目标：
  1. 理解为什么需要 Output Parser（模型输出是字符串，应用需要结构化数据）
  2. 掌握 StrOutputParser（字符串，最简单）
  3. 掌握 JsonOutputParser（解析 JSON 输出）
  4. 掌握 PydanticOutputParser（强类型结构化输出，含格式指令）
  5. 理解格式指令（format_instructions）的作用

核心概念：
  模型只能输出文本 → Output Parser 负责把文本转换成 Python 对象
  PydanticOutputParser 会自动生成格式指令注入 prompt，
  告诉模型"我需要你输出这种格式的 JSON"
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=512)


# =============================================================================
# Part 1：StrOutputParser（字符串）
# =============================================================================

def demo_str_parser():
    """最简单的 parser：把 AIMessage 变成纯字符串"""
    chain = (
        ChatPromptTemplate.from_template("用一句话解释{concept}")
        | llm
        | StrOutputParser()
    )
    result = chain.invoke({"concept": "封装"})
    print(f"[StrOutputParser] 类型: {type(result).__name__}")
    print(f"  内容: {result}")


# =============================================================================
# Part 2：JsonOutputParser
# =============================================================================

def demo_json_parser():
    """解析 JSON 格式的输出，返回 Python dict/list"""
    parser = JsonOutputParser()
    
    chain = (
        ChatPromptTemplate.from_template(
            "返回一个 JSON 对象，包含以下字段：\n"
            "name（语言名）, year（发明年份）, paradigm（编程范式列表）\n"
            "介绍编程语言：{language}\n"
            "只输出 JSON，不要其他内容。"
        )
        | llm
        | parser
    )
    
    result = chain.invoke({"language": "Python"})
    print(f"[JsonOutputParser] 类型: {type(result).__name__}")
    print(f"  name: {result.get('name')}")
    print(f"  year: {result.get('year')}")
    print(f"  paradigm: {result.get('paradigm')}")


# =============================================================================
# Part 3：PydanticOutputParser（强类型）
# =============================================================================

class BookReview(BaseModel):
    """书评结构"""
    title: str = Field(description="书名")
    author: str = Field(description="作者")
    rating: int = Field(description="评分，1-10分")
    summary: str = Field(description="一句话简介")
    pros: List[str] = Field(description="优点列表，3条")
    cons: List[str] = Field(description="缺点列表，2条")


def demo_pydantic_parser():
    """PydanticOutputParser：输出强类型的 Pydantic 对象"""
    from langchain_core.output_parsers import PydanticOutputParser
    
    parser = PydanticOutputParser(pydantic_object=BookReview)
    
    # parser.get_format_instructions() 自动生成格式要求
    print(f"[格式指令预览]\n{parser.get_format_instructions()[:200]}...\n")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位书评家。\n\n{format_instructions}"),
        ("human", "请给《{book}》写一篇结构化书评"),
    ]).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | llm | parser
    result = chain.invoke({"book": "Python编程：从入门到实践"})
    
    print(f"[PydanticOutputParser] 类型: {type(result).__name__}")
    print(f"  书名: {result.title}")
    print(f"  作者: {result.author}")
    print(f"  评分: {result.rating}/10")
    print(f"  简介: {result.summary}")
    print(f"  优点: {result.pros}")
    print(f"  缺点: {result.cons}")


def main():
    print("=" * 60)
    print("Part 1：StrOutputParser")
    print("=" * 60)
    demo_str_parser()
    
    print("\n" + "=" * 60)
    print("Part 2：JsonOutputParser")
    print("=" * 60)
    demo_json_parser()
    
    print("\n" + "=" * 60)
    print("Part 3：PydanticOutputParser（强类型）")
    print("=" * 60)
    demo_pydantic_parser()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证运行**

```bash
python 01_lc_basics/04_output_parsers.py
```

预期：Pydantic 对象的字段均有值，JSON 格式正确解析

- [ ] **Step 3: Commit**

```bash
git add langchain_langgraph/01_lc_basics/04_output_parsers.py
git commit -m "feat(lc-basics): 04 output parsers - Str/Json/Pydantic"
```

---

### Task 5: `05_memory.py`

**Files:**
- Create: `langchain_langgraph/01_lc_basics/05_memory.py`

- [ ] **Step 1: 写文件**

```python
"""
主题：Memory（记忆）—— 让对话有上下文

学习目标：
  1. 理解 LLM 本身无状态，Memory 是外部维护历史的机制
  2. 掌握手动管理消息历史（最透明的方式）
  3. 掌握 ChatMessageHistory（标准历史存储）
  4. 掌握 RunnableWithMessageHistory（LCEL 链的历史管理包装器）
  5. 对比不同 Memory 策略：完整历史 vs 摘要历史

核心概念：
  LLM 的"记忆"本质：把历史消息作为上下文一起发给模型
  每次对话都要携带完整历史 → token 越来越多 → 需要管理策略
  
  完整历史（Buffer）: 保留所有消息，简单但 token 多
  摘要历史（Summary）: 把旧消息压缩成摘要，节省 token
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=256)


# =============================================================================
# Part 1：手动管理历史（最透明的方式）
# =============================================================================

def demo_manual_history():
    """手动维护消息列表 —— 最直接，最透明"""
    from langchain_core.messages import HumanMessage, AIMessage
    
    history = []
    
    def chat(user_input: str) -> str:
        history.append(HumanMessage(content=user_input))
        response = llm.invoke(history)
        history.append(AIMessage(content=response.content))
        return response.content
    
    print("── 手动历史对话 ──")
    r1 = chat("我叫小明，今年学Python")
    print(f"  User: 我叫小明，今年学Python")
    print(f"  AI: {r1[:60]}...")
    
    r2 = chat("我的名字是什么？")
    print(f"  User: 我的名字是什么？")
    print(f"  AI: {r2}")
    
    print(f"  历史消息数: {len(history)}")


# =============================================================================
# Part 2：RunnableWithMessageHistory（LCEL 链的历史管理）
# =============================================================================

def demo_runnable_with_history():
    """用 RunnableWithMessageHistory 包装链，自动管理历史"""
    
    # 构建带历史插槽的 prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位友好的编程助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # 存储每个 session 的历史
    store: dict[str, InMemoryChatMessageHistory] = {}
    
    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    # 用 RunnableWithMessageHistory 包装
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    config_alice = {"configurable": {"session_id": "alice"}}
    config_bob   = {"configurable": {"session_id": "bob"}}
    
    print("── session: alice ──")
    r = chain_with_history.invoke({"input": "我叫Alice，我在学习数据分析"}, config=config_alice)
    print(f"  AI: {r[:60]}...")
    r = chain_with_history.invoke({"input": "我的名字是什么？"}, config=config_alice)
    print(f"  AI: {r}")
    
    print("\n── session: bob（独立历史）──")
    r = chain_with_history.invoke({"input": "我的名字是什么？"}, config=config_bob)
    print(f"  AI: {r}")  # bob 没有历史，不知道名字
    
    print(f"\n  Alice 历史消息数: {len(store['alice'].messages)}")


# =============================================================================
# Part 3：历史管理策略对比
# =============================================================================

def demo_history_strategies():
    """对比完整历史 vs 手动截断的 token 消耗"""
    from langchain_anthropic import ChatAnthropic
    
    # 模拟10轮对话后的历史
    from langchain_core.messages import HumanMessage, AIMessage
    long_history = []
    for i in range(10):
        long_history.append(HumanMessage(content=f"第{i+1}轮问题：{['Python','Java','Go','Rust','JS','TS','C++','Swift','Kotlin','Ruby'][i]}有什么特点？"))
        long_history.append(AIMessage(content=f"第{i+1}轮回答：这是一种编程语言，有各自特点。"))
    
    print(f"── 历史策略对比 ──")
    print(f"  10轮完整历史消息数: {len(long_history)}")
    
    # 策略1：保留最近 N 轮
    keep_last_n = 4
    trimmed = long_history[-keep_last_n * 2:]
    print(f"  保留最近{keep_last_n}轮后消息数: {len(trimmed)}")
    print(f"  → 适合：对话内容无强依赖，只需近期上下文")
    print(f"  → 适合：节省token，降低成本")
    
    # 策略2：完整保留
    print(f"\n  完整历史消息数: {len(long_history)}")
    print(f"  → 适合：对话内容高度依赖早期信息（如姓名、需求）")
    print(f"  → 代价：token 随对话轮数线性增长")


def main():
    print("=" * 60)
    print("Part 1：手动管理历史")
    print("=" * 60)
    demo_manual_history()
    
    print("\n" + "=" * 60)
    print("Part 2：RunnableWithMessageHistory")
    print("=" * 60)
    demo_runnable_with_history()
    
    print("\n" + "=" * 60)
    print("Part 3：历史管理策略对比")
    print("=" * 60)
    demo_history_strategies()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证运行**

```bash
python 01_lc_basics/05_memory.py
```

预期：Alice session 能记住名字，Bob session 不知道名字（独立）

- [ ] **Step 3: Commit**

```bash
git add langchain_langgraph/01_lc_basics/05_memory.py
git commit -m "feat(lc-basics): 05 memory - manual history and RunnableWithMessageHistory"
```

---

### Task 6: `06_multi_model.py`

**Files:**
- Create: `langchain_langgraph/01_lc_basics/06_multi_model.py`

- [ ] **Step 1: 写文件**

```python
"""
主题：多模型切换 —— LangChain 最核心价值：一套代码多个模型

学习目标：
  1. 理解 LangChain 的"模型无关性"设计理念
  2. 对比 ChatAnthropic 和 ChatOpenAI 的接口一致性
  3. 学会在运行时动态切换模型
  4. 理解不同模型的响应差异
  5. 掌握 configurable_alternatives（可配置模型）

核心概念：
  LangChain 的核心价值之一：把不同厂商的模型包装成统一的 BaseChatModel 接口
  ChatAnthropic、ChatOpenAI、ChatGoogleGenerativeAI 都实现了相同的接口
  → 切换模型 = 换一行初始化代码，其余代码不变
  
  注意：需要在 .env 中同时配置 ANTHROPIC_API_KEY 和 OPENAI_API_KEY
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

ANTHROPIC_MODEL = "ppio/pa/claude-sonnet-4-6"


# =============================================================================
# Part 1：统一接口演示
# =============================================================================

def demo_unified_interface():
    """用同一个函数调用不同模型 —— 接口完全一致"""
    
    prompt = ChatPromptTemplate.from_template("用一句话（中文）解释：{concept}")
    parser = StrOutputParser()
    
    # Claude（Anthropic）
    claude = ChatAnthropic(model=ANTHROPIC_MODEL, max_tokens=128)
    claude_chain = prompt | claude | parser
    
    concept = "量子纠缠"
    claude_result = claude_chain.invoke({"concept": concept})
    print(f"[Claude] {claude_result}")
    
    # OpenAI（如果有 key 则运行，否则跳过）
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        gpt = ChatOpenAI(model="gpt-4o-mini", max_tokens=128)
        gpt_chain = prompt | gpt | parser
        gpt_result = gpt_chain.invoke({"concept": concept})
        print(f"[GPT-4o-mini] {gpt_result}")
    else:
        print(f"[GPT-4o-mini] 跳过（未配置 OPENAI_API_KEY）")
        print(f"  → 如果换成 ChatOpenAI，代码完全一样，只改一行初始化")


# =============================================================================
# Part 2：可配置模型（运行时切换）
# =============================================================================

def demo_configurable_model():
    """使用 configurable_alternatives 在运行时选择模型"""
    
    # 定义一个以 Claude 为默认、可配置替换模型的链
    claude = ChatAnthropic(model=ANTHROPIC_MODEL, max_tokens=128).configurable_alternatives(
        which={"id": "llm"},   # 配置项名称
        default_key="claude",
        # 如果有 OpenAI Key，可以添加：
        # openai=ChatOpenAI(model="gpt-4o-mini"),
    )
    
    chain = (
        ChatPromptTemplate.from_template("用一句话介绍：{topic}")
        | claude
        | StrOutputParser()
    )
    
    # 使用默认模型（Claude）
    result = chain.invoke(
        {"topic": "深度学习"},
        config={"configurable": {"llm": "claude"}}
    )
    print(f"[可配置链 - Claude] {result}")


# =============================================================================
# Part 3：模型参数对比
# =============================================================================

def demo_model_params():
    """展示不同模型参数的影响"""
    
    prompt = ChatPromptTemplate.from_template("写一首关于{topic}的四行短诗")
    parser = StrOutputParser()
    
    # 低温度（更确定性）
    cold = ChatAnthropic(model=ANTHROPIC_MODEL, max_tokens=128, temperature=0.0)
    # 高温度（更有创意）
    hot = ChatAnthropic(model=ANTHROPIC_MODEL, max_tokens=128, temperature=1.0)
    
    topic = "秋天"
    print(f"[temperature=0.0 - 确定性]\n{(prompt | cold | parser).invoke({'topic': topic})}")
    print(f"\n[temperature=1.0 - 创意]\n{(prompt | hot | parser).invoke({'topic': topic})}")


def main():
    print("=" * 60)
    print("Part 1：统一接口 —— 一套代码调用不同模型")
    print("=" * 60)
    demo_unified_interface()
    
    print("\n" + "=" * 60)
    print("Part 2：可配置模型（运行时切换）")
    print("=" * 60)
    demo_configurable_model()
    
    print("\n" + "=" * 60)
    print("Part 3：模型参数对比（temperature）")
    print("=" * 60)
    demo_model_params()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证运行**

```bash
python 01_lc_basics/06_multi_model.py
```

- [ ] **Step 3: Commit**

```bash
git add langchain_langgraph/01_lc_basics/06_multi_model.py
git commit -m "feat(lc-basics): 06 multi-model switching with configurable_alternatives"
```

---

## Module 2: LangChain 进阶

### Task 7-12: 进阶模块（07~12）

> 以下六个文件遵循相同模式：详细中文注释 + Part 分节 + main() 入口

**Task 7: `07_document_loaders.py`**
- Create: `langchain_langgraph/02_lc_advanced/07_document_loaders.py`
- 核心导入：`TextLoader`, `WebBaseLoader`, `DirectoryLoader` from `langchain_community.document_loaders`
- Part 1：`TextLoader` 加载本地 txt 文件，打印 `doc.page_content[:200]` 和 `doc.metadata`
- Part 2：`WebBaseLoader` 加载网页（`https://python.org`），提取纯文本
- Part 3：`DirectoryLoader` 批量加载目录下所有 `.txt`，展示文档数量
- 验证：`python 02_lc_advanced/07_document_loaders.py`

- [ ] **Step 1: 写 07_document_loaders.py**（结构同上，包含完整中文注释和三个 Part）
- [ ] **Step 2: 在 demo 目录创建 `sample_docs/` 测试文件夹，写入 2-3 个示例 txt**
- [ ] **Step 3: 验证运行**
- [ ] **Step 4: Commit** `feat(lc-advanced): 07 document loaders`

---

**Task 8: `08_text_splitters.py`**
- Create: `langchain_langgraph/02_lc_advanced/08_text_splitters.py`
- 核心导入：`RecursiveCharacterTextSplitter`, `CharacterTextSplitter` from `langchain_text_splitters`
- Part 1：对长文本演示 `RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)`，打印每个 chunk
- Part 2：对比不同 `chunk_size`（100/500/1000）的分块数量和内容预览
- Part 3：演示 `overlap` 的作用（相邻 chunk 共享内容）
- 验证：`python 02_lc_advanced/08_text_splitters.py`

- [ ] **Step 1: 写 08_text_splitters.py**
- [ ] **Step 2: 验证运行**
- [ ] **Step 3: Commit** `feat(lc-advanced): 08 text splitters`

---

**Task 9: `09_embeddings_vectorstore.py`**
- Create: `langchain_langgraph/02_lc_advanced/09_embeddings_vectorstore.py`
- 核心导入：`HuggingFaceEmbeddings` from `langchain_huggingface`（或 `langchain_community`）；`FAISS` from `langchain_community.vectorstores`
- Part 1：用 `HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")` 生成向量，打印维度
- Part 2：`FAISS.from_texts(texts, embeddings)` 构建向量库，演示 `similarity_search(query, k=3)`
- Part 3：`vectorstore.as_retriever()` 与 LCEL 链串联，做简单问答
- 注意：首次运行会下载模型（约 100MB），需要网络

- [ ] **Step 1: 写 09_embeddings_vectorstore.py**
- [ ] **Step 2: 验证运行**（首次运行会下载模型）
- [ ] **Step 3: Commit** `feat(lc-advanced): 09 embeddings and vector store`

---

**Task 10: `10_rag_chain.py`**
- Create: `langchain_langgraph/02_lc_advanced/10_rag_chain.py`
- 核心导入：以上所有加 `create_retrieval_chain`, `create_stuff_documents_chain`
- Part 1：加载 `sample_docs/` → 分块 → 嵌入 → FAISS
- Part 2：`create_retrieval_chain` 构建完整 RAG pipeline，提问并展示检索来源
- Part 3：对比有 RAG vs 无 RAG 的回答差异（同一问题）

- [ ] **Step 1: 写 10_rag_chain.py**
- [ ] **Step 2: 验证运行**
- [ ] **Step 3: Commit** `feat(lc-advanced): 10 complete RAG pipeline`

---

**Task 11: `11_tools_agents.py`**
- Create: `langchain_langgraph/02_lc_advanced/11_tools_agents.py`
- 核心导入：`@tool` from `langchain_core.tools`；`create_react_agent` from `langchain.agents`；`AgentExecutor`
- Part 1：用 `@tool` 定义 3 个工具（计算器、字符串处理、假天气查询），打印工具描述
- Part 2：`create_react_agent` + `AgentExecutor(verbose=True)` 跑 ReAct 循环，观察思考过程
- Part 3：多工具协作（要求同时用到 2 个工具的问题）

- [ ] **Step 1: 写 11_tools_agents.py**
- [ ] **Step 2: 验证运行**（verbose=True 可见 ReAct 推理步骤）
- [ ] **Step 3: Commit** `feat(lc-advanced): 11 tools and ReAct agent`

---

**Task 12: `12_callbacks.py`**
- Create: `langchain_langgraph/02_lc_advanced/12_callbacks.py`
- 核心导入：`StdOutCallbackHandler`, `BaseCallbackHandler` from `langchain_core.callbacks`
- Part 1：`StdOutCallbackHandler` 追踪链的完整调用过程（开始/结束/token）
- Part 2：自定义 `BaseCallbackHandler` 子类，记录每次 LLM 调用的 token 消耗
- Part 3：把 callback 传给 chain（`chain.invoke(input, config={"callbacks": [handler]})`）

- [ ] **Step 1: 写 12_callbacks.py**
- [ ] **Step 2: 验证运行**
- [ ] **Step 3: Commit** `feat(lc-advanced): 12 callbacks and call tracing`

---

## Module 3: LangGraph 基础

### Task 13: `13_hello_langgraph.py`

**Files:**
- Create: `langchain_langgraph/03_lg_basics/13_hello_langgraph.py`

- [ ] **Step 1: 写文件**

```python
"""
主题：Hello LangGraph —— 用图结构编排 AI 工作流

学习目标：
  1. 理解 LangGraph 是什么：用有向图描述 AI 工作流的框架
  2. 掌握核心三要素：State（状态）、Node（节点）、Edge（边）
  3. 理解 TypedDict 如何定义图的共享状态
  4. 掌握 StateGraph 的创建、编译和调用
  5. 理解 START / END 特殊节点的作用

核心概念：
  LangGraph vs LangChain Chain：
  Chain  = 线性流水线（A→B→C，固定顺序）
  Graph  = 有向图（可循环、可分支、可并行，更灵活）
  
  State  = 图中所有节点共享的数据字典（TypedDict）
  Node   = 处理 State 并返回更新的函数
  Edge   = 节点之间的连接关系
  
  工作流：
  1. 定义 State 类型
  2. 写节点函数（接收 State，返回更新字典）
  3. 建图：add_node → add_edge → compile
  4. 调用：graph.invoke(初始State)
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=256)


# =============================================================================
# Part 1：最简单的图（单节点）
# =============================================================================

# State 定义：图中所有节点共享的状态
class SimpleState(TypedDict):
    input: str       # 用户输入
    output: str      # 最终输出
    step_count: int  # 执行步骤计数


def process_node(state: SimpleState) -> dict:
    """节点函数：接收当前 State，返回要更新的字段"""
    print(f"  [process_node] 收到输入: {state['input']}")
    
    response = llm.invoke(f"用一句话回答：{state['input']}")
    
    return {
        "output": response.content,
        "step_count": state["step_count"] + 1,
    }


def demo_simple_graph():
    """构建并运行最简单的单节点图"""
    # 1. 创建图，指定 State 类型
    graph = StateGraph(SimpleState)
    
    # 2. 添加节点（名称，函数）
    graph.add_node("process", process_node)
    
    # 3. 添加边（START → 节点 → END）
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    
    # 4. 编译（生成可执行图）
    app = graph.compile()
    
    # 5. 调用
    result = app.invoke({
        "input": "什么是人工智能？",
        "output": "",
        "step_count": 0,
    })
    
    print(f"\n[简单图] 最终State:")
    print(f"  input     : {result['input']}")
    print(f"  output    : {result['output']}")
    print(f"  step_count: {result['step_count']}")


# =============================================================================
# Part 2：多节点顺序图
# =============================================================================

class PipelineState(TypedDict):
    user_query: str
    refined_query: str   # 改写后的查询
    final_answer: str


def refine_node(state: PipelineState) -> dict:
    """节点1：改写用户问题，使其更清晰"""
    print(f"  [refine_node] 改写问题...")
    response = llm.invoke(
        f"把这个问题改写得更清晰（只输出改写后的问题）：{state['user_query']}"
    )
    return {"refined_query": response.content}


def answer_node(state: PipelineState) -> dict:
    """节点2：回答改写后的问题"""
    print(f"  [answer_node] 回答问题...")
    response = llm.invoke(f"请简洁地回答：{state['refined_query']}")
    return {"final_answer": response.content}


def demo_multi_node_graph():
    """多节点顺序图：问题改写 → 回答"""
    graph = StateGraph(PipelineState)
    
    graph.add_node("refine", refine_node)
    graph.add_node("answer", answer_node)
    
    graph.add_edge(START, "refine")
    graph.add_edge("refine", "answer")
    graph.add_edge("answer", END)
    
    app = graph.compile()
    
    result = app.invoke({
        "user_query": "py咋学",
        "refined_query": "",
        "final_answer": "",
    })
    
    print(f"\n[多节点图]")
    print(f"  原始问题 : {result['user_query']}")
    print(f"  改写问题 : {result['refined_query']}")
    print(f"  最终回答 : {result['final_answer'][:80]}...")


# =============================================================================
# Part 3：消息图（内置 add_messages reducer）
# =============================================================================

class ChatState(TypedDict):
    # Annotated[list, add_messages] 表示：
    # messages 字段使用 add_messages reducer（自动追加，不是覆盖）
    messages: Annotated[list, add_messages]


def chat_node(state: ChatState) -> dict:
    """聊天节点：调用 LLM，追加回复消息"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # add_messages 会把这条消息追加到列表


def demo_message_graph():
    """消息图：最接近真实聊天机器人的模式"""
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    app = graph.compile()
    
    result = app.invoke({
        "messages": [HumanMessage(content="你好，用一句话自我介绍")]
    })
    
    print(f"\n[消息图]")
    for msg in result["messages"]:
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content[:80]}")


def main():
    print("=" * 60)
    print("Part 1：最简单的图（单节点）")
    print("=" * 60)
    demo_simple_graph()
    
    print("\n" + "=" * 60)
    print("Part 2：多节点顺序图")
    print("=" * 60)
    demo_multi_node_graph()
    
    print("\n" + "=" * 60)
    print("Part 3：消息图（add_messages reducer）")
    print("=" * 60)
    demo_message_graph()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证运行**

```bash
python 03_lg_basics/13_hello_langgraph.py
```

- [ ] **Step 3: Commit**

```bash
git add langchain_langgraph/03_lg_basics/13_hello_langgraph.py
git commit -m "feat(lg-basics): 13 hello langgraph - StateGraph, nodes, edges"
```

---

### Task 14-17: LangGraph 基础其余文件

**Task 14: `14_conditional_edges.py`**

- Create: `langchain_langgraph/03_lg_basics/14_conditional_edges.py`
- Part 1：定义路由函数（分析 State 返回节点名字符串），用 `add_conditional_edges`
- Part 2：实现"分诊"图：用户输入 → 分类节点 → 根据类别路由到不同处理节点（技术/非技术）
- Part 3：多分支路由（3+ 个出口）

```python
# 路由函数示例
def route(state: State) -> str:
    if state["category"] == "technical":
        return "tech_node"
    else:
        return "general_node"

graph.add_conditional_edges("classify", route, {
    "tech_node": "tech_node",
    "general_node": "general_node",
})
```

- [ ] **Step 1: 写文件（含完整中文注释）**
- [ ] **Step 2: 验证运行**
- [ ] **Step 3: Commit** `feat(lg-basics): 14 conditional edges and routing`

---

**Task 15: `15_cycles_loops.py`**

- Create: `langchain_langgraph/03_lg_basics/15_cycles_loops.py`
- Part 1：实现写作→评分→重写的循环图，循环条件：评分 < 8 且循环次数 < 3
- Part 2：State 中用 `iteration_count` 计数，路由函数检查是否退出
- Part 3：对比无循环 vs 有循环的输出质量

```python
class WritingState(TypedDict):
    topic: str
    draft: str
    score: int
    iteration: int

def should_continue(state: WritingState) -> str:
    if state["score"] >= 8 or state["iteration"] >= 3:
        return END
    return "write"
```

- [ ] **Step 1: 写文件**
- [ ] **Step 2: 验证运行**
- [ ] **Step 3: Commit** `feat(lg-basics): 15 cycles and loops`

---

**Task 16: `16_human_in_the_loop.py`**

- Create: `langchain_langgraph/03_lg_basics/16_human_in_the_loop.py`
- 核心导入：`interrupt` from `langgraph.types`；`MemorySaver`；`Command`
- Part 1：在节点内用 `interrupt("请审核此内容")` 暂停图执行
- Part 2：检查暂停状态（`graph.get_state(config)`），用 `Command(resume=...)` 恢复
- Part 3：演示"AI 写草稿 → 暂停人工审核 → 继续执行发布"完整流程

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

def review_node(state):
    # 暂停，等待人工输入
    human_feedback = interrupt("请审核以下内容并输入反馈：\n" + state["draft"])
    return {"feedback": human_feedback}

# 编译时必须传入 checkpointer（interrupt 依赖持久化）
app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "review-1"}}

# 第一次运行（会在 interrupt 处暂停）
app.invoke(initial_state, config)

# 恢复执行（传入人工反馈）
app.invoke(Command(resume="内容通过，请继续"), config)
```

- [ ] **Step 1: 写文件**
- [ ] **Step 2: 验证运行**
- [ ] **Step 3: Commit** `feat(lg-basics): 16 human-in-the-loop with interrupt`

---

**Task 17: `17_persistence.py`**

- Create: `langchain_langgraph/03_lg_basics/17_persistence.py`
- 核心导入：`MemorySaver`, `SqliteSaver`（或 `InMemorySaver`）from `langgraph.checkpoint`
- Part 1：`MemorySaver` + `thread_id` 实现跨调用的状态持久化
- Part 2：`graph.get_state(config)` 查看当前 checkpoint；`graph.get_state_history(config)` 查看历史
- Part 3：从历史 checkpoint 恢复（时间旅行）

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "session-1"}}

# 第一次对话
app.invoke({"messages": [HumanMessage("我叫小明")]}, config)

# 第二次对话（自动读取之前的状态）
result = app.invoke({"messages": [HumanMessage("我叫什么？")]}, config)
```

- [ ] **Step 1: 写文件**
- [ ] **Step 2: 验证运行**（验证跨调用记忆名字）
- [ ] **Step 3: Commit** `feat(lg-basics): 17 persistence with MemorySaver`

---

## Module 4: LangGraph 进阶

### Task 18-23: 进阶模块（遵循相同模式）

**Task 18: `18_subgraphs.py`**

- Create: `langchain_langgraph/04_lg_advanced/18_subgraphs.py`
- 核心：父图把子图作为一个节点调用（`parent.add_node("sub", subgraph_app)`）
- Part 1：定义独立的子图（有自己的 State 和节点），编译为 `subgraph_app`
- Part 2：父图把 `subgraph_app` 当作普通节点嵌入
- Part 3：演示"父图 State → 子图 State 映射 → 子图结果回写父图"

- [ ] **Step 1: 写文件**
- [ ] **Step 2: 验证运行**
- [ ] **Step 3: Commit** `feat(lg-advanced): 18 subgraphs`

---

**Task 19: `19_parallel_nodes.py`**

- Create: `langchain_langgraph/04_lg_advanced/19_parallel_nodes.py`
- 核心：`Send` API 实现动态并行（运行时决定有多少个并行分支）
- Part 1：静态并行（从一个节点出发连接多个节点，汇聚到一个节点）
- Part 2：`Send` 动态扇出（根据列表动态创建 N 个并行任务）
- Part 3：并行汇总（用 `operator.add` reducer 收集所有并行结果）

```python
from langgraph.types import Send

def fan_out(state: State):
    # 动态创建 N 个并行任务
    return [Send("process_item", {"item": item}) for item in state["items"]]

graph.add_conditional_edges("start", fan_out)
```

- [ ] **Step 1: 写文件**
- [ ] **Step 2: 验证运行**
- [ ] **Step 3: Commit** `feat(lg-advanced): 19 parallel nodes with Send API`

---

**Task 20: `20_tool_node.py`**

- Create: `langchain_langgraph/04_lg_advanced/20_tool_node.py`
- 核心：`ToolNode` from `langgraph.prebuilt`；`create_react_agent`
- Part 1：定义工具 → `ToolNode` 自动处理工具调用/结果
- Part 2：手动搭建 ReAct 图（LLM节点 → 工具节点 → 条件边判断是否继续）
- Part 3：`create_react_agent` 一行创建完整 Agent（是 Part 2 的封装）

```python
from langgraph.prebuilt import ToolNode, create_react_agent

# 方式1：手动
tool_node = ToolNode(tools)
# 方式2：一行
agent = create_react_agent(llm, tools)
```

- [ ] **Step 1: 写文件**
- [ ] **Step 2: 验证运行**
- [ ] **Step 3: Commit** `feat(lg-advanced): 20 ToolNode and create_react_agent`

---

**Task 21: `21_streaming.py`**

- Create: `langchain_langgraph/04_lg_advanced/21_streaming.py`
- Part 1：`graph.stream(input, config, stream_mode="updates")` 逐步骤查看每个节点输出
- Part 2：`graph.stream(input, config, stream_mode="values")` 查看每步后的完整 State
- Part 3：`graph.astream_events(input, config)` 异步流，追踪模型 token 级别的输出

```python
# 逐节点追踪
for step in graph.stream(input, config, stream_mode="updates"):
    node_name, node_output = list(step.items())[0]
    print(f"[{node_name}] {node_output}")
```

- [ ] **Step 1: 写文件**
- [ ] **Step 2: 验证运行**
- [ ] **Step 3: Commit** `feat(lg-advanced): 21 streaming modes`

---

**Task 22: `22_supervisor_pattern.py`**

- Create: `langchain_langgraph/04_lg_advanced/22_supervisor_pattern.py`
- 核心：Supervisor Agent 调用工具（工具名 = 子 Agent 名），决定下一步
- Part 1：定义 2 个专职 Agent（搜索员、写作者），包装成工具
- Part 2：Supervisor LLM 决策：调用哪个子 Agent，还是直接结束
- Part 3：完整运行，展示 Supervisor 的调度日志

```python
# Supervisor 的工具 = 调用子 Agent
members = ["researcher", "writer"]
supervisor_tools = [
    {"name": agent, "description": f"调用 {agent}"}
    for agent in members
]

def supervisor_node(state):
    # Supervisor 决定下一步调用哪个成员
    response = llm.bind_tools(supervisor_tools).invoke(state["messages"])
    return {"next": response.tool_calls[0]["name"] if response.tool_calls else END}
```

- [ ] **Step 1: 写文件**
- [ ] **Step 2: 验证运行**
- [ ] **Step 3: Commit** `feat(lg-advanced): 22 supervisor multi-agent pattern`

---

**Task 23: `23_network_pattern.py`**

- Create: `langchain_langgraph/04_lg_advanced/23_network_pattern.py`
- 核心：`Command(goto="agent_name")` + `Command(goto=END)` 实现去中心化路由
- Part 1：每个 Agent 自己决定把控制权交给谁（Handoff）
- Part 2：实现两个 Agent 互相协作（A→B→A→END）
- Part 3：对比 Supervisor vs Network 的适用场景

```python
from langgraph.types import Command

def agent_a(state):
    # ...处理...
    # 把控制权交给 agent_b
    return Command(goto="agent_b", update={"messages": [response]})

def agent_b(state):
    # ...处理...
    # 结束
    return Command(goto=END, update={"messages": [response]})
```

- [ ] **Step 1: 写文件**
- [ ] **Step 2: 验证运行**
- [ ] **Step 3: Commit** `feat(lg-advanced): 23 network multi-agent pattern`

---

## Module 5: 综合项目

### Task 24: `24_research_team.py`

**Files:**
- Create: `langchain_langgraph/05_multi_agent/24_research_team.py`

- [ ] **Step 1: 写文件**

```python
"""
项目：研究团队 —— 多 Agent 协作完成深度研究报告

角色：
  - 搜索员（Researcher）：接收研究主题，用工具搜集信息，整理成条目
  - 分析师（Analyst）   ：接收搜集的信息，提炼关键洞察，标出重点
  - 写作者（Writer）    ：接收洞察，撰写完整报告（标题+正文+结论）
  - 协调员（Supervisor）：决定调用顺序和是否完成

工作流：
  用户输入主题
  → Supervisor 决定先调用 Researcher
  → Researcher 搜集信息后 handoff 给 Analyst
  → Analyst 分析后 handoff 给 Writer  
  → Writer 完成报告后 Supervisor 宣告结束

架构：Supervisor 模式（22_supervisor_pattern 的实战版本）
"""
# ... 完整实现（约 200 行）
```

- [ ] **Step 2: 验证运行**（输入"量子计算现状"，应输出完整报告）
- [ ] **Step 3: Commit** `feat(multi-agent): 24 research team`

---

### Task 25: `25_code_review_team.py`

**Files:**
- Create: `langchain_langgraph/05_multi_agent/25_code_review_team.py`

- [ ] **Step 1: 写文件**（三个 Agent：开发者写代码 → 测试员写测试 → 审核者评审 → 循环直到通过）
- [ ] **Step 2: 验证运行**（输入"写一个Python二分搜索函数"，应经过多轮后输出完整代码+测试）
- [ ] **Step 3: Commit** `feat(multi-agent): 25 code review team`

---

### Task 26: `26_production_patterns.py`

**Files:**
- Create: `langchain_langgraph/05_multi_agent/26_production_patterns.py`

- [ ] **Step 1: 写文件**

```python
"""
项目：生产级模式 —— 让多 Agent 系统在真实环境中可靠运行

涵盖：
  1. 错误恢复：节点失败后重试（RetryPolicy），或路由到错误处理节点
  2. 超时控制：节点执行超时后退出循环
  3. 结构化日志：每个节点的输入/输出/耗时写入日志
  4. LangSmith 追踪：一行配置开启全链路追踪（需要 LANGCHAIN_API_KEY）
"""
from langgraph.types import RetryPolicy

# RetryPolicy 示例
graph.add_node(
    "flaky_node",
    flaky_function,
    retry=RetryPolicy(max_attempts=3, backoff_factor=2.0)
)
```

- [ ] **Step 2: 验证运行**
- [ ] **Step 3: Commit** `feat(multi-agent): 26 production patterns`

---

## 收尾

### Task 27: 最终提交

- [ ] **Step 1: 推送到 GitHub**

```bash
git push origin master
```

- [ ] **Step 2: 验证所有 demo 可运行**

```bash
# 快速验证（逐一运行每个 demo）
cd langchain_langgraph
for f in 01_lc_basics/*.py 02_lc_advanced/*.py 03_lg_basics/*.py 04_lg_advanced/*.py 05_multi_agent/*.py; do
    echo "=== $f ===" && python "$f" && echo "PASS" || echo "FAIL"
done
```
