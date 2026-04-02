# Claude 学习课程 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建一套完整的 Claude API 学习 demo，涵盖基础层、进阶层、MCP/Skill/Command 层和综合项目

**Architecture:** 独立 demo 文件 + 综合项目，每个文件可独立运行，包含详细中文注释。使用官方 anthropic SDK，MCP 层用 fastmcp，结构化输出用 pydantic。

**Tech Stack:** Python 3.14, anthropic SDK, fastmcp, pydantic, python-dotenv, claude-sonnet-4-6

---

## Task 0: 项目基础设施

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`

**Step 1: 创建 requirements.txt**

```
anthropic>=0.50.0
fastmcp>=0.4.0
pydantic>=2.0.0
python-dotenv>=1.0.0
httpx>=0.27.0
```

**Step 2: 创建 .env.example**

```
ANTHROPIC_API_KEY=your_api_key_here
```

**Step 3: 创建 .gitignore**

```
.env
.venv/
__pycache__/
*.pyc
.idea/
```

**Step 4: 安装依赖**

```bash
.venv/Scripts/pip.exe install -r requirements.txt
```

**Step 5: 创建 .env（用户自行填写 API Key）**

```bash
cp .env.example .env
# 然后编辑 .env 填入真实的 ANTHROPIC_API_KEY
```

**Step 6: Commit**

```bash
git add requirements.txt .env.example .gitignore
git commit -m "chore: add project infrastructure"
```

---

## Task 1: 01_basics/01_hello_claude.py

**Files:**
- Create: `01_basics/01_hello_claude.py`

**内容要点:**
- 加载 .env，初始化 anthropic.Anthropic() 客户端
- 调用 client.messages.create()，model/max_tokens/messages 三个核心参数
- 解析响应对象：response.content[0].text
- 打印使用量：response.usage

**注释层次:** SDK 是什么、API Key 从哪来、响应结构长什么样

**Step 1: 写文件**（见完整代码规范，下方执行时实现）

**Step 2: 运行验证**

```bash
.venv/Scripts/python.exe 01_basics/01_hello_claude.py
```
Expected: 打印 Claude 的回复文本 + token 使用量

**Step 3: Commit**

```bash
git add 01_basics/01_hello_claude.py
git commit -m "feat: add hello claude demo"
```

---

## Task 2: 01_basics/02_messages_format.py

**Files:**
- Create: `01_basics/02_messages_format.py`

**内容要点:**
- system prompt 的作用与写法
- user/assistant 角色的 messages 列表结构
- 多条消息组合（模拟对话上下文）
- content 的两种形式：字符串 vs 列表

**Step 1: 写文件并运行验证**

```bash
.venv/Scripts/python.exe 01_basics/02_messages_format.py
```

**Step 2: Commit**

```bash
git add 01_basics/02_messages_format.py
git commit -m "feat: add messages format demo"
```

---

## Task 3: 01_basics/03_parameters.py

**Files:**
- Create: `01_basics/03_parameters.py`

**内容要点:**
- temperature（0=确定性，1=创造性）对比实验
- max_tokens 控制输出长度
- top_p 核采样说明
- 同一 prompt 不同参数的输出对比

**Step 1: 写文件并运行验证**

```bash
.venv/Scripts/python.exe 01_basics/03_parameters.py
```

**Step 2: Commit**

```bash
git add 01_basics/03_parameters.py
git commit -m "feat: add parameters tuning demo"
```

---

## Task 4: 01_basics/04_streaming.py

**Files:**
- Create: `01_basics/04_streaming.py`

**内容要点:**
- client.messages.stream() context manager 用法
- on_text / on_message 回调方式
- 直接迭代 stream 的方式
- 流式 vs 非流式的使用场景对比

**Step 1: 写文件并运行验证**

```bash
.venv/Scripts/python.exe 01_basics/04_streaming.py
```

**Step 2: Commit**

```bash
git add 01_basics/04_streaming.py
git commit -m "feat: add streaming demo"
```

---

## Task 5: 01_basics/05_error_handling.py

**Files:**
- Create: `01_basics/05_error_handling.py`

**内容要点:**
- anthropic.APIError 及子类（AuthenticationError, RateLimitError, APIStatusError）
- 指数退避重试模式
- 超时配置：httpx.Timeout
- 实际触发各类错误的示例

**Step 1: 写文件并运行验证**

```bash
.venv/Scripts/python.exe 01_basics/05_error_handling.py
```

**Step 2: Commit**

```bash
git add 01_basics/05_error_handling.py
git commit -m "feat: add error handling demo"
```

---

## Task 6: 02_advanced/06_prompt_engineering.py

**Files:**
- Create: `02_advanced/06_prompt_engineering.py`

**内容要点:**
- Zero-shot vs Few-shot 对比
- Chain-of-Thought（让模型逐步推理）
- XML 标签组织 prompt 结构
- 角色扮演 system prompt
- 负向提示（告诉模型不要做什么）

**Step 1: 写文件并运行验证**

```bash
.venv/Scripts/python.exe 02_advanced/06_prompt_engineering.py
```

**Step 2: Commit**

```bash
git add 02_advanced/06_prompt_engineering.py
git commit -m "feat: add prompt engineering demo"
```

---

## Task 7: 02_advanced/07_tool_use.py

**Files:**
- Create: `02_advanced/07_tool_use.py`

**内容要点:**
- tools 参数定义（name/description/input_schema）
- 识别 stop_reason == "tool_use"
- 解析 tool_use block，执行本地函数
- 构造 tool_result 消息回传
- 完整的 agentic loop（多轮工具调用）

**Step 1: 写文件并运行验证**

```bash
.venv/Scripts/python.exe 02_advanced/07_tool_use.py
```

**Step 2: Commit**

```bash
git add 02_advanced/07_tool_use.py
git commit -m "feat: add tool use demo"
```

---

## Task 8: 02_advanced/08_vision.py

**Files:**
- Create: `02_advanced/08_vision.py`
- Create: `02_advanced/sample.png`（程序自动生成测试图片）

**内容要点:**
- base64 编码本地图片
- URL 方式引用网络图片
- image content block 格式
- 文字+图片混合消息
- 图片描述、OCR、图表分析示例

**Step 1: 写文件并运行验证**

```bash
.venv/Scripts/python.exe 02_advanced/08_vision.py
```

**Step 2: Commit**

```bash
git add 02_advanced/08_vision.py
git commit -m "feat: add vision/multimodal demo"
```

---

## Task 9: 02_advanced/09_conversation.py

**Files:**
- Create: `02_advanced/09_conversation.py`

**内容要点:**
- ConversationManager 类封装历史消息
- 上下文窗口大小管理（滑动窗口 / 摘要压缩）
- 对话历史持久化（JSON 文件读写）
- 交互式终端对话循环

**Step 1: 写文件并运行验证**

```bash
.venv/Scripts/python.exe 02_advanced/09_conversation.py
```

**Step 2: Commit**

```bash
git add 02_advanced/09_conversation.py
git commit -m "feat: add conversation management demo"
```

---

## Task 10: 02_advanced/10_structured_output.py

**Files:**
- Create: `02_advanced/10_structured_output.py`

**内容要点:**
- 让 Claude 输出 JSON 的 prompt 技巧
- pydantic BaseModel 定义期望结构
- 解析并校验 Claude 返回的 JSON
- 嵌套结构、列表字段的处理
- 解析失败时的重试策略

**Step 1: 写文件并运行验证**

```bash
.venv/Scripts/python.exe 02_advanced/10_structured_output.py
```

**Step 2: Commit**

```bash
git add 02_advanced/10_structured_output.py
git commit -m "feat: add structured output demo"
```

---

## Task 11: 03_mcp_skill_command/11_mcp_intro.py

**Files:**
- Create: `03_mcp_skill_command/11_mcp_intro.py`

**内容要点:**
- MCP（Model Context Protocol）概念说明
- MCP Client vs MCP Server 架构图（注释中）
- 连接本地 MCP Server（stdio transport）
- 列出可用 tools/resources
- 通过 Claude + MCP 调用工具

**Step 1: 写文件并运行验证**

```bash
.venv/Scripts/python.exe 03_mcp_skill_command/11_mcp_intro.py
```

**Step 2: Commit**

```bash
git add 03_mcp_skill_command/11_mcp_intro.py
git commit -m "feat: add MCP intro demo"
```

---

## Task 12: 03_mcp_skill_command/12_mcp_custom_server.py

**Files:**
- Create: `03_mcp_skill_command/12_mcp_custom_server.py`

**内容要点:**
- 用 fastmcp 定义 MCP Server
- @mcp.tool() 装饰器注册工具
- @mcp.resource() 注册资源
- server.run() 启动 stdio server
- 如何在 Claude Desktop / Claude Code 中配置

**Step 1: 写文件并运行验证**

```bash
.venv/Scripts/python.exe 03_mcp_skill_command/12_mcp_custom_server.py
```

**Step 2: Commit**

```bash
git add 03_mcp_skill_command/12_mcp_custom_server.py
git commit -m "feat: add custom MCP server demo"
```

---

## Task 13: 03_mcp_skill_command/13_skill_usage.py

**Files:**
- Create: `03_mcp_skill_command/13_skill_usage.py`

**内容要点:**
- Claude Code Skill 的概念与工作原理
- Skill 的目录结构（.claude/skills/）
- 如何在 API 调用中模拟 skill invoke 模式
- system prompt 注入 skill 内容的实现
- 实战示例：代码审查 skill

**Step 1: 写文件并运行验证**

```bash
.venv/Scripts/python.exe 03_mcp_skill_command/13_skill_usage.py
```

**Step 2: Commit**

```bash
git add 03_mcp_skill_command/13_skill_usage.py
git commit -m "feat: add skill usage demo"
```

---

## Task 14: 03_mcp_skill_command/14_slash_command.py

**Files:**
- Create: `03_mcp_skill_command/14_slash_command.py`
- Create: `03_mcp_skill_command/commands/review.md`
- Create: `03_mcp_skill_command/commands/summarize.md`

**内容要点:**
- slash command 的 Markdown 文件格式规范
- 参数占位符 $ARGUMENTS 的用法
- 在代码中模拟 slash command 解析和执行
- 自定义 /review、/summarize command 示例
- 如何放置到 .claude/commands/ 使其在 Claude Code 中生效

**Step 1: 写文件并运行验证**

```bash
.venv/Scripts/python.exe 03_mcp_skill_command/14_slash_command.py
```

**Step 2: Commit**

```bash
git add 03_mcp_skill_command/
git commit -m "feat: add slash command demo"
```

---

## Task 15: 03_mcp_skill_command/15_mcp_tool_integration.py

**Files:**
- Create: `03_mcp_skill_command/15_mcp_tool_integration.py`

**内容要点:**
- 启动内嵌 MCP Server（subprocess）
- Claude 通过 MCP 发现并调用工具的完整链路
- tool_use 响应 → MCP 调用 → tool_result 回传
- 错误处理：MCP Server 宕机的降级策略
- 与 Task 7 的 tool_use 对比：直接 tools vs MCP tools

**Step 1: 写文件并运行验证**

```bash
.venv/Scripts/python.exe 03_mcp_skill_command/15_mcp_tool_integration.py
```

**Step 2: Commit**

```bash
git add 03_mcp_skill_command/15_mcp_tool_integration.py
git commit -m "feat: add MCP tool integration demo"
```

---

## Task 16: 04_project/cli_assistant.py（综合项目）

**Files:**
- Create: `04_project/cli_assistant.py`

**内容要点（整合所有前置知识）:**
- 多轮对话管理（Task 9）
- 流式输出（Task 4）
- Tool Use agentic loop（Task 7）：内置工具：计算器、天气查询（mock）、文件读写
- 结构化日志（Task 10）
- 错误处理与优雅退出（Task 5）
- slash command 风格的用户指令（/help /clear /save /load）
- 彩色终端输出（内置 ANSI 颜色，无需额外依赖）

**Step 1: 写文件并运行验证**

```bash
.venv/Scripts/python.exe 04_project/cli_assistant.py
```

**Step 2: Commit**

```bash
git add 04_project/cli_assistant.py
git commit -m "feat: add CLI assistant project"
```

---

## Task 17: README.md

**Files:**
- Create: `README.md`

**内容要点:**
- 课程概览与学习路径
- 环境配置步骤（安装、API Key 配置）
- 每个 demo 的一句话说明 + 运行命令
- 前置知识说明

**Step 1: 写文件**

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```
