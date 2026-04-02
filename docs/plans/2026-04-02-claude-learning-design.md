# Claude 使用方法学习课程 — 设计文档

**日期**: 2026-04-02  
**目标用户**: 有 Python 基础、未使用过 Claude/LLM API、目标成为 AI 应用开发者

---

## 课程目标

系统学习 Claude API 的初级到高级使用方法，最终能够独立构建 AI 应用。

---

## 课程结构

### 第一层：基础层 `01_basics/`

| 文件 | 主题 | 核心概念 |
|------|------|----------|
| 01_hello_claude.py | 第一次 API 调用 | SDK 初始化、API Key、最简调用 |
| 02_messages_format.py | 消息格式详解 | role/content 结构、system prompt |
| 03_parameters.py | 模型参数调优 | temperature、max_tokens、top_p |
| 04_streaming.py | 流式输出 | stream=True、逐 token 处理 |
| 05_error_handling.py | 错误处理 | 异常类型、重试策略、rate limit |

### 第二层：进阶层 `02_advanced/`

| 文件 | 主题 | 核心概念 |
|------|------|----------|
| 06_prompt_engineering.py | Prompt 工程技巧 | few-shot、chain-of-thought、角色扮演 |
| 07_tool_use.py | 工具调用 | tools 定义、tool_use/tool_result 循环 |
| 08_vision.py | 图像理解 | base64/url 图片、多模态消息 |
| 09_conversation.py | 多轮对话管理 | 历史消息维护、上下文窗口管理 |
| 10_structured_output.py | 结构化输出 | JSON mode、Pydantic 校验 |

### 第三层：MCP/Skill/Command `03_mcp_skill_command/`

| 文件 | 主题 | 核心概念 |
|------|------|----------|
| 11_mcp_intro.py | MCP 概念与连接 | MCP 协议、连接本地 MCP Server |
| 12_mcp_custom_server.py | 自定义 MCP Server | FastMCP、tool 注册、server 启动 |
| 13_skill_usage.py | Skill 调用模式 | invoke skill pattern、skill 生命周期 |
| 14_slash_command.py | 自定义 slash command | command 定义、参数解析、响应格式 |
| 15_mcp_tool_integration.py | MCP + Tool Use 联动 | Claude 通过 MCP 调用外部工具的完整链路 |

### 综合项目：`04_project/cli_assistant.py`

整合所有知识点的 CLI 智能助手：
- 多轮对话
- Tool Use（查天气、计算器、文件读写）
- MCP 工具集成
- 流式输出
- 错误处理与优雅降级

---

## 技术栈

- **SDK**: `anthropic` (官方 Python SDK)
- **MCP**: `fastmcp` (MCP Server 开发)
- **结构化输出**: `pydantic`
- **环境变量**: `python-dotenv`
- **模型**: `claude-sonnet-4-6`（最新 Sonnet）

---

## 文件规范

每个 demo 文件包含：
1. 文件顶部：主题说明、学习目标、前置知识
2. 代码中：「是什么、为什么、怎么用」三层注释
3. 文件底部：`if __name__ == "__main__"` 可直接运行的示例
