# Claude API 学习项目

这是一个全面的 Claude API 学习项目，从 API 基础到高级功能，再到 MCP（Model Context Protocol）和实际应用项目，适合想要系统学习和掌握 Claude API 的开发者。

## 学习路径

本项目分为四个循序渐进的模块：

### 01_basics 基础必学（API 核心）
掌握 Claude API 的基本使用方法和核心概念
- 01_hello_claude.py - 第一次 API 调用，学会初始化客户端和提取响应
- 02_messages_format.py - 理解消息格式，掌握 user / assistant / system 三种角色
- 03_parameters.py - 学会调优模型参数（temperature、top_p、stop sequences 等）
- 04_streaming.py - 实现流式输出，获得实时反馈体验
- 05_error_handling.py - 处理 API 超时、限流、认证失败等常见错误

### 02_advanced 进阶必学（高级特性）
学习 Claude API 的高级功能，应对复杂场景
- 06_prompt_engineering.py - 掌握有效的提示词编写技巧，提高输出质量
- 07_tool_use.py - 教会 Claude 调用外部工具（函数调用）
- 08_vision.py - 处理多模态输入，让 Claude 分析图像和文档
- 09_conversation.py - 管理多轮对话，保持上下文连贯性
- 10_structured_output.py - 使用 JSON Schema 获得结构化输出

### 03_mcp_skill_command MCP 与扩展（高级特性）
深入理解 MCP 协议和 Claude Code 的扩展机制
- 11_mcp_intro.py - 了解 MCP 协议的基本概念
- 12_mcp_custom_server.py - 开发自定义 MCP Server
- 12_mcp_server_demo.py - MCP Server 的完整实现示例
- 13_skill_usage.py - 学会在 Claude Code 中使用 Skill
- 14_slash_command.py - 创建自定义 Slash Command
- 15_mcp_tool_integration.py - 整合 MCP 工具与模型能力

### 04_project 综合项目（实战应用）
- cli_assistant.py - 完整的命令行 AI 助手，整合前三个模块的所有知识

## 环境配置

### 前置要求
- Python 3.8 或更高版本
- Anthropic API Key（获取地址：https://console.anthropic.com/）

### 安装步骤

#### 1. 克隆仓库
```bash
git clone <repository-url>
cd claude_study
```

#### 2. 创建虚拟环境
```bash
python -m venv .venv
```

#### 3. 激活虚拟环境

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

#### 4. 安装依赖
```bash
pip install -r requirements.txt
```

#### 5. 配置 API Key
```bash
cp .env.example .env
```

使用编辑器打开 `.env` 文件，将 `your_api_key_here` 替换为你的 Anthropic API Key：
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx
```

不要提交 `.env` 文件到版本控制中（已添加到 .gitignore）。

## Demo 列表

按照下表顺序运行示例，逐步掌握 Claude API：

| # | 文件 | 主题 | 运行命令 |
|---|------|------|----------|
| 01 | 01_basics/01_hello_claude.py | 第一次 API 调用 | `python 01_basics/01_hello_claude.py` |
| 02 | 01_basics/02_messages_format.py | 消息格式详解 | `python 01_basics/02_messages_format.py` |
| 03 | 01_basics/03_parameters.py | 模型参数调优 | `python 01_basics/03_parameters.py` |
| 04 | 01_basics/04_streaming.py | 流式输出 | `python 01_basics/04_streaming.py` |
| 05 | 01_basics/05_error_handling.py | 错误处理 | `python 01_basics/05_error_handling.py` |
| 06 | 02_advanced/06_prompt_engineering.py | Prompt 工程 | `python 02_advanced/06_prompt_engineering.py` |
| 07 | 02_advanced/07_tool_use.py | 工具调用 | `python 02_advanced/07_tool_use.py` |
| 08 | 02_advanced/08_vision.py | 多模态/图像处理 | `python 02_advanced/08_vision.py` |
| 09 | 02_advanced/09_conversation.py | 对话管理 | `python 02_advanced/09_conversation.py` |
| 10 | 02_advanced/10_structured_output.py | 结构化输出 | `python 02_advanced/10_structured_output.py` |
| 11 | 03_mcp_skill_command/11_mcp_intro.py | MCP 入门 | `python 03_mcp_skill_command/11_mcp_intro.py` |
| 12 | 03_mcp_skill_command/12_mcp_custom_server.py | 自定义 MCP Server | `python 03_mcp_skill_command/12_mcp_custom_server.py` |
| 13 | 03_mcp_skill_command/12_mcp_server_demo.py | MCP Server 实现 | `python 03_mcp_skill_command/12_mcp_server_demo.py` |
| 14 | 03_mcp_skill_command/13_skill_usage.py | Skill 使用 | `python 03_mcp_skill_command/13_skill_usage.py` |
| 15 | 03_mcp_skill_command/14_slash_command.py | Slash Command | `python 03_mcp_skill_command/14_slash_command.py` |
| 16 | 03_mcp_skill_command/15_mcp_tool_integration.py | MCP 工具集成 | `python 03_mcp_skill_command/15_mcp_tool_integration.py` |

## 综合项目：CLI 智能助手

位置：`04_project/cli_assistant.py`

这是一个完整的命令行 AI 助手，演示了如何将前三个模块的技能整合在一起：

**运行方式：**
```bash
python 04_project/cli_assistant.py
```

**功能特点：**
- 交互式多轮对话
- 支持流式输出
- 智能上下文管理
- 工具调用集成
- 结构化响应处理

**使用示例：**
```
> 请帮我分析一段代码
> 如何用 Python 实现二分查找？
> 给我讲讲 RESTful API 的设计原则
> 输入 'quit' 或 'exit' 退出程序
```

## 前置知识

### 必需的 Python 基础
- 函数定义和调用
- 类和对象（面向对象编程）
- 异常处理（try/except）
- 字典和列表操作
- 理解 API 和 HTTP 请求的基本概念

### 推荐的补充知识
- JSON 格式
- 环境变量管理
- 虚拟环境使用

## 项目文件依赖

```
requirements.txt
  - anthropic>=0.50.0      # Claude API 官方 SDK
  - fastmcp>=0.4.0         # MCP Protocol 实现
  - pydantic>=2.0.0        # 数据验证和结构化输出
  - python-dotenv>=1.0.0   # 环境变量管理
  - httpx>=0.27.0          # 高级 HTTP 客户端
```

## 常见问题

### Q: 如何获取 API Key？
A: 访问 https://console.anthropic.com/，注册账户并在 API Keys 部分生成新的 Key。

### Q: 运行示例时出现 "API rate limit exceeded" 错误？
A: 这说明 API 调用频率过高。在代码中添加延迟或升级你的 API 账户。

### Q: 如何知道某个示例的输出应该是什么样的？
A: 每个示例文件的开头都有详细的注释说明，描述了学习目标和预期行为。

### Q: 可以修改示例代码吗？
A: 完全可以！建议复制示例到新文件中，进行实验性修改，这是最好的学习方式。

## 许可证

MIT License

---

**最后更新：** 2024-04-02
