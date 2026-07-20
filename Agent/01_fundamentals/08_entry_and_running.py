"""
==============================================================================
第八课：入口文件与运行方式
==============================================================================

【学习目标】
- 掌握 CLI（命令行）入口实现
- 掌握 API 服务入口实现（FastAPI）
- 理解异步运行模式
- 学会优雅关闭和资源清理

【核心概念】
- CLI 交互：命令行交互式对话
- API 服务：HTTP 接口服务
- 异步编程：asyncio 并发处理
- 优雅关闭：正确处理资源释放

【前置知识】
- 第七课：Agents 模块

==============================================================================
"""

import asyncio
import json
import signal
import sys
from datetime import datetime
from typing import Optional

# ============================================================================
# 简化版依赖（便于独立运行）
# ============================================================================

from enum import Enum
from pydantic import BaseModel, Field


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: Role
    content: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# 第一部分：为什么需要不同的运行方式？
# ============================================================================

def why_multiple_entry_points():
    """
    解释为什么 Agent 需要多种运行方式

    不同的使用场景需要不同的入口：
    1. CLI：开发调试、个人使用、脚本自动化
    2. API 服务：Web 应用、移动应用、第三方集成
    3. 定时任务：批量处理、数据同步
    4. 事件驱动：消息队列、Webhook
    """

    scenarios = [
        {
            "mode": "CLI（命令行）",
            "scenario": "开发者调试、个人助手",
            "example": "python main.py chat",
            "pros": ["简单直接", "易于调试", "无需额外服务"],
            "cons": ["只能单用户", "无法远程访问"],
        },
        {
            "mode": "API 服务（HTTP）",
            "scenario": "Web 应用后端、移动应用",
            "example": "python main.py serve --port 8000",
            "pros": ["多用户并发", "远程访问", "易于集成"],
            "cons": ["需要服务器", "复杂度更高"],
        },
        {
            "mode": "定时任务（Cron）",
            "scenario": "批量处理、定期报告",
            "example": "python main.py batch --config daily.json",
            "pros": ["自动化", "可调度"],
            "cons": ["非实时", "无交互"],
        },
        {
            "mode": "事件驱动（Queue）",
            "scenario": "消息处理、异步任务",
            "example": "python main.py worker --queue tasks",
            "pros": ["高吞吐", "解耦"],
            "cons": ["复杂度高", "需要消息队列"],
        },
    ]

    print("=" * 60)
    print("不同运行方式对比")
    print("=" * 60)
    for s in scenarios:
        print(f"\n {s['mode']}")
        print(f"   场景: {s['scenario']}")
        print(f"   示例: {s['example']}")
        print(f"   优点: {', '.join(s['pros'])}")
        print(f"   缺点: {', '.join(s['cons'])}")
    print()


# ============================================================================
# 第二部分：CLI 入口实现（cli.py）
# ============================================================================
#
# CLI 入口提供命令行交互式对话。
# 特点：
# - 实时交互
# - 支持命令（/help, /clear, /exit）
# - 彩色输出
# - 历史记录


class CLIAgent:
    """
    CLI 交互式 Agent

    提供终端交互界面，用户可以输入问题，Agent 实时回复。
    支持特殊命令：
    - /help: 显示帮助
    - /clear: 清空对话历史
    - /history: 显示对话历史
    - /exit: 退出程序
    """

    def __init__(self, agent_name: str = "AI助手"):
        self.agent_name = agent_name
        self.history: list[dict] = []
        self.running = True

    def print_welcome(self):
        """打印欢迎信息"""
        print(f"\n{'='*50}")
        print(f"  {self.agent_name} - CLI 交互模式")
        print(f"{'='*50}")
        print(f"  输入问题开始对话")
        print(f"  输入 /help 查看可用命令")
        print(f"  输入 /exit 退出程序")
        print(f"{'='*50}\n")

    def print_help(self):
        """打印帮助信息"""
        print(f"\n── 可用命令 ──")
        print(f"  /help     - 显示帮助信息")
        print(f"  /clear    - 清空对话历史")
        print(f"  /history  - 显示对话历史")
        print(f"  /stats    - 显示统计信息")
        print(f"  /exit     - 退出程序")
        print()

    def process_command(self, command: str) -> bool:
        """
        处理特殊命令

        Returns:
            True 如果命令已处理，False 如果是普通输入
        """
        cmd = command.strip().lower()

        if cmd == "/help":
            self.print_help()
            return True
        elif cmd == "/clear":
            self.history.clear()
            print("  ✅ 对话历史已清空\n")
            return True
        elif cmd == "/history":
            self.show_history()
            return True
        elif cmd == "/stats":
            self.show_stats()
            return True
        elif cmd in ("/exit", "/quit", "/q"):
            print(f"\n  再见！感谢使用 {self.agent_name}。\n")
            self.running = False
            return True

        return False

    def show_history(self):
        """显示对话历史"""
        if not self.history:
            print("  （暂无对话历史）\n")
            return

        print(f"\n── 对话历史 ({len(self.history)} 条) ──")
        for i, msg in enumerate(self.history, 1):
            role = msg["role"]
            content = msg["content"][:50]
            print(f"  [{i}] {role}: {content}")
        print()

    def show_stats(self):
        """显示统计信息"""
        user_msgs = sum(1 for m in self.history if m["role"] == "user")
        ai_msgs = sum(1 for m in self.history if m["role"] == "assistant")
        print(f"\n── 统计信息 ──")
        print(f"  总消息数: {len(self.history)}")
        print(f"  用户消息: {user_msgs}")
        print(f"  AI 回复:  {ai_msgs}")
        print()

    def simulate_response(self, user_input: str) -> str:
        """
        模拟 AI 回复

        实际项目中，这里会调用 Agent 处理用户输入。
        """
        responses = {
            "你好": f"你好！我是 {self.agent_name}，很高兴为你服务。",
            "帮助": "我可以回答你的问题，也可以执行一些工具操作。",
            "谢谢": "不客气！还有其他问题吗？",
        }

        for key, response in responses.items():
            if key in user_input:
                return response

        return f"收到你的问题：「{user_input[:30]}」。这是一个模拟回复。"

    def run(self):
        """
        运行 CLI 交互循环

        主循环：
        1. 读取用户输入
        2. 检查是否是命令
        3. 如果是普通输入，调用 Agent 处理
        4. 显示回复
        5. 重复
        """
        self.print_welcome()

        while self.running:
            try:
                # 读取用户输入
                user_input = input("你: ").strip()

                if not user_input:
                    continue

                # 检查是否是命令
                if user_input.startswith("/"):
                    self.process_command(user_input)
                    continue

                # 处理普通输入
                print(f"\n{self.agent_name}: ", end="", flush=True)
                response = self.simulate_response(user_input)

                # 模拟打字效果
                for char in response:
                    print(char, end="", flush=True)
                    # 实际项目中不需要这个延迟
                print("\n")

                # 记录历史
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": response})

            except KeyboardInterrupt:
                # Ctrl+C 退出
                print(f"\n\n  再见！\n")
                self.running = False
            except EOFError:
                # Ctrl+D 退出
                print(f"\n\n  再见！\n")
                self.running = False


# ============================================================================
# 第三部分：API 服务入口实现（api.py）
# ============================================================================
#
# API 服务入口提供 HTTP 接口。
# 使用 FastAPI 框架（需要安装：pip install fastapi uvicorn）
#
# API 端点：
# - POST /chat: 发送消息并获取回复
# - GET /sessions: 获取会话列表
# - GET /health: 健康检查


# 注意：以下代码需要 fastapi 和 uvicorn 才能运行
# 这里展示代码结构和设计，实际运行需要安装依赖

API_CODE_EXAMPLE = '''
# api.py - FastAPI 服务入口
# 运行方式: uvicorn api:app --reload --port 8000

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

app = FastAPI(title="Agent API", version="1.0.0")

# ── 请求/响应模型 ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    session_id: Optional[str] = None
    user_id: str = "default"

class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    session_id: str
    is_final: bool = True

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    sessions: int

# ─ 全局状态 ───────────────────────────────────────────────────

sessions = {}  # 会话存储（生产环境应该用 Redis）

# ── API 端点 ───────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        sessions=len(sessions),
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天端点

    发送消息并获取 AI 回复。
    支持会话管理：同一 session_id 的消息会保持上下文。
    """
    # 获取或创建会话
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = {
            "user_id": request.user_id,
            "messages": [],
            "created_at": datetime.now().isoformat(),
        }

    session = sessions[session_id]

    # 添加用户消息
    session["messages"].append({
        "role": "user",
        "content": request.message,
    })

    # 调用 Agent 处理（实际项目中）
    # response = await agent.run(request.message, session)
    response = f"收到: {request.message}"

    # 添加 AI 回复
    session["messages"].append({
        "role": "assistant",
        "content": response,
    })

    return ChatResponse(
        response=response,
        session_id=session_id,
        is_final=True,
    )

@app.get("/sessions")
async def list_sessions():
    """获取所有会话"""
    return {
        "sessions": [
            {"id": sid, "messages": len(s["messages"])}
            for sid, s in sessions.items()
        ]
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")
'''


def demonstrate_api_design():
    """演示 API 设计"""
    print("=" * 60)
    print("API 服务设计")
    print("=" * 60)
    print(API_CODE_EXAMPLE)


# ============================================================================
# 第四部分：异步运行模式
# ============================================================================

async def demonstrate_async_patterns():
    """
    演示异步运行模式

    异步编程让 Agent 可以：
    1. 同时处理多个请求
    2. 非阻塞地等待 LLM 响应
    3. 并发执行多个工具调用
    """

    print("=" * 60)
    print("异步运行模式演示")
    print("=" * 60)

    # ─ 模式1：顺序执行 ─────────────────────────────────────────
    print("\n── 模式1：顺序执行 ──")

    async def process_request(request_id: str, delay: float):
        """模拟处理请求"""
        print(f"  开始处理请求 {request_id}...")
        await asyncio.sleep(delay)  # 模拟 LLM 调用
        print(f"  完成请求 {request_id}")
        return f"结果_{request_id}"

    start = asyncio.get_event_loop().time()

    # 顺序执行（总时间 = 各任务时间之和）
    result1 = await process_request("A", 0.5)
    result2 = await process_request("B", 0.5)
    result3 = await process_request("C", 0.5)

    elapsed = asyncio.get_event_loop().time() - start
    print(f"  顺序执行总时间: {elapsed:.1f}秒")
    print(f"  结果: {result1}, {result2}, {result3}")

    # ── 模式2：并发执行 ────────────────────────────────────────
    print("\n── 模式2：并发执行 ──")

    start = asyncio.get_event_loop().time()

    # 并发执行（总时间 ≈ 最慢任务的时间）
    results = await asyncio.gather(
        process_request("X", 0.5),
        process_request("Y", 0.5),
        process_request("Z", 0.5),
    )

    elapsed = asyncio.get_event_loop().time() - start
    print(f"  并发执行总时间: {elapsed:.1f}秒")
    print(f"  结果: {results}")

    # ── 模式3：带超时的执行 ─────────────────────────────────────
    print("\n── 模式3：带超时 ─")

    try:
        result = await asyncio.wait_for(
            process_request("slow", 2.0),
            timeout=1.0,  # 1秒超时
        )
        print(f"  结果: {result}")
    except asyncio.TimeoutError:
        print(f"  请求超时！")

    print()


# ============================================================================
# 第五部分：优雅关闭与资源清理
# ============================================================================

class GracefulShutdown:
    """
    优雅关闭处理器

    确保程序退出时：
    1. 完成正在处理的请求
    2. 保存未保存的数据
    3. 释放资源（连接、文件等）
    4. 记录关闭日志
    """

    def __init__(self):
        self.is_shutting_down = False
        self.active_requests = 0

    def request_start(self):
        """请求开始"""
        self.active_requests += 1

    def request_end(self):
        """请求结束"""
        self.active_requests -= 1

    def shutdown_signal(self):
        """收到关闭信号"""
        print("\n  ️ 收到关闭信号，正在优雅关闭...")
        self.is_shutting_down = True

    def wait_for_completion(self, timeout: float = 5.0):
        """等待活跃请求完成"""
        if self.active_requests == 0:
            print("  ✅ 没有活跃请求，可以安全关闭")
            return True

        print(f"  等待 {self.active_requests} 个活跃请求完成...")
        # 实际实现会用 asyncio.wait_for 等待
        return self.active_requests == 0

    def cleanup(self):
        """清理资源"""
        print("  清理资源...")
        print("  - 保存会话数据")
        print("  - 关闭数据库连接")
        print("  - 释放内存")
        print("  ✅ 清理完成")


def demonstrate_graceful_shutdown():
    """演示优雅关闭"""
    print("=" * 60)
    print("优雅关闭演示")
    print("=" * 60)

    handler = GracefulShutdown()

    # 模拟请求处理
    print("\n── 正常处理请求 ──")
    handler.request_start()
    handler.request_start()
    print(f"  活跃请求: {handler.active_requests}")

    handler.request_end()
    handler.request_end()
    print(f"  活跃请求: {handler.active_requests}")

    # 模拟关闭
    print("\n── 收到关闭信号 ──")
    handler.shutdown_signal()
    handler.wait_for_completion()
    handler.cleanup()
    print()


# ============================================================================
# 第六部分：综合演示 - CLI 交互
# ============================================================================

def demonstrate_cli_interaction():
    """演示 CLI 交互（非交互式，自动演示）"""
    print("=" * 60)
    print("CLI 交互演示（自动模式）")
    print("=" * 60)

    cli = CLIAgent(agent_name="学习助手")

    # 模拟用户输入
    demo_inputs = [
        "/help",
        "你好",
        "什么是 Agent？",
        "/stats",
        "/history",
        "谢谢",
        "/exit",
    ]

    print(f"\n  模拟用户输入序列:\n")
    for user_input in demo_inputs:
        print(f"  你: {user_input}")

        if user_input.startswith("/"):
            cli.process_command(user_input)
            if not cli.running:
                break
        else:
            response = cli.simulate_response(user_input)
            print(f"  学习助手: {response}\n")
            cli.history.append({"role": "user", "content": user_input})
            cli.history.append({"role": "assistant", "content": response})
    print()


# ============================================================================
# 第七部分：进阶 - 不同运行方式的配置
# ============================================================================

def advanced_running_configurations():
    """
    进阶：不同运行方式的配置示例
    """

    print("=" * 60)
    print("进阶：运行配置示例")
    print("=" * 60)

    configurations = {
        "开发环境": {
            "mode": "cli",
            "log_level": "DEBUG",
            "mock_llm": True,
            "description": "使用 Mock LLM，详细日志",
        },
        "测试环境": {
            "mode": "api",
            "port": 8000,
            "log_level": "INFO",
            "mock_llm": True,
            "description": "API 服务，Mock LLM",
        },
        "生产环境": {
            "mode": "api",
            "port": 8000,
            "workers": 4,
            "log_level": "WARNING",
            "mock_llm": False,
            "redis_url": "redis://localhost:6379",
            "description": "多 worker，真实 LLM，Redis 会话",
        },
        "批处理": {
            "mode": "batch",
            "input_file": "tasks.json",
            "output_file": "results.json",
            "concurrency": 10,
            "description": "批量处理任务文件",
        },
    }

    for env, config in configurations.items():
        print(f"\n🔷 {env}")
        for key, value in config.items():
            if key != "description":
                print(f"   {key}: {value}")
        print(f"   说明: {config['description']}")
    print()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("第八课：入口文件与运行方式")
    print("🚀" * 30 + "\n")

    # 1. 为什么需要多种运行方式
    why_multiple_entry_points()

    # 2. CLI 演示
    demonstrate_cli_interaction()

    # 3. API 设计
    demonstrate_api_design()

    # 4. 异步模式
    asyncio.run(demonstrate_async_patterns())

    # 5. 优雅关闭
    demonstrate_graceful_shutdown()

    # 6. 进阶配置
    advanced_running_configurations()

    print("=" * 60)
    print("✅ 第八课学习完成！")
    print("=" * 60)
    print("""
 本课要点总结：
1. CLI 入口：适合开发调试和个人使用
2. API 服务：适合多用户、远程访问、第三方集成
3. 异步编程：并发处理多个请求，提高效率
4. 优雅关闭：确保资源正确释放
5. 不同环境需要不同的运行配置

 下一课：测试驱动开发 - 我们将学习如何为 Agent 编写测试
    """)
