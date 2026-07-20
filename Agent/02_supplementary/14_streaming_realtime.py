"""
==============================================================================
第十四课（补充）：Streaming 与实时输出
==============================================================================

【为什么需要单独一课？】
现有课程只简单提到流式输出，没有深入讲解。
生产级 Agent 必须支持实时流式输出，提升用户体验。

【学习目标】
- 理解流式输出的原理和优势
- 掌握 SSE (Server-Sent Events) 实现
- 掌握 WebSocket 实时通信
- 学会流式工具调用状态推送
- 理解前端如何消费流式数据

【核心概念】
- SSE (Server-Sent Events)
- WebSocket 双向通信
- 流式 Token 输出
- 工具调用状态推送
- 打字机效果

==============================================================================
"""

import asyncio
import json
import time
from datetime import datetime
from enum import Enum
from typing import Optional, AsyncIterator


# ============================================================================
# 第一部分：为什么需要流式输出？
# ============================================================================

def why_streaming_matters():
    """
    流式输出的重要性

    非流式（等待完整回复）：
    - 用户等待 5-10 秒才能看到第一个字
    - 体验差，用户可能以为卡住了

    流式（逐字/逐句输出）：
    - 用户几乎立即看到响应开始
    - 体验好，像真人对话
    - 可以实时显示工具调用状态
    """

    print("=" * 60)
    print("为什么需要流式输出")
    print("=" * 60)

    print("""
  ── 非流式体验 ──
  用户: 帮我写一篇文章
  [等待 8 秒...]
  AI:   [一次性显示完整文章]

  ── 流式体验 ──
  用户: 帮我写一篇文章
  AI:   好的，我来帮你写一篇文章...（立即开始）
        首先，我们需要确定文章的主题...
        然后，构建文章的大纲...
        [工具调用: 搜索最新资料...]
        根据最新资料，文章可以这样写...
    """)

    # ─ 流式输出的三种场景 ───────────────────────────────────────
    scenarios = [
        {
            "scenario": "Token 流式输出",
            "description": "LLM 生成文本时逐 token 返回",
            "latency": "首字 < 500ms",
        },
        {
            "scenario": "工具调用状态推送",
            "description": "推送工具调用的开始/进度/结果",
            "latency": "实时",
        },
        {
            "scenario": "Agent 思考过程展示",
            "description": "展示 Thought/Action/Observation 循环",
            "latency": "每轮循环实时",
        },
    ]

    print("\n── 三种流式场景 ─")
    for s in scenarios:
        print(f"  {s['scenario']}: {s['description']} (延迟: {s['latency']})")
    print()


# ============================================================================
# 第二部分：流式输出类型定义
# ============================================================================

class StreamEventType(Enum):
    """流式事件类型"""
    # Token 输出
    TOKEN = "token"               # 文本 token
    TOKEN_END = "token_end"       # token 输出结束

    # 工具调用
    TOOL_START = "tool_start"     # 工具调用开始
    TOOL_PROGRESS = "tool_progress"  # 工具调用进度
    TOOL_RESULT = "tool_result"   # 工具调用结果

    # Agent 状态
    THINKING = "thinking"         # 思考中
    ACTION = "action"             # 行动中
    OBSERVATION = "observation"   # 观察结果

    # 生命周期
    START = "start"               # 开始处理
    COMPLETE = "complete"         # 处理完成
    ERROR = "error"               # 错误


class StreamEvent:
    """流式事件"""

    def __init__(
        self,
        event_type: StreamEventType,
        data: dict,
        timestamp: Optional[float] = None,
    ):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or time.time()

    def to_sse_format(self) -> str:
        """转换为 SSE 格式"""
        return (
            f"event: {self.event_type.value}\n"
            f"data: {json.dumps(self.data, ensure_ascii=False)}\n"
            f"id: {int(self.timestamp * 1000)}\n\n"
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }


# ============================================================================
# 第三部分：模拟流式 Agent
# ============================================================================

class StreamingAgent:
    """
    支持流式输出的 Agent

    通过 AsyncIterator 逐步产出事件，
    前端可以实时显示处理进度。
    """

    def __init__(self, name: str = "AI助手"):
        self.name = name

    async def stream_response(
        self, user_input: str
    ) -> AsyncIterator[StreamEvent]:
        """
        流式处理用户输入

        Yields:
            StreamEvent: 各种类型的流式事件
        """
        # 1. 开始事件
        yield StreamEvent(
            StreamEventType.START,
            {"message": f"开始处理: {user_input[:30]}..."},
        )

        # 2. 思考事件
        yield StreamEvent(
            StreamEventType.THINKING,
            {"message": "正在分析用户意图..."},
        )
        await asyncio.sleep(0.3)  # 模拟思考

        # 3. 判断是否需要工具
        needs_tool = any(
            kw in user_input
            for kw in ["天气", "计算", "搜索", "查"]
        )

        if needs_tool:
            # 4. 工具调用开始
            tool_name = self._decide_tool(user_input)
            yield StreamEvent(
                StreamEventType.TOOL_START,
                {"tool": tool_name, "message": f"调用工具: {tool_name}"},
            )
            await asyncio.sleep(0.2)

            # 5. 工具进度
            yield StreamEvent(
                StreamEventType.TOOL_PROGRESS,
                {"progress": 50, "message": "正在获取数据..."},
            )
            await asyncio.sleep(0.3)

            # 6. 工具结果
            tool_result = self._execute_tool(tool_name, user_input)
            yield StreamEvent(
                StreamEventType.TOOL_RESULT,
                {"tool": tool_name, "result": tool_result},
            )

            # 7. 观察结果
            yield StreamEvent(
                StreamEventType.OBSERVATION,
                {"message": f"获取到结果: {tool_result[:30]}..."},
            )
        else:
            await asyncio.sleep(0.2)

        # 8. 生成回复（逐句流式输出）
        response = self._generate_response(user_input, needs_tool)
        sentences = response.split("。")

        yield StreamEvent(
            StreamEventType.TOKEN,
            {"token": "", "message": "开始生成回复..."},
        )

        for sentence in sentences:
            if sentence.strip():
                yield StreamEvent(
                    StreamEventType.TOKEN,
                    {"token": sentence + "。"},
                )
                await asyncio.sleep(0.1)  # 模拟逐句输出

        # 9. 完成事件
        yield StreamEvent(
            StreamEventType.COMPLETE,
            {"message": "处理完成", "full_response": response},
        )

    def _decide_tool(self, user_input: str) -> str:
        """决定使用哪个工具"""
        if "天气" in user_input:
            return "get_weather"
        elif "计算" in user_input or any(c in user_input for c in "+-*/"):
            return "calculator"
        elif "搜索" in user_input or "查" in user_input:
            return "web_search"
        return "web_search"

    def _execute_tool(self, tool_name: str, user_input: str) -> str:
        """执行工具（模拟）"""
        if tool_name == "get_weather":
            return "北京：晴天，28°C"
        elif tool_name == "calculator":
            return "计算结果: 42"
        return "搜索到相关信息"

    def _generate_response(self, user_input: str, used_tool: bool) -> str:
        """生成回复"""
        if used_tool:
            return f"根据你的问题，我已经查询到相关信息。{user_input[:10]}的结果已经获取。"
        return f"收到你的问题：{user_input[:20]}。这是一个模拟回复。"


# ============================================================================
# 第四部分：SSE 服务端实现
# ============================================================================

def demonstrate_sse_server():
    """
    SSE (Server-Sent Events) 服务端实现

    SSE 是 HTTP 长连接技术，适合单向推送（服务端→客户端）。
    比 WebSocket 简单，适合流式文本输出场景。
    """

    print("=" * 60)
    print("SSE 服务端实现")
    print("=" * 60)

    sse_code = '''
# FastAPI SSE 实现

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

@app.get("/stream/chat")
async def stream_chat(message: str, session_id: str):
    """
    SSE 流式聊天端点

    客户端通过 EventSource 连接：
    const eventSource = new EventSource('/stream/chat?message=你好&session_id=xxx');

    eventSource.addEventListener('token', (event) => {
        const data = JSON.parse(event.data);
        appendToChat(data.token);  // 逐字显示
    });

    eventSource.addEventListener('tool_start', (event) => {
        const data = JSON.parse(event.data);
        showToolStatus(data.tool);  // 显示工具调用状态
    });

    eventSource.addEventListener('complete', (event) => {
        eventSource.close();  // 关闭连接
    });
    """

    agent = StreamingAgent()

    async def event_generator():
        # 发送 SSE 头
        yield f"data: {json.dumps({'type': 'connected'})}\\n\\n"

        # 流式产出事件
        async for event in agent.stream_response(message):
            yield event.to_sse_format()
            await asyncio.sleep(0.01)  # 避免发送过快

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx 不缓冲
        },
    )
'''
    print(sse_code)


# ============================================================================
# 第五部分：WebSocket 双向通信
# ============================================================================

def demonstrate_websocket():
    """
    WebSocket 实现

    与 SSE 不同，WebSocket 支持双向通信。
    适合需要客户端也发送消息的场景（如中断、反馈）。
    """

    print("=" * 60)
    print("WebSocket 双向通信")
    print("=" * 60)

    ws_code = '''
# FastAPI WebSocket 实现

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket 聊天端点

    客户端连接：
    const ws = new WebSocket('ws://localhost:8000/ws/chat');

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
            appendToChat(data.data.token);
        }
    };

    // 发送消息
    ws.send(JSON.stringify({message: '你好', session_id: 'xxx'}));

    // 中断生成
    ws.send(JSON.stringify({action: 'interrupt'}));
    """
    await websocket.accept()
    agent = StreamingAgent()
    generating = False

    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_json()

            if data.get("action") == "interrupt":
                generating = False
                await websocket.send_json({
                    "type": "interrupted",
                    "message": "生成已中断"
                })
                continue

            if "message" in data:
                generating = True
                # 流式发送事件
                async for event in agent.stream_response(data["message"]):
                    if not generating:
                        break
                    await websocket.send_json(event.to_dict())

    except WebSocketDisconnect:
        print("客户端断开连接")
'''
    print(ws_code)

    # ── SSE vs WebSocket 对比 ─────────────────────────────────────
    print("\n── SSE vs WebSocket 对比 ──")
    print(f"{'特性':<20} {'SSE':<20} {'WebSocket':<20}")
    print("-" * 60)
    print(f"{'方向':<20} {'单向(服务→客户)':<20} {'双向':<20}")
    print(f"{'协议':<20} {'HTTP':<20} {'WebSocket':<20}")
    print(f"{'复杂度':<20} {'低':<20} {'中':<20}")
    print(f"{'自动重连':<20} {'内置支持':<20} {'需手动实现':<20}")
    print(f"{'适用场景':<20} {'流式文本输出':<20} {'实时交互':<20}")
    print()


# ============================================================================
# 第六部分：前端消费流式数据
# ============================================================================

def demonstrate_frontend_consumption():
    """演示前端如何消费流式数据"""

    print("=" * 60)
    print("前端消费流式数据")
    print("=" * 60)

    frontend_code = '''
// 前端 SSE 消费示例

class AgentChatClient {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
        this.eventSource = null;
        this.onToken = null;      // 逐字回调
        this.onToolStart = null;  // 工具开始回调
        this.onComplete = null;   // 完成回调
        this.onError = null;      // 错误回调
    }

    async streamChat(message, sessionId) {
        // 关闭之前的连接
        if (this.eventSource) {
            this.eventSource.close();
        }

        // 创建 EventSource
        const url = `${this.apiUrl}/stream/chat?message=${
            encodeURIComponent(message)}&session_id=${sessionId}`;

        this.eventSource = new EventSource(url);

        // 监听 token 事件（逐字显示）
        this.eventSource.addEventListener('token', (event) => {
            const data = JSON.parse(event.data);
            if (this.onToken && data.token) {
                this.onToken(data.token);
            }
        });

        // 监听工具调用事件
        this.eventSource.addEventListener('tool_start', (event) => {
            const data = JSON.parse(event.data);
            if (this.onToolStart) {
                this.onToolStart(data.tool);
            }
        });

        // 监听完成事件
        this.eventSource.addEventListener('complete', (event) => {
            const data = JSON.parse(event.data);
            if (this.onComplete) {
                this.onComplete(data.full_response);
            }
            this.eventSource.close();
        });

        // 监听错误
        this.eventSource.onerror = (error) => {
            if (this.onError) {
                this.onError(error);
            }
            this.eventSource.close();
        };
    }

    interrupt() {
        // SSE 不支持中断，需要用 WebSocket
        if (this.eventSource) {
            this.eventSource.close();
        }
    }
}

// 使用示例
const client = new AgentChatClient('http://localhost:8000');

client.onToken = (token) => {
    document.getElementById('response').textContent += token;
};

client.onToolStart = (tool) => {
    showLoading(`正在调用 ${tool}...`);
};

client.onComplete = (fullResponse) => {
    hideLoading();
};

client.streamChat('北京今天天气怎么样？', 'session_001');
'''
    print(frontend_code)


# ============================================================================
# 第七部分：流式演示
# ============================================================================

async def demonstrate_streaming():
    """演示流式输出"""

    print("=" * 60)
    print("流式输出演示")
    print("=" * 60)

    agent = StreamingAgent()

    # ── 演示1：简单对话（无工具）──────────────────────────────────
    print("\n── 简单对话 ─")
    async for event in agent.stream_response("你好"):
        if event.event_type == StreamEventType.TOKEN:
            print(event.data.get("token", ""), end="", flush=True)
        elif event.event_type == StreamEventType.THINKING:
            print(f"\n[{event.data['message']}]", end="")
        elif event.event_type == StreamEventType.COMPLETE:
            print(f"\n[{event.data['message']}]")
    print()

    # ── 演示2：需要工具的对话 ─────────────────────────────────────
    print("\n── 工具调用 ──")
    async for event in agent.stream_response("北京今天天气怎么样？"):
        d = event.data
        if event.event_type == StreamEventType.TOOL_START:
            print(f"\n[调用工具: {d['tool']}]", end="")
        elif event.event_type == StreamEventType.TOOL_RESULT:
            print(f"\n[工具结果: {d['result']}]", end="")
        elif event.event_type == StreamEventType.TOKEN:
            print(d.get("token", ""), end="", flush=True)
        elif event.event_type == StreamEventType.COMPLETE:
            print(f"\n[{d['message']}]")
    print()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "📡" * 30)
    print("第十四课（补充）：Streaming 与实时输出")
    print("📡" * 30 + "\n")

    # 1. 为什么需要流式
    why_streaming_matters()

    # 2. SSE 实现
    demonstrate_sse_server()

    # 3. WebSocket 实现
    demonstrate_websocket()

    # 4. 前端消费
    demonstrate_frontend_consumption()

    # 5. 流式演示
    asyncio.run(demonstrate_streaming())

    print("=" * 60)
    print("✅ 第十四课学习完成！")
    print("=" * 60)
    print("""
 本课要点总结：
1. 流式输出大幅提升用户体验（首字 < 500ms）
2. SSE 适合单向推送（简单、内置重连）
3. WebSocket 适合双向通信（支持中断、反馈）
4. 工具调用状态也应该实时推送
5. 前端通过 EventSource 或 WebSocket 消费流式数据
    """)
