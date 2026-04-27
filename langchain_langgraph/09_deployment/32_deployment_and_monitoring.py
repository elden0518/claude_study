"""
主题：LangGraph 部署 —— API 服务化、Docker 容器化、生产监控

学习目标：
  1. 掌握 FastAPI 封装 LangGraph（REST API）
  2. 学会 Docker 容器化部署
  3. 理解生产环境的关键配置（并发、超时、健康检查）
  4. 掌握日志记录和性能监控
  5. 了解水平扩展和负载均衡策略

核心概念：
  开发环境 → 生产环境的转变：
  - 本地调用 → HTTP API
  - 手动运行 → 容器化部署
  - 单机执行 → 分布式扩展
  
  生产级要求：
  ✅ 高可用（99.9%+ uptime）
  ✅ 可扩展（支持水平扩展）
  ✅ 可观测（日志、指标、追踪）
  ✅ 安全性（认证、限流、加密）

前置知识：已完成 04_lg_advanced/21_streaming.py
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Part 1：FastAPI 封装 LangGraph
# =============================================================================

def demo_fastapi_wrapper():
    """
    将 LangGraph 应用包装为 REST API 服务。
    
    架构：
    Client (HTTP) → FastAPI → LangGraph → LLM
    
    关键设计点：
    1. 异步处理（避免阻塞）
    2. 流式响应（SSE，提升用户体验）
    3. 请求验证（Pydantic Schema）
    4. 错误处理（统一错误格式）
    5. CORS 配置（跨域支持）
    """
    print("=" * 60)
    print("Part 1: FastAPI Wrapper —— REST API 封装")
    print("=" * 60)
    
    api_code = '''
# file: api_server.py
"""
LangGraph API 服务器示例
启动命令：uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import os
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# 导入你的 LangGraph 应用
# from your_module import build_agent_graph

# ── 数据模型 ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(..., description="用户消息", min_length=1, max_length=5000)
    thread_id: Optional[str] = Field(None, description="会话 ID（用于多轮对话）")
    stream: bool = Field(False, description="是否启用流式输出")

class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    thread_id: str
    tokens_used: int

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str

# ── 全局状态 ──────────────────────────────────────────────

checkpointer = None
agent_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    global checkpointer, agent_app
    checkpointer = MemorySaver()
    # agent_app = build_agent_graph().compile(checkpointer=checkpointer)
    print("✅ LangGraph 应用已初始化")
    
    yield
    
    # 关闭时清理
    print("🔄 LangGraph 应用已关闭")

# ── 创建 FastAPI 应用 ─────────────────────────────────────

app = FastAPI(
    title="LangGraph API",
    description="基于 LangGraph 的智能对话 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API 端点 ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天端点（同步模式）
    
    示例请求：
    curl -X POST http://localhost:8000/chat \\
      -H "Content-Type: application/json" \\
      -d '{"message": "你好", "thread_id": "user123"}'
    """
    try:
        # 生成或使用提供的 thread_id
        thread_id = request.thread_id or f"session_{os.urandom(8).hex()}"
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # 调用 LangGraph
        # result = agent_app.invoke(
        #     {"messages": [{"role": "user", "content": request.message}]},
        #     config
        # )
        
        # 模拟响应
        result = {
            "messages": [{"role": "assistant", "content": f"收到：{request.message}"}]
        }
        
        return ChatResponse(
            response=result["messages"][-1]["content"],
            thread_id=thread_id,
            tokens_used=100  # 实际应从 response.usage_metadata 获取
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    聊天端点（流式模式，Server-Sent Events）
    
    使用 SSE 实时推送每个 token，适合前端打字机效果。
    """
    from fastapi.responses import StreamingResponse
    
    async def generate():
        try:
            thread_id = request.thread_id or f"session_{os.urandom(8).hex()}"
            config = {"configurable": {"thread_id": thread_id}}
            
            # 流式调用 LangGraph
            # async for event in agent_app.astream_events(...):
            #     yield f"data: {json.dumps(event)}\\n\\n"
            
            # 模拟流式输出
            yield "data: {\"token\": \"你\"}\\n\\n"
            yield "data: {\"token\": \"好\"}\\n\\n"
            yield "data: {\"done\": true}\\n\\n"
        
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\\n\\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/threads/{thread_id}/history")
async def get_thread_history(thread_id: str):
    """获取会话历史"""
    try:
        # history = agent_app.get_state_history({"configurable": {"thread_id": thread_id}})
        return {"thread_id": thread_id, "messages": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 启动命令 ──────────────────────────────────────────────
"""
开发环境：
$ uvicorn api_server:app --reload --port 8000

生产环境：
$ uvicorn api_server:app \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --workers 4 \\
    --log-level info
"""
    '''
    
    print(api_code)
    print("\n💡 关键点：")
    print("  • 使用 Pydantic 进行请求验证")
    print("  • 提供同步和流式两种接口")
    print("  • 健康检查端点用于负载均衡器")
    print("  • lifespan 管理资源生命周期")


# =============================================================================
# Part 2：Docker 容器化部署
# =============================================================================

def demo_docker_deployment():
    """
    Docker 容器化部署的优势：
    - 环境一致性（避免"在我机器上能跑"）
    - 快速部署和回滚
    - 资源隔离和限制
    - 易于水平扩展
    """
    print("\n" + "=" * 60)
    print("Part 2: Docker Deployment —— 容器化部署")
    print("=" * 60)
    
    # Dockerfile
    dockerfile = '''
# file: Dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非 root 用户（安全最佳实践）
RUN useradd --create-home appuser
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 启动命令
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
    '''
    
    print("【Dockerfile】")
    print("-" * 60)
    print(dockerfile)
    print("-" * 60)
    
    # docker-compose.yml
    compose_file = '''
# file: docker-compose.yml
version: '3.8'

services:
  langgraph-api:
    build: .
    container_name: langgraph-api
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
      - LANGCHAIN_TRACING_V2=true
    volumes:
      - ./logs:/app/logs  # 持久化日志
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - app-network

  # Redis 缓存（可选）
  redis:
    image: redis:7-alpine
    container_name: langgraph-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
    networks:
      - app-network

volumes:
  redis-data:

networks:
  app-network:
    driver: bridge
    '''
    
    print("\n【docker-compose.yml】")
    print("-" * 60)
    print(compose_file)
    print("-" * 60)
    
    # 部署命令
    print("\n【部署命令】")
    print("-" * 60)
    commands = '''
# 1. 构建镜像
$ docker-compose build

# 2. 启动服务
$ docker-compose up -d

# 3. 查看日志
$ docker-compose logs -f langgraph-api

# 4. 检查健康状态
$ docker-compose ps

# 5. 停止服务
$ docker-compose down

# 6. 更新部署
$ docker-compose pull
$ docker-compose up -d

# 7. 查看资源使用
$ docker stats langgraph-api
    '''
    print(commands)
    print("-" * 60)


# =============================================================================
# Part 3：生产环境配置
# =============================================================================

def demo_production_config():
    """
    生产环境的关键配置项。
    """
    print("\n" + "=" * 60)
    print("Part 3: Production Configuration")
    print("=" * 60)
    
    config_guide = """
  ⚙️  生产环境配置清单：
  ──────────────────────────────────────────────
  
  1️⃣  并发控制
  ──────────────────────────────────────────────
  
  Uvicorn  workers 配置：
  • 公式：workers = (2 × CPU cores) + 1
  • 示例：4 核 CPU → 9 workers
  
  异步 vs 同步：
  • I/O 密集型（LLM 调用）→ 异步（async/await）
  • CPU 密集型（数据处理）→ 多进程
  
  连接池：
  • 数据库连接池大小：10-20
  • HTTP 客户端连接池：50-100
  
  
  2️⃣  超时配置
  ──────────────────────────────────────────────
  
  各级超时设置：
  • LLM 调用：30-60 秒
  • API 请求：120 秒
  • 健康检查：10 秒
  • 优雅关闭：30 秒
  
  Nginx 反向代理：
  proxy_read_timeout 120s;
  proxy_connect_timeout 10s;
  
  
  3️⃣  资源限制
  ──────────────────────────────────────────────
  
  Docker 资源限制：
  • CPU：根据负载设置（建议 2-4 核）
  • 内存：预留 20% 缓冲（防止 OOM）
  • 磁盘：监控日志增长
  
  Kubernetes 资源配置：
  resources:
    requests:
      cpu: "1"
      memory: "2Gi"
    limits:
      cpu: "2"
      memory: "4Gi"
  
  
  4️⃣  日志配置
  ──────────────────────────────────────────────
  
  日志级别：
  • 开发：DEBUG
  • 测试：INFO
  • 生产：WARNING（错误时临时切换到 DEBUG）
  
  日志格式（JSON，便于解析）：
  {
    "timestamp": "2024-01-01T00:00:00Z",
    "level": "INFO",
    "message": "Request processed",
    "thread_id": "abc123",
    "duration_ms": 1234,
    "tokens_used": 500
  }
  
  日志轮转：
  • 单文件大小：< 100MB
  • 保留天数：7-30 天
  • 压缩：gzip
  
  
  5️⃣  监控指标
  ──────────────────────────────────────────────
  
  关键指标（Prometheus）：
  • 请求率（requests/sec）
  • 延迟分布（P50/P95/P99）
  • 错误率（5xx 比例）
  • Token 消耗量
  • 活跃会话数
  
  告警规则：
  • 错误率 > 5% 持续 5 分钟
  • P95 延迟 > 10 秒
  • 内存使用 > 85%
  • Token 成本超出预算
  
  
  6️⃣  安全配置
  ──────────────────────────────────────────────
  
  API 认证：
  • API Key（简单场景）
  • JWT Token（用户系统）
  • OAuth2（第三方集成）
  
  速率限制：
  • 每用户：100 req/min
  • 每 IP：1000 req/hour
  
  输入验证：
  • 最大请求长度：5000 字符
  • 过滤敏感词
  • SQL 注入防护
  
  HTTPS：
  • 强制 HTTPS 重定向
  • TLS 1.2+
  • 证书自动续期（Let's Encrypt）
    """
    print(config_guide)


# =============================================================================
# Part 4：水平扩展与负载均衡
# =============================================================================

def demo_horizontal_scaling():
    """
    如何应对高并发流量。
    """
    print("\n" + "=" * 60)
    print("Part 4: Horizontal Scaling & Load Balancing")
    print("=" * 60)
    
    scaling_guide = """
  📈 水平扩展策略：
  ──────────────────────────────────────────────
  
  架构演进：
  
  阶段 1：单机部署（< 100 QPS）
  ┌──────────┐
  │ Client   │
  └────┬─────┘
       ↓
  ┌──────────┐
  │ Server   │ ← 单实例
  └──────────┘
  
  阶段 2：多实例 + 负载均衡（100-1000 QPS）
  ┌──────────┐
  │ Client   │
  └────┬─────┘
       ↓
  ┌──────────┐
  │   LB     │ ← Nginx / AWS ALB
  └─┬─┬─┬────┘
    ↓ ↓ ↓
  ┌──┐┌──┐┌──┐
  │S1││S2││S3│ ← 多实例
  └──┘└──┘└──┘
  
  阶段 3：分布式 + 缓存（> 1000 QPS）
  ┌──────────┐
  │ Client   │
  └────┬─────┘
       ↓
  ┌──────────┐
  │   CDN    │ ← 静态资源
  └────┬─────┘
       ↓
  ┌──────────┐
  │   LB     │
  └─┬─┬─┬────┘
    ↓ ↓ ↓
  ┌──┐┌──┐┌──┐
  │S1││S2││S3│
  └──┘└──┘└──┘
    ↓
  ┌──────────┐
  │  Redis   │ ← 共享缓存
  └──────────┘
  
  
  🔧 负载均衡策略：
  ──────────────────────────────────────────────
  
  1. Round Robin（轮询）
     • 简单均匀分配
     • 适合实例性能一致
  
  2. Least Connections（最少连接）
     • 优先分配给空闲实例
     • 适合长连接场景
  
  3. IP Hash（会话保持）
     • 同一用户始终路由到同一实例
     • 适合有状态应用（但 LangGraph 用外部存储）
  
  
  💾 状态管理（无状态设计）：
  ──────────────────────────────────────────────
  
  问题：多实例如何共享会话状态？
  
  方案 1：外部 Checkpointer
  • PostgreSQL / Redis
  • 所有实例读写同一存储
  • 推荐：langgraph-postgres
  
  方案 2：会话粘性
  • 同一 thread_id 始终路由到同一实例
  • 简单但不够灵活
  
  推荐架构：
  ┌─────────────────────────────────────┐
  │         Load Balancer               │
  └──┬──────────┬──────────┬───────────┘
     ↓          ↓          ↓
  ┌──────┐  ┌──────┐  ┌──────┐
  │Inst 1│  │Inst 2│  │Inst 3│
  └──┬───┘  └──┬───┘  └──┬───┘
     │          │          │
     └──────────┴──────────┘
                ↓
     ┌──────────────────┐
     │  PostgreSQL      │ ← 共享状态存储
     │  (Checkpointer)  │
     └──────────────────┘
  
  
  🚀 Kubernetes 部署示例：
  ──────────────────────────────────────────────
  
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: langgraph-api
  spec:
    replicas: 3  # 初始副本数
    selector:
      matchLabels:
        app: langgraph-api
    template:
      metadata:
        labels:
          app: langgraph-api
      spec:
        containers:
        - name: api
          image: langgraph-api:latest
          ports:
          - containerPort: 8000
          resources:
            requests:
              cpu: "1"
              memory: "2Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
  
  ---
  apiVersion: v1
  kind: Service
  metadata:
    name: langgraph-service
  spec:
    selector:
      app: langgraph-api
    ports:
    - port: 80
      targetPort: 8000
    type: LoadBalancer
  
  ---
  apiVersion: autoscaling/v2
  kind: HorizontalPodAutoscaler
  metadata:
    name: langgraph-hpa
  spec:
    scaleTargetRef:
      apiVersion: apps/v1
      kind: Deployment
      name: langgraph-api
    minReplicas: 3
    maxReplicas: 10
    metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    """
    print(scaling_guide)


def main():
    print("=" * 60)
    print("LangGraph 部署详解")
    print("=" * 60)
    
    demo_fastapi_wrapper()
    demo_docker_deployment()
    demo_production_config()
    demo_horizontal_scaling()
    
    print("\n" + "=" * 60)
    print("学习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
