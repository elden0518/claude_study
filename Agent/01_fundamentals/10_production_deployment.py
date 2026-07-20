"""
==============================================================================
第十课：生产环境部署
==============================================================================

【学习目标】
- 掌握 Docker 容器化部署
- 理解生产环境的配置管理
- 掌握日志收集和监控
- 学会水平扩展和负载均衡
- 了解 CI/CD 流程

【核心概念】
- Docker 容器化
- 环境变量与配置
- 日志管理
- 监控与告警
- 水平扩展

【前置知识】
- 所有前序课程

==============================================================================
"""

import json
import os
from datetime import datetime
from typing import Optional


# ============================================================================
# 第一部分：生产环境 vs 开发环境的区别
# ============================================================================

def compare_environments():
    """对比开发环境和生产环境的区别"""

    comparisons = [
        {
            "aspect": "LLM 调用",
            "development": "Mock LLM 或免费额度",
            "production": "真实 API，需要处理限流和错误",
        },
        {
            "aspect": "会话存储",
            "development": "内存存储",
            "production": "Redis/数据库，持久化",
        },
        {
            "aspect": "日志",
            "development": "控制台输出",
            "production": "文件 + 集中式日志系统",
        },
        {
            "aspect": "配置",
            "development": ".env 文件",
            "production": "环境变量 + 密钥管理",
        },
        {
            "aspect": "部署",
            "development": "本地运行",
            "production": "Docker + 编排系统",
        },
        {
            "aspect": "监控",
            "development": "手动检查",
            "production": "自动监控 + 告警",
        },
        {
            "aspect": "扩展性",
            "development": "单进程",
            "production": "多 worker + 负载均衡",
        },
    ]

    print("=" * 60)
    print("开发环境 vs 生产环境")
    print("=" * 60)
    print(f"{'方面':<15} {'开发环境':<25} {'生产环境':<25}")
    print("-" * 65)
    for c in comparisons:
        print(f"{c['aspect']:<15} {c['development']:<25} {c['production']:<25}")
    print()


# ============================================================================
# 第二部分：Docker 容器化部署
# ============================================================================

def docker_deployment():
    """
    Docker 容器化部署

    Docker 的好处：
    1. 环境一致性（开发/测试/生产一样）
    2. 依赖隔离
    3. 易于扩展
    4. 快速部署
    """

    print("=" * 60)
    print("Docker 容器化部署")
    print("=" * 60)

    # ── Dockerfile ───────────────────────────────────────────────
    dockerfile = '''
# Dockerfile - Agent 服务容器化配置

# 使用 Python 3.11  slim 镜像（体积小）
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非 root 用户（安全最佳实践）
RUN useradd -m -u 1000 agentuser && \\
    chown -R agentuser:agentuser /app
USER agentuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \\
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" \\
    || exit 1

# 启动命令
CMD ["uvicorn", "entry.api:app", "--host", "0.0.0.0", "--port", "8000"]
'''

    print("\n── Dockerfile ──")
    print(dockerfile)

    # ── docker-compose.yml ───────────────────────────────────────
    compose_file = '''
# docker-compose.yml - 多服务编排

version: '3.8'

services:
  # Agent API 服务
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      replicas: 2  # 运行2个实例
      resources:
        limits:
          memory: 512M
          cpus: '0.5'

  # Redis 缓存/会话存储
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  # Nginx 反向代理（可选）
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - agent-api
    restart: unless-stopped

volumes:
  redis_data:
'''

    print("\n── docker-compose.yml ─")
    print(compose_file)

    # ── 部署命令 ─────────────────────────────────────────────────
    print("\n── 部署命令 ──")
    commands = [
        "# 构建镜像",
        "docker-compose build",
        "",
        "# 启动服务",
        "docker-compose up -d",
        "",
        "# 查看日志",
        "docker-compose logs -f agent-api",
        "",
        "# 扩展实例数",
        "docker-compose up -d --scale agent-api=4",
        "",
        "# 停止服务",
        "docker-compose down",
    ]
    for cmd in commands:
        print(f"  {cmd}")
    print()


# ============================================================================
# 第三部分：生产环境配置管理
# ============================================================================

def production_config():
    """生产环境配置管理"""

    print("=" * 60)
    print("生产环境配置管理")
    print("=" * 60)

    # ─ 环境变量配置 ─────────────────────────────────────────────
    env_config = '''
# .env.production - 生产环境配置示例

# LLM 配置
ANTHROPIC_API_KEY=sk-ant-xxxxx
LLM_MODEL=claude-sonnet-4-20250514
LLM_MAX_TOKENS=4096
LLM_TEMPERATURE=0.7

# Redis 配置
REDIS_URL=redis://redis:6379/0
REDIS_PASSWORD=your_redis_password

# 应用配置
APP_ENV=production
LOG_LEVEL=WARNING
LOG_FILE=/var/log/agent/app.log

# 安全配置
API_SECRET_KEY=your_secret_key_here
CORS_ORIGINS=https://yourdomain.com

# 性能配置
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30
'''

    print("\n── .env.production ──")
    print(env_config)

    # ── 配置最佳实践 ─────────────────────────────────────────────
    print("\n── 配置最佳实践 ─")
    best_practices = [
        "1. 敏感信息（API Key、密码）使用环境变量，不要写入代码",
        "2. 不同环境使用不同的 .env 文件（.env.dev, .env.prod）",
        "3. .env 文件不要提交到 Git（添加到 .gitignore）",
        "4. 使用密钥管理服务（AWS Secrets Manager、Vault 等）",
        "5. 配置验证：启动时检查必要配置是否存在",
        "6. 配置热重载：修改配置后不需要重启服务",
    ]
    for practice in best_practices:
        print(f"  {practice}")
    print()


# ============================================================================
# 第四部分：日志管理
# ============================================================================

def logging_management():
    """生产环境日志管理"""

    print("=" * 60)
    print("生产环境日志管理")
    print("=" * 60)

    # ── 日志配置 ─────────────────────────────────────────────────
    log_config = '''
# 生产环境日志配置

import logging
import logging.handlers
from pathlib import Path

def setup_production_logger(name: str, log_dir: str = "/var/log/agent"):
    """配置生产环境日志"""

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 1. 文件 Handler（按天滚动）
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_path / "agent.log",
        when="midnight",      # 每天滚动
        interval=1,
        backupCount=30,       # 保留30天
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)

    # 2. 错误文件 Handler（只记录 ERROR 及以上）
    error_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_path / "error.log",
        when="midnight",
        interval=1,
        backupCount=90,       # 错误日志保留更久
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)

    # 3. JSON 格式（便于日志系统解析）
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps({
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }, ensure_ascii=False)

    formatter = JSONFormatter()
    file_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger
'''

    print("\n── 日志配置代码 ──")
    print(log_config)

    # ── 日志级别说明 ─────────────────────────────────────────────
    print("\n── 日志级别说明 ──")
    levels = [
        ("DEBUG", "调试信息，生产环境通常不启用"),
        ("INFO", "一般信息，记录重要操作"),
        ("WARNING", "警告信息，需要注意但不影响运行"),
        ("ERROR", "错误信息，操作失败"),
        ("CRITICAL", "严重错误，系统可能无法继续运行"),
    ]
    for level, desc in levels:
        print(f"  {level:<10} - {desc}")
    print()


# ============================================================================
# 第五部分：监控与告警
# ============================================================================

def monitoring_and_alerting():
    """监控与告警"""

    print("=" * 60)
    print("监控与告警")
    print("=" * 60)

    # ── 监控指标 ─────────────────────────────────────────────────
    print("\n── 关键监控指标 ──")
    metrics = [
        {
            "category": "应用指标",
            "items": [
                "请求量（QPS）",
                "响应时间（P50/P95/P99）",
                "错误率",
                "活跃会话数",
                "Agent 迭代次数分布",
            ],
        },
        {
            "category": "LLM 指标",
            "items": [
                "Token 消耗量",
                "API 调用次数",
                "API 错误率",
                "平均响应时间",
            ],
        },
        {
            "category": "系统指标",
            "items": [
                "CPU 使用率",
                "内存使用率",
                "磁盘使用率",
                "网络 IO",
            ],
        },
    ]

    for category in metrics:
        print(f"\n  {category['category']}:")
        for item in category["items"]:
            print(f"    - {item}")

    # ── 告警规则 ─────────────────────────────────────────────────
    print("\n── 告警规则示例 ─")
    alert_rules = [
        {
            "condition": "错误率 > 5%",
            "severity": "WARNING",
            "action": "发送通知到 Slack",
        },
        {
            "condition": "错误率 > 20%",
            "severity": "CRITICAL",
            "action": "发送通知 + 自动降级",
        },
        {
            "condition": "P99 响应时间 > 10s",
            "severity": "WARNING",
            "action": "发送通知",
        },
        {
            "condition": "Token 消耗超过预算 80%",
            "severity": "WARNING",
            "action": "发送通知给管理员",
        },
        {
            "condition": "内存使用率 > 90%",
            "severity": "CRITICAL",
            "action": "发送通知 + 考虑扩容",
        },
    ]

    for rule in alert_rules:
        print(f"  条件: {rule['condition']}")
        print(f"    级别: {rule['severity']}")
        print(f"    动作: {rule['action']}")
        print()


# ============================================================================
# 第六部分：水平扩展与负载均衡
# ============================================================================

def horizontal_scaling():
    """水平扩展与负载均衡"""

    print("=" * 60)
    print("水平扩展与负载均衡")
    print("=" * 60)

    # ── 扩展策略 ─────────────────────────────────────────────────
    print("\n── 扩展策略 ──")
    strategies = [
        {
            "name": "垂直扩展（Scale Up）",
            "description": "增加单台机器的资源（CPU、内存）",
            "pros": ["简单", "不需要改代码"],
            "cons": ["有上限", "成本高", "单点故障"],
        },
        {
            "name": "水平扩展（Scale Out）",
            "description": "增加机器数量，分散负载",
            "pros": ["理论上无限扩展", "高可用", "成本可控"],
            "cons": ["需要负载均衡", "会话状态需要共享存储"],
        },
    ]

    for s in strategies:
        print(f"\n  {s['name']}")
        print(f"    描述: {s['description']}")
        print(f"    优点: {', '.join(s['pros'])}")
        print(f"    缺点: {', '.join(s['cons'])}")

    # ── 无状态设计 ───────────────────────────────────────────────
    print("\n── 无状态设计（水平扩展的关键）──")
    print("""
  要支持水平扩展，应用必须是无状态的：

  ❌ 有状态设计（无法水平扩展）:
     - 会话数据存在内存中
     - 每个请求必须路由到同一台机器

  ✅ 无状态设计（可以水平扩展）:
     - 会话数据存在 Redis/数据库中
     - 任何机器都可以处理任何请求
     - 通过负载均衡器分发请求

  ┌─────────────────────────────────────────────────────────────┐
  │                    水平扩展架构                               │
  │                                                             │
  │   用户 → 负载均衡器 → [实例1] [实例2] [实例3]               │
  │                              │      │      │                │
  │                              └──────┴──────┘                │
  │                                    │                        │
  │                              ┌───────┴───────┐              │
  │                              │   Redis/DB    │              │
  │                              │  (共享存储)    │              │
  │                              └───────────────┘              │
  └─────────────────────────────────────────────────────────────┘
    """)


# ============================================================================
# 第七部分：CI/CD 流程
# ============================================================================

def cicd_pipeline():
    """CI/CD 流程"""

    print("=" * 60)
    print("CI/CD 流程")
    print("=" * 60)

    pipeline = """
  ── CI/CD Pipeline ──

  代码提交 → CI 检查 → 构建 → 测试 → 部署 → 监控

  详细步骤：

  1. 代码提交（Push）
     - 开发者推送代码到 Git 仓库
     - 触发 CI/CD Pipeline

  2. CI 检查（Continuous Integration）
     - 代码风格检查（lint）
     - 类型检查（mypy）
     - 单元测试
     - 集成测试
     - 覆盖率检查

  3. 构建（Build）
     - 构建 Docker 镜像
     - 推送到镜像仓库

  4. 部署（Deploy）
     - 部署到测试环境
     - 运行端到端测试
     - 部署到生产环境（蓝绿部署/金丝雀部署）

  5. 监控（Monitor）
     - 监控应用指标
     - 监控 LLM 调用
     - 异常告警
    """
    print(pipeline)

    # ── GitHub Actions 示例 ──────────────────────────────────────
    github_actions = '''
# .github/workflows/ci.yml

name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: pytest tests/ -v --cov=agent

      - name: Build Docker image
        run: docker build -t agent-app .

      - name: Deploy to production
        if: github.ref == 'refs/heads/main'
        run: |
          docker-compose pull
          docker-compose up -d
    '''

    print("\n── GitHub Actions 示例 ──")
    print(github_actions)


# ============================================================================
# 第八部分：部署检查清单
# ============================================================================

def deployment_checklist():
    """部署检查清单"""

    print("=" * 60)
    print("部署检查清单")
    print("=" * 60)

    checklist = [
        {
            "category": "代码质量",
            "items": [
                "所有测试通过",
                "代码覆盖率 > 80%",
                "无安全漏洞",
                "依赖版本锁定",
            ],
        },
        {
            "category": "配置管理",
            "items": [
                "API Key 使用环境变量",
                "不同环境配置分离",
                "敏感信息不提交到 Git",
                "配置验证通过",
            ],
        },
        {
            "category": "安全",
            "items": [
                "使用 HTTPS",
                "CORS 配置正确",
                "输入验证",
                "速率限制",
                "非 root 用户运行",
            ],
        },
        {
            "category": "性能",
            "items": [
                "连接池配置",
                "缓存策略",
                "超时设置",
                "重试机制",
            ],
        },
        {
            "category": "监控",
            "items": [
                "日志配置",
                "健康检查端点",
                "监控指标",
                "告警规则",
            ],
        },
        {
            "category": "备份与恢复",
            "items": [
                "数据库备份",
                "会话数据备份",
                "灾难恢复计划",
                "回滚方案",
            ],
        },
    ]

    for category in checklist:
        print(f"\n {category['category']}:")
        for i, item in enumerate(category["items"], 1):
            print(f"   ☐ {i}. {item}")
    print()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "" * 30)
    print("第十课：生产环境部署")
    print("" * 30 + "\n")

    # 1. 环境对比
    compare_environments()

    # 2. Docker 部署
    docker_deployment()

    # 3. 配置管理
    production_config()

    # 4. 日志管理
    logging_management()

    # 5. 监控告警
    monitoring_and_alerting()

    # 6. 水平扩展
    horizontal_scaling()

    # 7. CI/CD
    cicd_pipeline()

    # 8. 检查清单
    deployment_checklist()

    print("=" * 60)
    print("✅ 第十课学习完成！")
    print("=" * 60)
    print("""
 本课要点总结：
1. 生产环境需要 Docker 容器化保证环境一致性
2. 配置管理：环境变量、密钥管理、配置验证
3. 日志管理：分级日志、滚动日志、JSON 格式
4. 监控告警：应用指标、LLM 指标、系统指标
5. 水平扩展：无状态设计 + 负载均衡 + 共享存储
6. CI/CD：自动化测试、构建、部署流程

 下一课：进阶 - 多 Agent 协作与高级模式
    """)
