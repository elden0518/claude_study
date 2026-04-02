"""
主题：Prompt Caching（提示词缓存）—— 大幅降低 API 成本

学习目标：
  1. 理解 cache_control 参数的作用原理
  2. 掌握在 system prompt 和 messages 中添加缓存断点
  3. 学会通过 usage.cache_read_input_tokens 验证缓存命中
  4. 理解缓存的限制条件（最小长度、TTL、可缓存位置）
  5. 掌握多轮对话中的缓存最优策略

核心原理：
  - 普通调用：每次请求都重新处理所有 token（费用 = 输入 × 单价）
  - 缓存命中：首次写入缓存（写入费 = 输入 × 1.25x），后续读取（读取费 = 输入 × 0.1x）
  - 节省最大场景：大量重复使用同一 system prompt / 长文档分析

缓存限制：
  - 最小缓存块：1024 tokens（Sonnet / Opus），2048 tokens（Haiku）
  - 缓存 TTL：5 分钟（不活跃后过期）
  - 每个请求最多 4 个 cache_control 断点

前置知识：
  - 已完成 01_hello_claude.py 和 02_messages_format.py
"""

# ── 0. Windows 控制台编码修复 ──────────────────────────────────────────────────
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── 1. 导入 ────────────────────────────────────────────────────────────────────
import os
import time
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
MODEL = "ppio/pa/claude-sonnet-4-6"

# =============================================================================
# 构造一个"超长 system prompt"用于演示缓存效果
# 实际项目中，这可以是：API 文档、代码库说明、知识库文章等
# =============================================================================

LONG_SYSTEM_PROMPT = """
你是一位资深的软件架构师，拥有 20 年企业级系统设计经验。
你熟悉以下所有领域，并能给出详细、有依据的建议：

【后端架构】
- 微服务架构（Service Mesh、API Gateway、服务注册/发现）
- 事件驱动架构（Kafka、RabbitMQ、事件溯源）
- 分层架构（DDD 领域驱动设计、CQRS 命令查询分离）
- 高可用设计（主从复制、分片、熔断限流）

【数据库选型】
- 关系型数据库：PostgreSQL（窗口函数、分区表）、MySQL（InnoDB 引擎调优）
- NoSQL：Redis（数据结构选型、集群模式）、MongoDB（索引优化）、Cassandra（宽列存储）
- NewSQL：TiDB、CockroachDB（分布式事务）
- 搜索引擎：Elasticsearch（倒排索引、分片策略）

【性能优化】
- 缓存策略（多级缓存、缓存击穿/雪崩/穿透）
- 数据库优化（慢查询分析、执行计划、索引覆盖）
- 异步化（消息队列解耦、异步任务）
- CDN 与边缘计算

【安全架构】
- OAuth 2.0 / OIDC 认证授权流程
- JWT 安全设计（签名算法、刷新策略、撤销机制）
- API 安全（限流、防重放、参数校验）
- 零信任安全模型

【云原生】
- Kubernetes（Pod 调度、HPA/VPA 自动扩缩容、Helm Chart）
- CI/CD（GitOps、ArgoCD、流水线设计）
- 可观测性（Prometheus + Grafana、分布式追踪 Jaeger）
- Serverless（FaaS 冷启动优化、状态管理）

你的回答风格：
- 先给出清晰的结论和建议
- 再说明技术选型理由（权衡 Trade-off）
- 提供具体的实现路径或代码示例
- 指出常见陷阱和需要注意的边界情况
""" * 3  # 重复 3 次以确保超过 1024 token 的缓存最低要求


# =============================================================================
# Part 1：在 system prompt 中添加缓存断点
# =============================================================================

def demo_system_prompt_cache():
    """演示缓存 system prompt，对比首次调用与缓存命中的 token 用量差异。"""

    print("=" * 60)
    print("Part 1：System Prompt 缓存")
    print("=" * 60)
    print(f"  System prompt 长度（字符）：{len(LONG_SYSTEM_PROMPT)}")
    print()

    # cache_control 添加位置：system 参数改为列表格式，在要缓存的内容末尾加断点
    # {"type": "ephemeral"} 表示"在此处设置缓存断点"（ephemeral = 短暂缓存，TTL 5分钟）
    system_with_cache = [
        {
            "type": "text",
            "text": LONG_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}   # ← 缓存断点
        }
    ]

    # ── 第一次调用：写入缓存（cache write）─────────────────────────────────────
    print("── 第 1 次调用（写入缓存）──────────────────────────────────")
    resp1 = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=system_with_cache,
        messages=[{"role": "user", "content": "微服务架构和单体架构各有什么优缺点？"}],
    )

    u1 = resp1.usage
    print(f"  输入 tokens            : {u1.input_tokens}")
    print(f"  缓存写入 tokens        : {getattr(u1, 'cache_creation_input_tokens', 0)}")
    print(f"  缓存读取 tokens        : {getattr(u1, 'cache_read_input_tokens', 0)}")
    print(f"  输出 tokens            : {u1.output_tokens}")
    print(f"  回答（前200字）        : {resp1.content[0].text[:200]}...")

    # ── 第二次调用：命中缓存（cache hit）──────────────────────────────────────
    print("\n── 第 2 次调用（命中缓存，成本降至约 10%）───────────────────")
    resp2 = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=system_with_cache,
        messages=[{"role": "user", "content": "Redis 在高并发场景下应该如何选择数据结构？"}],
    )

    u2 = resp2.usage
    print(f"  输入 tokens            : {u2.input_tokens}")
    print(f"  缓存写入 tokens        : {getattr(u2, 'cache_creation_input_tokens', 0)}")
    print(f"  缓存读取 tokens（命中）: {getattr(u2, 'cache_read_input_tokens', 0)}")
    print(f"  输出 tokens            : {u2.output_tokens}")

    cache_read = getattr(u2, 'cache_read_input_tokens', 0)
    if cache_read > 0:
        print(f"\n  ✓ 缓存命中！节省了约 {cache_read} tokens 的处理费用（约 90% 折扣）")
    else:
        print("\n  ℹ 未命中（可能缓存已过期，或 prompt 不满足最小长度要求）")


# =============================================================================
# Part 2：在 messages 中缓存长文档（RAG / 文档分析场景）
# =============================================================================

def demo_message_cache():
    """演示缓存 user 消息中的长文档，适用于反复对同一文档提问的场景。"""

    print("\n" + "=" * 60)
    print("Part 2：Messages 中的文档缓存（RAG 场景）")
    print("=" * 60)

    # 模拟一份很长的技术文档
    LONG_DOC = """
    # PostgreSQL 性能调优完全指南（示例文档）

    ## 1. 索引优化
    索引是提升查询性能最直接的手段。常见索引类型：
    - B-Tree 索引：适合等值查询、范围查询、排序
    - Hash 索引：仅适合等值查询，不支持范围查询
    - GIN 索引：适合数组、JSONB、全文搜索
    - GiST 索引：适合几何数据、全文搜索

    创建索引时需考虑：
    - 写多读少的表避免过多索引
    - 联合索引的列顺序影响查询计划
    - 部分索引（Partial Index）减少索引大小

    ## 2. VACUUM 与 AUTOVACUUM
    PostgreSQL 使用 MVCC（多版本并发控制），旧版本数据（dead tuples）需要 VACUUM 清理。
    关键参数：
    - autovacuum_vacuum_threshold: 默认 50，触发 vacuum 的最少 dead tuples 数
    - autovacuum_vacuum_scale_factor: 默认 0.2，触发 vacuum 的比例阈值
    - autovacuum_analyze_threshold / scale_factor: 触发 analyze 的阈值

    ## 3. 连接池
    PostgreSQL 的进程模型使每个连接消耗约 5-10MB 内存。
    生产环境必须使用连接池（PgBouncer 或应用层连接池）。
    - PgBouncer Transaction 模式：最高效，但不支持 session-level 特性
    - PgBouncer Session 模式：兼容性好，性能略低

    ## 4. 慢查询分析
    启用 pg_stat_statements 扩展，追踪最慢的 SQL：
    SELECT query, mean_exec_time, calls FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;

    EXPLAIN ANALYZE 查看执行计划：
    - Seq Scan：全表扫描，数据量大时需优化
    - Index Scan：索引扫描，通常高效
    - Hash Join / Nested Loop：连接算法选择
    """ * 5  # 扩充至足够长度

    QUESTIONS = [
        "如何判断什么时候需要创建 GIN 索引？",
        "autovacuum 的默认参数适合高写入场景吗？需要怎么调整？",
        "PgBouncer Transaction 模式有什么限制？",
    ]

    # 将长文档放在 user 消息中，并设置缓存断点
    for i, question in enumerate(QUESTIONS):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"以下是一份 PostgreSQL 调优文档：\n\n{LONG_DOC}",
                        "cache_control": {"type": "ephemeral"}   # ← 缓存长文档
                    },
                    {
                        "type": "text",
                        "text": f"\n基于以上文档，请回答：{question}"
                    }
                ]
            }
        ]

        resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            messages=messages,
        )

        u = resp.usage
        cache_hit = getattr(u, 'cache_read_input_tokens', 0)
        print(f"\n问题 {i+1}：{question}")
        print(f"  缓存命中 tokens：{cache_hit} | 输出 tokens：{u.output_tokens}")
        print(f"  回答：{resp.content[0].text[:200]}...")


# =============================================================================
# Part 3：多轮对话缓存策略 —— 缓存断点放在历史消息末尾
# =============================================================================

def demo_conversation_cache():
    """
    多轮对话场景的缓存策略：
    将 cache_control 放在最后一轮用户消息前面的历史尾部，
    这样每轮新消息加入时，历史部分可以被缓存复用。
    """

    print("\n" + "=" * 60)
    print("Part 3：多轮对话的缓存策略")
    print("=" * 60)

    # 模拟已有的对话历史（多轮后会很长）
    history = []

    turns = [
        "请介绍一下 Kubernetes 的核心概念。",
        "Pod 和 Deployment 有什么区别？",
        "HPA 自动扩缩容是如何工作的？",
    ]

    for turn_idx, question in enumerate(turns):
        # 构造带缓存的消息列表：
        # 策略：在最后一条历史消息上打 cache_control，新问题不打
        messages_with_cache = []
        for idx, msg in enumerate(history):
            if idx == len(history) - 1 and msg["role"] == "user":
                # 最后一条历史 user 消息加缓存断点
                messages_with_cache.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": msg["content"],
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                })
            else:
                messages_with_cache.append(msg)

        # 添加新问题
        messages_with_cache.append({"role": "user", "content": question})

        resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=[{
                "type": "text",
                "text": "你是一位 Kubernetes 专家。",
                "cache_control": {"type": "ephemeral"}
            }],
            messages=messages_with_cache,
        )

        answer = resp.content[0].text
        u = resp.usage
        cache_hit = getattr(u, 'cache_read_input_tokens', 0)

        print(f"\n第 {turn_idx + 1} 轮：{question}")
        print(f"  缓存命中 tokens：{cache_hit} | 输入 tokens：{u.input_tokens}")
        print(f"  回答（前150字）：{answer[:150]}...")

        # 将对话加入历史
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


# =============================================================================
# 主入口
# =============================================================================

def main():
    demo_system_prompt_cache()
    demo_message_cache()
    demo_conversation_cache()


if __name__ == "__main__":
    main()