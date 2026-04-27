"""
主题：重试机制与降级策略 —— 构建高可用的 LLM 应用

学习目标：
  1. 理解为什么需要重试（网络波动、API 限流、临时错误）
  2. 掌握 retry_decorator（指数退避重试）
  3. 掌握 fallback 链（主模型失败时切换到备用模型）
  4. 掌握超时控制和并发限制
  5. 学会设计优雅降级策略（LLM 不可用时的备选方案）

核心概念：
  Retry = 自动重试失败的请求（带退避策略）
  Fallback = 主路径失败时切换到备用路径
  Graceful Degradation = 逐步降低功能等级，而非完全崩溃

  生产级应用的可靠性金字塔：
  第 1 层：Retry（处理临时故障）
  第 2 层：Fallback（切换备用服务）
  第 3 层：Degradation（降级到简单逻辑）
  第 4 层：Circuit Breaker（熔断保护）

前置知识：已完成 01_lc_basics/05_error_handling.py
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
import time
from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL_CLAUDE = "ppio/pa/claude-sonnet-4-6"


# =============================================================================
# Part 1：Retry 重试机制
# =============================================================================

def demo_retry_with_backoff():
    """
    使用 tenacity 库实现指数退避重试。
    
    指数退避策略：
    - 第 1 次失败：等待 1s
    - 第 2 次失败：等待 2s
    - 第 3 次失败：等待 4s
    - ...
    
    优点：避免频繁重试导致雪崩效应
    适用场景：网络抖动、API 临时限流
    """
    print("=" * 60)
    print("Part 1: Retry with Exponential Backoff")
    print("=" * 60)
    
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )
    
    # 定义重试装饰器
    @retry(
        stop=stop_after_attempt(3),  # 最多重试 3 次
        wait=wait_exponential(multiplier=1, min=1, max=10),  # 指数退避：1s, 2s, 4s...
        retry=retry_if_exception_type(Exception),  # 捕获所有异常
        reraise=True,  # 最后一次失败时抛出异常
    )
    def call_llm_with_retry(question: str) -> str:
        """带重试的 LLM 调用"""
        llm = ChatAnthropic(model=MODEL_CLAUDE, max_tokens=128)
        prompt = ChatPromptTemplate.from_template("回答：{question}")
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question})
    
    # 正常情况（无需重试）
    print("\n【测试 1】正常调用（预期成功）")
    try:
        result = call_llm_with_retry("Python 是什么？")
        print(f"  结果：{result[:60]}...")
        print(f"  ✅ 成功（无需重试）")
    except Exception as e:
        print(f"  ❌ 失败：{e}")
    
    # 模拟失败场景
    print("\n【测试 2】模拟 API 失败（展示重试过程）")
    
    attempt_count = 0
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type(ConnectionError),
        reraise=True,
    )
    def simulate_api_call():
        nonlocal attempt_count
        attempt_count += 1
        print(f"  尝试第 {attempt_count} 次...")
        
        if attempt_count < 3:
            raise ConnectionError("模拟网络错误")
        
        return "成功响应"
    
    try:
        result = simulate_api_call()
        print(f"  ✅ 最终成功：{result}")
    except Exception as e:
        print(f"  ❌ 所有重试均失败：{e}")


# =============================================================================
# Part 2：Fallback 链 —— 主备模型切换
# =============================================================================

def demo_fallback_chain():
    """
    Fallback 链允许在主模型失败时自动切换到备用模型。
    
    典型配置：
    主模型：Claude Sonnet（高质量，但可能限流）
    备用 1：GPT-4 Turbo（中等质量）
    备用 2：本地小模型（低质量，但稳定）
    
    使用场景：
    - API 配额耗尽
    - 某个模型服务商宕机
    - 成本控制（优先用便宜模型）
    """
    print("\n" + "=" * 60)
    print("Part 2: Fallback Chain —— 主备模型切换")
    print("=" * 60)
    
    # 创建主模型和备用模型
    primary_llm = ChatAnthropic(
        model=MODEL_CLAUDE,
        max_tokens=128,
    )
    
    # 备用模型（需要配置 OPENAI_API_KEY）
    fallback_llm = None
    if os.getenv("OPENAI_API_KEY"):
        fallback_llm = ChatOpenAI(
            model="gpt-4o-mini",
            max_tokens=128,
        )
        print("  主模型：Claude Sonnet")
        print("  备用模型：GPT-4o Mini")
    else:
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过备用模型测试")
        print("  [提示] 在 .env 中添加 OPENAI_API_KEY 以启用此功能")
        return
    
    # 构建带 fallback 的链
    prompt = ChatPromptTemplate.from_template("用一句话解释：{topic}")
    output_parser = StrOutputParser()
    
    # 方式 1：在 LLM 级别设置 fallback
    llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])
    chain = prompt | llm_with_fallback | output_parser
    
    print("\n【测试】正常情况（主模型可用）")
    try:
        result = chain.invoke({"topic": "机器学习"})
        print(f"  结果：{result}")
        print(f"  ✅ 使用主模型成功")
    except Exception as e:
        print(f"  ❌ 失败：{e}")
    
    # 方式 2：在 Chain 级别设置 fallback
    print("\n【进阶】Chain 级别的 Fallback")
    
    # 主链（复杂逻辑）
    main_chain = prompt | primary_llm | output_parser
    
    # 备用链（简化逻辑）
    simple_prompt = ChatPromptTemplate.from_template("简要说明：{topic}")
    fallback_chain = simple_prompt | fallback_llm | output_parser
    
    # 组合
    robust_chain = main_chain.with_fallbacks([fallback_chain])
    
    print("  主链：详细解释")
    print("  备用链：简要说明")
    print(f"  ✅ Fallback 链已配置")


# =============================================================================
# Part 3：超时控制与并发限制
# =============================================================================

def demo_timeout_and_rate_limiting():
    """
    生产环境必须设置超时和速率限制，防止：
    1. 单个请求卡死整个系统
    2. 超出 API 配额导致账号被封
    3. 资源耗尽影响其他用户
    """
    print("\n" + "=" * 60)
    print("Part 3: Timeout & Rate Limiting")
    print("=" * 60)
    
    # 设置超时
    print("\n【配置 1】请求超时控制")
    llm_with_timeout = ChatAnthropic(
        model=MODEL_CLAUDE,
        max_tokens=128,
        timeout=30,  # 30 秒超时
    )
    print(f"  超时时间：30 秒")
    print(f"  ✅ 超时配置完成")
    
    # 并发限制
    print("\n【配置 2】并发请求限制")
    import asyncio
    from asyncio import Semaphore
    
    # 限制同时最多 5 个请求
    semaphore = Semaphore(5)
    
    async def limited_llm_call(question: str):
        async with semaphore:
            llm = ChatAnthropic(model=MODEL_CLAUDE, max_tokens=64)
            return await llm.ainvoke(question)
    
    print(f"  最大并发数：5")
    print(f"  ✅ 并发限制已配置")
    
    # 批量调用示例
    print("\n【测试】批量调用（带并发控制）")
    
    async def batch_with_limit():
        questions = [f"问题 {i}" for i in range(10)]
        tasks = [limited_llm_call(q) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        print(f"  总请求数：{len(questions)}")
        print(f"  成功数：{success_count}")
        print(f"  失败数：{len(questions) - success_count}")
    
    # 运行异步测试
    asyncio.run(batch_with_limit())


# =============================================================================
# Part 4：优雅降级策略
# =============================================================================

def demo_graceful_degradation():
    """
    当 LLM 完全不可用时，提供降级方案保证基本功能。
    
    降级层次：
    Level 0：完整 LLM 推理（最佳体验）
    Level 1：缓存结果（稍旧但可用）
    Level 2：规则引擎（基于关键词匹配）
    Level 3：静态回复（友好提示）
    """
    print("\n" + "=" * 60)
    print("Part 4: Graceful Degradation —— 优雅降级")
    print("=" * 60)
    
    class RobustQASystem:
        """健壮的问答系统，支持多级降级"""
        
        def __init__(self):
            self.llm = ChatAnthropic(model=MODEL_CLAUDE, max_tokens=128, timeout=10)
            self.cache = {}  # 简单内存缓存
            self.faq_db = {  # FAQ 知识库
                "python": "Python 是一种高级编程语言，以简洁易读著称。",
                "langchain": "LangChain 是构建 LLM 应用的框架。",
            }
        
        def answer(self, question: str) -> dict:
            """
            尝试多个层级，返回最佳答案和使用的层级
            """
            # Level 0：尝试 LLM
            try:
                print(f"  [Level 0] 调用 LLM...")
                prompt = ChatPromptTemplate.from_template("回答：{q}")
                chain = prompt | self.llm | StrOutputParser()
                result = chain.invoke({"q": question}, config={"timeout": 10})
                return {"answer": result, "level": "LLM", "status": "success"}
            except Exception as e:
                print(f"  [Level 0] LLM 失败：{str(e)[:50]}")
            
            # Level 1：检查缓存
            cache_key = question.lower().strip()
            if cache_key in self.cache:
                print(f"  [Level 1] 使用缓存结果")
                return {"answer": self.cache[cache_key], "level": "Cache", "status": "degraded"}
            
            # Level 2：FAQ 匹配
            for keyword, answer in self.faq_db.items():
                if keyword in question.lower():
                    print(f"  [Level 2] FAQ 匹配")
                    return {"answer": answer, "level": "FAQ", "status": "degraded"}
            
            # Level 3：默认回复
            print(f"  [Level 3] 返回默认回复")
            return {
                "answer": "抱歉，暂时无法回答您的问题。请稍后重试或联系人工客服。",
                "level": "Default",
                "status": "minimal",
            }
    
    # 测试降级系统
    qa_system = RobustQASystem()
    
    print("\n【测试 1】正常问题（预期 Level 0）")
    result = qa_system.answer("什么是 Python？")
    print(f"  层级：{result['level']}")
    print(f"  状态：{result['status']}")
    print(f"  回答：{result['answer'][:60]}...")
    
    print("\n【测试 2】FAQ 问题（预期 Level 2）")
    result = qa_system.answer("告诉我关于 langchain 的信息")
    print(f"  层级：{result['level']}")
    print(f"  状态：{result['status']}")
    print(f"  回答：{result['answer']}")
    
    print("\n【测试 3】未知问题（预期 Level 3）")
    result = qa_system.answer("量子计算机如何工作？")
    print(f"  层级：{result['level']}")
    print(f"  状态：{result['status']}")
    print(f"  回答：{result['answer']}")


# =============================================================================
# Part 5：生产环境配置清单
# =============================================================================

def demo_production_checklist():
    """
    生产环境部署前的可靠性检查清单。
    """
    print("\n" + "=" * 60)
    print("Part 5: Production Checklist")
    print("=" * 60)
    
    checklist = """
  ✅ 必配项（Missing any = Critical Issue）：
  ──────────────────────────────────────────────
  □ 超时控制：所有 LLM 调用设置 timeout（建议 30-60s）
  □ 重试机制：关键路径配置 retry（3 次以内，指数退避）
  □ Fallback：至少一个备用模型或服务
  □ 错误日志：记录所有失败请求的详细信息
  □ 监控告警：API 错误率 > 5% 时触发告警
  
  ✅ 推荐项（Should have for production）：
  ──────────────────────────────────────────────
  □ 缓存层：高频问题启用缓存（TTL 1-24 小时）
  □ 速率限制：防止超出 API 配额
  □ 降级策略：LLM 不可用时的备选方案
  □ 熔断器：连续失败时快速失败（避免浪费资源）
  □ 健康检查：定期检查依赖服务状态
  
  ✅ 优化项（Nice to have）：
  ──────────────────────────────────────────────
  □ A/B 测试：对比不同模型的性价比
  □ 成本监控：实时追踪 token 消耗和费用
  □ 性能分析：识别慢请求和瓶颈节点
  □ 灰度发布：新配置先小范围验证
  □ 自动化回滚：检测到异常时自动恢复
  
  📊 关键指标（Monitor these metrics）：
  ──────────────────────────────────────────────
  - P95 延迟：< 5 秒（用户体验阈值）
  - 成功率：> 95%（低于 90% 需紧急处理）
  - 缓存命中率：> 30%（FAQ 场景应 > 60%）
  - API 错误率：< 5%
  - Token 成本：设定预算上限并监控
    """
    print(checklist)


def main():
    print("=" * 60)
    print("重试机制与降级策略详解")
    print("=" * 60)
    
    demo_retry_with_backoff()
    demo_fallback_chain()
    demo_timeout_and_rate_limiting()
    demo_graceful_degradation()
    demo_production_checklist()
    
    print("\n" + "=" * 60)
    print("学习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
