"""
主题：缓存层 —— 减少重复 LLM 调用，降低成本和延迟

学习目标：
  1. 理解为什么需要缓存（成本优化、降低延迟）
  2. 掌握 InMemoryCache（开发环境快速测试）
  3. 掌握 SQLiteCache（持久化缓存，适合生产）
  4. 掌握 GPTCache（语义缓存，相似问题复用）
  5. 理解缓存失效策略和 TTL（Time To Live）

核心概念：
  精确缓存 = 完全相同的输入 → 返回缓存结果
  语义缓存 = 语义相似的输入 → 返回缓存结果（更智能但可能有偏差）

  适用场景：
  - FAQ 问答系统（大量重复问题）
  - 代码补全（常见模式复用）
  - 数据转换任务（相同输入总是产生相同输出）

前置知识：已完成 01_lc_basics/03_lcel_chains.py
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.caches import InMemoryCache

# LangChain 1.x 使用 set_global_handler 或直接配置
# 对于缓存，推荐使用 with_config 或在 chain 级别配置

MODEL = "ppio/pa/claude-sonnet-4-6"


# =============================================================================
# Part 1：InMemoryCache —— 内存缓存（开发环境）
# =============================================================================

def demo_in_memory_cache():
    """
    InMemoryCache 是最简单的缓存实现，数据存储在 Python 字典中。
    
    优点：零配置、速度快
    缺点：进程重启后丢失、不适合分布式部署
    
    使用方式：
      1. 在 chain.invoke() 时通过 config 传入
      2. 或使用 with_config 绑定缓存
    """
    print("=" * 60)
    print("Part 1: InMemoryCache —— 内存缓存")
    print("=" * 60)
    
    # 创建缓存实例
    cache = InMemoryCache()
    
    llm = ChatAnthropic(model=MODEL, max_tokens=128)
    prompt = ChatPromptTemplate.from_template("用一句话解释：{topic}")
    chain = prompt | llm | StrOutputParser()
    
    question = {"topic": "什么是 Python？"}
    
    # 第一次调用（未命中缓存，会调用 LLM）
    print("\n【第 1 次调用】未命中缓存，调用 LLM...")
    start = time.time()
    result1 = chain.invoke(question, config={"cache": cache})
    elapsed1 = time.time() - start
    print(f"  结果：{result1}")
    print(f"  耗时：{elapsed1:.2f}s")
    
    # 第二次调用（命中缓存，直接返回）
    print("\n【第 2 次调用】命中缓存，直接返回...")
    start = time.time()
    result2 = chain.invoke(question, config={"cache": cache})
    elapsed2 = time.time() - start
    print(f"  结果：{result2}")
    print(f"  耗时：{elapsed2:.4f}s（快了 {elapsed1/elapsed2:.0f}x）")
    
    # 验证结果一致性
    print(f"\n  结果一致：{result1 == result2}")
    
    print("\n  [提示] 缓存仅在 config 中生效，无需清除")


# =============================================================================
# Part 2：SQLiteCache —— 持久化缓存（生产环境）
# =============================================================================

def demo_sqlite_cache():
    """
    SQLiteCache 将缓存数据保存到本地 SQLite 数据库文件。
    
    优点：持久化、支持进程重启、可查询缓存统计
    缺点：单机部署、并发性能有限
    
    适用场景：
    - 本地开发环境长期缓存
    - 小型生产应用
    - 需要缓存统计分析的场景
    """
    print("\n" + "=" * 60)
    print("Part 2: SQLiteCache —— 持久化缓存")
    print("=" * 60)
    
    from langchain_community.cache import SQLiteCache
    
    cache_path = "./.langchain_cache.db"
    cache = SQLiteCache(database_path=cache_path)
    
    llm = ChatAnthropic(model=MODEL, max_tokens=128)
    prompt = ChatPromptTemplate.from_template("列出 {language} 的三个特点")
    chain = prompt | llm | StrOutputParser()
    
    question = {"language": "Rust"}
    
    print(f"\n  缓存数据库：{cache_path}")
    
    # 第一次调用
    print("\n【第 1 次调用】写入缓存...")
    start = time.time()
    result1 = chain.invoke(question, config={"cache": cache})
    elapsed1 = time.time() - start
    print(f"  结果：{result1[:80]}...")
    print(f"  耗时：{elapsed1:.2f}s")
    
    # 第二次调用（命中缓存）
    print("\n【第 2 次调用】读取缓存...")
    start = time.time()
    result2 = chain.invoke(question, config={"cache": cache})
    elapsed2 = time.time() - start
    print(f"  结果：{result2[:80]}...")
    print(f"  耗时：{elapsed2:.4f}s")
    
    # 查看缓存文件大小
    if os.path.exists(cache_path):
        size_kb = os.path.getsize(cache_path) / 1024
        print(f"\n  缓存文件大小：{size_kb:.2f} KB")
    
    # 清理
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"  [清理] 已删除缓存文件")


# =============================================================================
# Part 3：自定义缓存键策略
# =============================================================================

def demo_custom_cache_key():
    """
    默认情况下，LangChain 使用完整的 prompt + model 参数作为缓存键。
    
    有时我们希望自定义缓存策略，例如：
    - 忽略某些无关参数（如 temperature 的微小差异）
    - 只缓存特定类型的请求
    - 添加业务维度的缓存键（如 user_id、session_id）
    """
    print("\n" + "=" * 60)
    print("Part 3: 自定义缓存策略")
    print("=" * 60)
    
    from langchain_community.cache import InMemoryCache
    
    custom_cache = InMemoryCache()
    
    # 手动管理缓存示例
    cache_key = "faq_what_is_python_v1"
    
    # 模拟缓存查找
    cached_result = custom_cache.lookup(cache_key)
    
    if cached_result is None:
        print("\n  [缓存未命中] 调用 LLM...")
        llm = ChatAnthropic(model=MODEL, max_tokens=64)
        result = llm.invoke("用一句话解释 Python")
        custom_cache.update(cache_key, result)
        print(f"  结果：{result.content}")
        print(f"  [已缓存] key={cache_key}")
    else:
        print(f"\n  [缓存命中] 直接返回")
        print(f"  结果：{cached_result.content}")
    
    # 第二次查找
    print("\n  【再次查询相同 key】")
    cached_result = custom_cache.lookup(cache_key)
    if cached_result:
        print(f"  [缓存命中] {cached_result.content}")


# =============================================================================
# Part 4：缓存最佳实践与注意事项
# =============================================================================

def demo_cache_best_practices():
    """
    缓存使用的最佳实践和常见陷阱。
    """
    print("\n" + "=" * 60)
    print("Part 4: 缓存最佳实践")
    print("=" * 60)
    
    print("""
  ✅ 适合缓存的场景：
  ──────────────────────────────────────────────
  1. 确定性任务：相同输入总是产生相同输出
     - 代码翻译、格式转换、数据提取
  
  2. 高频重复问题：FAQ、客服机器人
     - "如何重置密码？"、"退款政策是什么？"
  
  3. 昂贵的计算：复杂推理、长文档总结
     - 避免重复处理相同文档
  
  4. 离线批处理：批量生成内容
     - 失败重试时避免重复调用
  
  ❌ 不适合缓存的场景：
  ──────────────────────────────────────────────
  1. 时间敏感内容：新闻、天气、股票价格
     - 缓存会导致信息过时
  
  2. 个性化推荐：依赖用户历史、上下文
     - 不同用户看到不同结果
  
  3. 随机性任务：创意写作、头脑风暴
     - 需要多样性而非一致性
  
  4. 安全敏感操作：身份验证、权限检查
     - 必须实时验证
  
  ⚙️ 缓存配置建议：
  ──────────────────────────────────────────────
  1. 设置 TTL（Time To Live）：定期清理过期缓存
  2. 监控缓存命中率：评估缓存效果
  3. 版本化管理：prompt 变更时更新缓存键版本
  4. 限制缓存大小：防止内存溢出
  5. 灰度测试：新缓存策略先小范围验证
    """)


def main():
    print("=" * 60)
    print("LangChain 缓存层详解")
    print("=" * 60)
    
    demo_in_memory_cache()
    demo_sqlite_cache()
    demo_custom_cache_key()
    demo_cache_best_practices()
    
    print("\n" + "=" * 60)
    print("学习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
