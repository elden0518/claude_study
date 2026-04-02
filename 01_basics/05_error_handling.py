"""
05_error_handling.py - Claude API 错误处理演示

涵盖知识点：
  1. 错误类型层级（APIError 及其子类）
  2. 指数退避重试（针对 RateLimitError）
  3. 超时配置（httpx.Timeout）
  4. 触发各类错误的演示方式
  5. 生产环境最佳实践（注释形式说明）

运行：.venv/Scripts/python.exe 01_basics/05_error_handling.py
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
import time
import functools
from dotenv import load_dotenv
import httpx
import anthropic

load_dotenv()

MODEL = "ppio/pa/claude-sonnet-4-6"
VALID_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# --------------------------------------------------
# 辅助函数：构造最小化的 httpx.Request / httpx.Response
# --------------------------------------------------

def _dummy_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _dummy_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_dummy_request())


# --------------------------------------------------
# 1. 错误类型层级演示
# --------------------------------------------------

def demo_error_types():
    """演示捕获各种 Anthropic 异常类型。"""
    print("\n" + "=" * 60)
    print("[1] 错误类型层级演示")
    print("=" * 60)

    # --- AuthenticationError（401）---
    print("\n[1-1] AuthenticationError（无效 API Key / 401）")
    print("  触发方式：anthropic.Anthropic(api_key='invalid_key').messages.create(...)")
    try:
        # 模拟使用无效 API Key 时抛出的错误：
        #   bad_client = anthropic.Anthropic(api_key="invalid_key")
        #   bad_client.messages.create(...)  -> 401 AuthenticationError
        raise anthropic.AuthenticationError(
            message="无效的 API Key，请检查 ANTHROPIC_API_KEY。",
            response=_dummy_response(401),
            body={"error": {"type": "authentication_error",
                            "message": "Invalid API Key"}},
        )
    except anthropic.AuthenticationError as e:
        print(f"  捕获到：{type(e).__name__}")
        print(f"  HTTP 状态码：{e.status_code}")
        print(f"  错误信息：{e.message}")
        # 最佳实践：确认 ANTHROPIC_API_KEY 环境变量已正确设置；切勿硬编码密钥
        print("  最佳实践：确认 ANTHROPIC_API_KEY 正确设置；切勿在代码中硬编码密钥")

    # --- RateLimitError（429）---
    print("\n[1-2] RateLimitError（超出速率限制 / 429）")
    try:
        raise anthropic.RateLimitError(
            message="已超出速率限制，请 60 秒后重试。",
            response=_dummy_response(429),
            body={"error": {"type": "rate_limit_error"}},
        )
    except anthropic.RateLimitError as e:
        print(f"  捕获到：{type(e).__name__}")
        print(f"  HTTP 状态码：{e.status_code}")
        print(f"  错误信息：{e.message}")
        # 最佳实践：使用指数退避重试；见 demo_retry_backoff()
        print("  最佳实践：使用指数退避重试；降低并发请求频率")

    # --- BadRequestError（400）---
    print("\n[1-3] BadRequestError（请求参数无效 / 400）")
    print("  触发方式：messages.create(max_tokens=0, ...) - 非法参数")
    try:
        # 等价于：client.messages.create(model=MODEL, max_tokens=0, ...)
        raise anthropic.BadRequestError(
            message="max_tokens 的值必须 >= 1。",
            response=_dummy_response(400),
            body={"error": {"type": "invalid_request_error",
                            "message": "max_tokens: value must be >= 1"}},
        )
    except anthropic.BadRequestError as e:
        print(f"  捕获到：{type(e).__name__}")
        print(f"  HTTP 状态码：{e.status_code}")
        print(f"  错误信息：{e.message}")
        # 最佳实践：发送前在客户端侧校验参数
        print("  最佳实践：发送前校验 max_tokens >= 1、模型名称等参数")

    # --- APIStatusError（其他 4xx/5xx）---
    print("\n[1-4] APIStatusError（HTTP 错误兜底，如 500）")
    try:
        raise anthropic.InternalServerError(
            message="服务器内部错误，请重试。",
            response=_dummy_response(500),
            body={"error": {"type": "api_error",
                            "message": "Internal server error"}},
        )
    except anthropic.APIStatusError as e:
        # InternalServerError 是 APIStatusError 的子类
        print(f"  捕获到：{type(e).__name__}（APIStatusError 的子类）")
        print(f"  HTTP 状态码：{e.status_code}")
        print(f"  错误信息：{e.message}")
        # 最佳实践：4xx -> 修正请求参数；5xx -> 等待后重试
        print("  最佳实践：4xx -> 修正请求参数；5xx -> 等待后重试")

    # --- APIConnectionError（网络故障）---
    print("\n[1-5] APIConnectionError（网络/连接故障）")
    try:
        raise anthropic.APIConnectionError(
            message="连接错误：无法建立新连接。",
            request=_dummy_request(),
        )
    except anthropic.APIConnectionError as e:
        print(f"  捕获到：{type(e).__name__}")
        print(f"  错误信息：{e}")
        # 最佳实践：检查网络/代理；向用户展示友好的"服务不可用"提示
        print("  最佳实践：检查网络/代理；向用户展示友好的「服务不可用」提示")

    # --- APIError（所有 Anthropic 异常的基类）---
    print("\n[1-6] APIError（基类；可捕获所有 Anthropic 异常）")
    try:
        raise anthropic.APIError(
            message="示例：APIError 基类演示。",
            request=_dummy_request(),
            body=None,
        )
    except anthropic.APIError as e:
        print(f"  捕获到：{type(e).__name__}")
        print(f"  错误信息：{e.message}")
        # 最佳实践：将 APIError 作为最外层兜底捕获；记录完整日志
        print("  最佳实践：将 APIError 作为最外层兜底捕获；记录结构化日志")

    # --- 异常层级汇总 ---
    print("\n  [异常层级]")
    print("  anthropic.APIError                   <- 所有 Anthropic 异常的基类")
    print("  +- anthropic.APIStatusError          <- 带 HTTP 状态码的错误")
    print("  |   +- anthropic.AuthenticationError   (401)")
    print("  |   +- anthropic.PermissionDeniedError (403)")
    print("  |   +- anthropic.NotFoundError         (404)")
    print("  |   +- anthropic.RateLimitError         (429)")
    print("  |   +- anthropic.BadRequestError        (400)")
    print("  |   +- anthropic.InternalServerError    (500)")
    print("  +- anthropic.APIConnectionError      <- 网络/超时错误")


# --------------------------------------------------
# 2. 指数退避重试
# --------------------------------------------------

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """
    装饰器：遇到 RateLimitError 时自动指数退避重试。

    重试等待时间：1s -> 2s -> 4s（base_delay * 2^attempt）
    达到最大重试次数后重新抛出原始异常。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except anthropic.RateLimitError as e:
                    last_exc = e
                    if attempt < max_retries:
                        wait = base_delay * (2 ** attempt)  # 1 / 2 / 4 秒
                        print(f"  [限速] 第 {attempt + 1} 次请求被限速；"
                              f"{wait:.1f}s 后重试...")
                        time.sleep(wait)
                    else:
                        print(f"  [限速] 已耗尽 {max_retries} 次重试机会，放弃。")
            raise last_exc
        return wrapper
    return decorator


def demo_retry_backoff():
    """指数退避重试模式演示。"""
    print("\n" + "=" * 60)
    print("[2] 指数退避重试演示")
    print("=" * 60)

    call_count = 0

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def flaky_request():
        """模拟前两次失败、第三次成功的请求。"""
        nonlocal call_count
        call_count += 1
        print(f"  -> 第 {call_count} 次尝试")
        if call_count <= 2:
            raise anthropic.RateLimitError(
                message=f"模拟限速（第 {call_count} 次）",
                response=_dummy_response(429),
                body=None,
            )
        return "请求成功！"

    print("\n场景 A：前 2 次被限速，第 3 次成功（等待 1s / 2s）")
    try:
        result = flaky_request()
        print(f"  最终结果：{result}")
        # 最佳实践：记录重试成功日志；监控重试频率
        print("  最佳实践：监控重试频率；频繁重试时应降低并发量")
    except anthropic.RateLimitError as e:
        print(f"  超出重试次数，放弃：{e.message}")

    print("\n场景 B：始终失败，耗尽重试次数（0.1s 延迟，快速演示）")
    call_count = 0

    @retry_with_backoff(max_retries=2, base_delay=0.1)
    def always_fail():
        nonlocal call_count
        call_count += 1
        print(f"  -> 第 {call_count} 次尝试（始终失败）")
        raise anthropic.RateLimitError(
            message="持续限速",
            response=_dummy_response(429),
            body=None,
        )

    try:
        always_fail()
    except anthropic.RateLimitError as e:
        print(f"  重试耗尽后捕获：{type(e).__name__} - {e.message}")
        # 最佳实践：告知用户"请求过于频繁，请稍后重试"
        print("  最佳实践：告知用户「请求过于频繁，请稍后重试」")


# --------------------------------------------------
# 3. 超时配置
# --------------------------------------------------

def demo_timeout():
    """使用 httpx.Timeout 配置超时。"""
    print("\n" + "=" * 60)
    print("[3] 超时配置演示")
    print("=" * 60)

    # httpx.Timeout 各字段说明：
    #   connect  - 建立 TCP 连接的最大等待时间
    #   read     - 接收响应数据的最大等待时间（流式响应应适当增大）
    #   write    - 发送请求体的最大等待时间
    #   pool     - 从连接池获取连接的最大等待时间
    timeout_config = httpx.Timeout(
        connect=5.0,    # 5s 建立连接
        read=30.0,      # 30s 接收响应（流式响应时适当增大）
        write=10.0,     # 10s 发送请求体
        pool=5.0,       # 5s 从连接池获取连接
    )

    print(f"\n  推荐超时配置：")
    print(f"    connect = {timeout_config.connect}s  （TCP 连接）")
    print(f"    read    = {timeout_config.read}s （响应数据；流式响应时适当增大）")
    print(f"    write   = {timeout_config.write}s （发送请求体）")
    print(f"    pool    = {timeout_config.pool}s  （从连接池获取连接）")

    # 演示如何将超时配置传入 Anthropic 客户端
    client_with_timeout = anthropic.Anthropic(
        api_key=VALID_API_KEY or "dummy",
        timeout=timeout_config,
    )
    print(f"\n  已配置超时的客户端，timeout 类型：{type(client_with_timeout.timeout).__name__}")

    # 演示超短超时（0.001s）触发 APIConnectionError 或 APITimeoutError
    print("\n  演示：极短超时（0.001s）-> 触发 APIConnectionError 或 APITimeoutError：")
    try:
        short_timeout_client = anthropic.Anthropic(
            api_key=VALID_API_KEY or "dummy",
            base_url="http://127.0.0.1:19999",  # 不存在的本地端口
            timeout=httpx.Timeout(connect=0.001, read=0.001,
                                  write=0.001, pool=0.001),
        )
        short_timeout_client.messages.create(
            model=MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )
    except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
        print(f"  捕获到：{type(e).__name__}")
        print(f"  错误信息（截断）：{str(e)[:80]}")
    except Exception as e:
        print(f"  捕获到底层异常：{type(e).__name__}: {str(e)[:80]}")

    print("\n  最佳实践：")
    print("  - 流式响应场景，read 超时建议设置为 60s 以上")
    print("  - 批量/长文本任务，read 超时可设置为 120s+")
    print("  - connect 超时：5~10s 通常已足够")
    print("  - 网络问题应同时捕获 APIConnectionError 和 APITimeoutError")


# --------------------------------------------------
# 4. 优雅降级
# --------------------------------------------------

def demo_graceful_degradation():
    """优雅降级：出错时返回友好提示，程序不崩溃。"""
    print("\n" + "=" * 60)
    print("[4] 优雅降级演示")
    print("=" * 60)

    FALLBACK = "抱歉，服务暂时不可用，请稍后重试。"

    def safe_ask(client: anthropic.Anthropic, question: str) -> str:
        """
        生产环境可用的封装函数：
        - 从最具体到最通用依次捕获异常
        - 出错时返回兜底文案，而非抛出异常导致崩溃
        - 生产环境中应使用 logging 模块记录结构化日志
        """
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=64,
                messages=[{"role": "user", "content": question}],
            )
            return response.content[0].text

        except anthropic.AuthenticationError as e:
            # 最佳实践：立即告警；生产环境中此错误应极少出现
            print(f"  [ERROR] AuthenticationError: {e.message}")
            print("          -> 请检查 ANTHROPIC_API_KEY 配置")
            return FALLBACK

        except anthropic.RateLimitError as e:
            # 最佳实践：降低并发或等待后重试
            print(f"  [WARN]  RateLimitError: {e.message}")
            print("          -> 降低请求频率或使用退避重试")
            return FALLBACK

        except anthropic.BadRequestError as e:
            # 最佳实践：这是客户端代码问题，需修复代码
            print(f"  [ERROR] BadRequestError (400): {e.message}")
            print("          -> 修正请求参数")
            return FALLBACK

        except anthropic.APIConnectionError as e:
            # 最佳实践：检查网络/代理；可短暂等待后重试
            print(f"  [ERROR] APIConnectionError: {e}")
            print("          -> 检查网络或代理配置")
            return FALLBACK

        except anthropic.APIStatusError as e:
            # 兜底捕获其他 HTTP 错误
            print(f"  [ERROR] APIStatusError ({e.status_code}): {e.message}")
            print("          -> 4xx：修正参数；5xx：等待后重试")
            return FALLBACK

        except anthropic.APIError as e:
            # 最终兜底，捕获意外的 Anthropic 异常
            print(f"  [ERROR] APIError（未预期）: {e.message}")
            return FALLBACK

    # --- 场景 A：AuthenticationError 优雅降级 ---
    print("\n  场景 A：AuthenticationError -> 优雅降级")
    try:
        raise anthropic.AuthenticationError(
            message="401 无效的 API Key",
            response=_dummy_response(401),
            body=None,
        )
    except anthropic.AuthenticationError as e:
        print(f"  [ERROR] AuthenticationError: {e.message}")
        print(f"  返回给用户：{FALLBACK!r}")
        print("  程序未崩溃！")

    # --- 场景 B：RateLimitError 优雅降级 ---
    print("\n  场景 B：RateLimitError -> 优雅降级")
    try:
        raise anthropic.RateLimitError(
            message="429 超出速率限制",
            response=_dummy_response(429),
            body=None,
        )
    except anthropic.RateLimitError as e:
        print(f"  [WARN]  RateLimitError: {e.message}")
        print(f"  返回给用户：{FALLBACK!r}")
        print("  程序未崩溃！")

    # --- 场景 C：使用有效 API Key 发送真实请求 ---
    print("\n  场景 C：使用有效 API Key 发送真实请求")
    if VALID_API_KEY:
        real_client = anthropic.Anthropic(api_key=VALID_API_KEY)
        answer = safe_ask(real_client, "用一句话介绍 Python。")
        print(f"  Claude 回复：{answer!r}")
    else:
        print("  [跳过] 未设置 ANTHROPIC_API_KEY")

    # --- 最佳实践总结 ---
    print("\n  最佳实践总结：")
    print("  - 优先捕获具体子类；将 APIError 作为最外层兜底")
    print("  - 4xx 错误通常是代码/配置问题，应通知开发者")
    print("  - 5xx 和 ConnectionError 可重试；使用指数退避")
    print("  - 向用户展示友好提示；切勿暴露内部错误详情")
    print("  - 生产环境使用 logging 模块进行结构化日志记录")
    print("  - 添加监控告警（如：AuthenticationError 频繁出现时告警）")


# --------------------------------------------------
# 主函数
# --------------------------------------------------

def main():
    print("=" * 60)
    print(" Claude API 错误处理演示")
    print("=" * 60)

    demo_error_types()
    demo_retry_backoff()
    demo_timeout()
    demo_graceful_degradation()

    print("\n" + "=" * 60)
    print(" 所有演示已完成，程序正常退出。")
    print("=" * 60)


if __name__ == "__main__":
    main()