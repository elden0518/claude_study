"""
05_error_handling.py - Claude API Error Handling Demo

Knowledge points covered:
  1. Error type hierarchy (APIError and subclasses)
  2. Exponential backoff retry (for RateLimitError)
  3. Timeout configuration (httpx.Timeout)
  4. Triggering each error type for demonstration
  5. Production best practices (as comments)

Run: .venv/Scripts/python.exe 01_basics/05_error_handling.py
"""

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
# Helpers: build minimal httpx.Request / httpx.Response
# --------------------------------------------------

def _dummy_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _dummy_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_dummy_request())


# --------------------------------------------------
# 1. Error type hierarchy demo
# --------------------------------------------------

def demo_error_types():
    """Demo catching each Anthropic exception type."""
    print("\n" + "=" * 60)
    print("[1] Error Type Hierarchy Demo")
    print("=" * 60)

    # --- AuthenticationError (401) ---
    print("\n[1-1] AuthenticationError (invalid API key / 401)")
    print("  Triggered by: anthropic.Anthropic(api_key='invalid_key').messages.create(...)")
    try:
        # Simulates the error raised when using an invalid API key:
        #   bad_client = anthropic.Anthropic(api_key="invalid_key")
        #   bad_client.messages.create(...)  -> 401 AuthenticationError
        raise anthropic.AuthenticationError(
            message="Invalid API Key. Please check your ANTHROPIC_API_KEY.",
            response=_dummy_response(401),
            body={"error": {"type": "authentication_error",
                            "message": "Invalid API Key"}},
        )
    except anthropic.AuthenticationError as e:
        print(f"  Caught: {type(e).__name__}")
        print(f"  HTTP status: {e.status_code}")
        print(f"  Message: {e.message}")
        # Best practice: verify ANTHROPIC_API_KEY env var; never hard-code keys
        print("  Best practice: check ANTHROPIC_API_KEY is set correctly; never hard-code keys")

    # --- RateLimitError (429) ---
    print("\n[1-2] RateLimitError (rate limit exceeded / 429)")
    try:
        raise anthropic.RateLimitError(
            message="Rate limit exceeded. Please retry after 60 seconds.",
            response=_dummy_response(429),
            body={"error": {"type": "rate_limit_error"}},
        )
    except anthropic.RateLimitError as e:
        print(f"  Caught: {type(e).__name__}")
        print(f"  HTTP status: {e.status_code}")
        print(f"  Message: {e.message}")
        # Best practice: exponential backoff retry; see demo_retry_backoff()
        print("  Best practice: use exponential backoff; reduce concurrent request rate")

    # --- BadRequestError (400) ---
    print("\n[1-3] BadRequestError (invalid request parameters / 400)")
    print("  Triggered by: messages.create(max_tokens=0, ...) - illegal parameter")
    try:
        # Equivalent to: client.messages.create(model=MODEL, max_tokens=0, ...)
        raise anthropic.BadRequestError(
            message="max_tokens: value must be >= 1.",
            response=_dummy_response(400),
            body={"error": {"type": "invalid_request_error",
                            "message": "max_tokens: value must be >= 1"}},
        )
    except anthropic.BadRequestError as e:
        print(f"  Caught: {type(e).__name__}")
        print(f"  HTTP status: {e.status_code}")
        print(f"  Message: {e.message}")
        # Best practice: validate params client-side before sending
        print("  Best practice: validate max_tokens >= 1, model name, etc. before sending")

    # --- APIStatusError (other 4xx/5xx) ---
    print("\n[1-4] APIStatusError (catch-all for HTTP errors, e.g. 500)")
    try:
        raise anthropic.InternalServerError(
            message="Internal server error. Please retry your request.",
            response=_dummy_response(500),
            body={"error": {"type": "api_error",
                            "message": "Internal server error"}},
        )
    except anthropic.APIStatusError as e:
        # InternalServerError is a subclass of APIStatusError
        print(f"  Caught: {type(e).__name__} (subclass of APIStatusError)")
        print(f"  HTTP status: {e.status_code}")
        print(f"  Message: {e.message}")
        # Best practice: 4xx -> fix request params; 5xx -> wait and retry
        print("  Best practice: 4xx -> fix request params; 5xx -> wait then retry")

    # --- APIConnectionError (network failure) ---
    print("\n[1-5] APIConnectionError (network/connection failure)")
    try:
        raise anthropic.APIConnectionError(
            message="Connection error: Failed to establish a new connection.",
            request=_dummy_request(),
        )
    except anthropic.APIConnectionError as e:
        print(f"  Caught: {type(e).__name__}")
        print(f"  Message: {e}")
        # Best practice: check network/proxy; show user-friendly "service unavailable"
        print("  Best practice: check network/proxy; show friendly 'service unavailable' message")

    # --- APIError (base class for all Anthropic exceptions) ---
    print("\n[1-6] APIError (base class; catches all Anthropic exceptions)")
    try:
        raise anthropic.APIError(
            message="Example: APIError base class demo.",
            request=_dummy_request(),
            body=None,
        )
    except anthropic.APIError as e:
        print(f"  Caught: {type(e).__name__}")
        print(f"  Message: {e.message}")
        # Best practice: use APIError as last-resort catch; log full details
        print("  Best practice: use APIError as outermost catch; record structured logs")

    # --- Exception hierarchy summary ---
    print("\n  [Exception Hierarchy]")
    print("  anthropic.APIError                   <- base for all Anthropic exceptions")
    print("  +- anthropic.APIStatusError          <- errors with HTTP status codes")
    print("  |   +- anthropic.AuthenticationError   (401)")
    print("  |   +- anthropic.PermissionDeniedError (403)")
    print("  |   +- anthropic.NotFoundError         (404)")
    print("  |   +- anthropic.RateLimitError         (429)")
    print("  |   +- anthropic.BadRequestError        (400)")
    print("  |   +- anthropic.InternalServerError    (500)")
    print("  +- anthropic.APIConnectionError      <- network/timeout errors")


# --------------------------------------------------
# 2. Exponential backoff retry
# --------------------------------------------------

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator: auto-retry on RateLimitError with exponential backoff.

    Retry delays: 1s -> 2s -> 4s  (base_delay * 2^attempt)
    Re-raises the original exception after max_retries is exhausted.
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
                        wait = base_delay * (2 ** attempt)  # 1 / 2 / 4 seconds
                        print(f"  [RateLimit] attempt {attempt + 1} rate-limited; "
                              f"retrying in {wait:.1f}s...")
                        time.sleep(wait)
                    else:
                        print(f"  [RateLimit] exhausted {max_retries} retries; giving up.")
            raise last_exc
        return wrapper
    return decorator


def demo_retry_backoff():
    """Exponential backoff retry pattern."""
    print("\n" + "=" * 60)
    print("[2] Exponential Backoff Retry Demo")
    print("=" * 60)

    call_count = 0

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def flaky_request():
        """Simulate a request that fails twice, then succeeds."""
        nonlocal call_count
        call_count += 1
        print(f"  -> attempt {call_count}")
        if call_count <= 2:
            raise anthropic.RateLimitError(
                message=f"Simulated rate limit (attempt {call_count})",
                response=_dummy_response(429),
                body=None,
            )
        return "Request succeeded!"

    print("\nScenario A: first 2 attempts rate-limited, 3rd succeeds (delay 1s / 2s)")
    try:
        result = flaky_request()
        print(f"  Final result: {result}")
        # Best practice: log successful retries; monitor retry frequency
        print("  Best practice: monitor retry rate; frequent retries -> reduce concurrency")
    except anthropic.RateLimitError as e:
        print(f"  Exceeded retry limit, giving up: {e.message}")

    print("\nScenario B: always fails, exceeds retry limit (0.1s delay for quick demo)")
    call_count = 0

    @retry_with_backoff(max_retries=2, base_delay=0.1)
    def always_fail():
        nonlocal call_count
        call_count += 1
        print(f"  -> attempt {call_count} (always fails)")
        raise anthropic.RateLimitError(
            message="Persistent rate limit",
            response=_dummy_response(429),
            body=None,
        )

    try:
        always_fail()
    except anthropic.RateLimitError as e:
        print(f"  Caught after exhausted retries: {type(e).__name__} - {e.message}")
        # Best practice: inform user "too many requests, please try again later"
        print("  Best practice: tell user 'Too many requests, please try again later'")


# --------------------------------------------------
# 3. Timeout configuration
# --------------------------------------------------

def demo_timeout():
    """Timeout configuration with httpx.Timeout."""
    print("\n" + "=" * 60)
    print("[3] Timeout Configuration Demo")
    print("=" * 60)

    # httpx.Timeout fields:
    #   connect  - max wait to establish a TCP connection
    #   read     - max wait for response data (increase for streaming)
    #   write    - max wait to send request body
    #   pool     - max wait to acquire a connection from the pool
    timeout_config = httpx.Timeout(
        connect=5.0,    # 5s to establish connection
        read=30.0,      # 30s to receive response (increase for streaming)
        write=10.0,     # 10s to send request body
        pool=5.0,       # 5s to acquire connection from pool
    )

    print(f"\n  Recommended timeout config:")
    print(f"    connect = {timeout_config.connect}s  (TCP connection)")
    print(f"    read    = {timeout_config.read}s (response data; increase for streaming)")
    print(f"    write   = {timeout_config.write}s (send request body)")
    print(f"    pool    = {timeout_config.pool}s  (acquire from connection pool)")

    # Show how to pass timeout to the Anthropic client
    client_with_timeout = anthropic.Anthropic(
        api_key=VALID_API_KEY or "dummy",
        timeout=timeout_config,
    )
    print(f"\n  Client configured with timeout type: {type(client_with_timeout.timeout).__name__}")

    # Demonstrate timeout triggering APIConnectionError with unreachable host
    print("\n  Demo: very short timeout (0.001s) -> APIConnectionError or APITimeoutError:")
    try:
        short_timeout_client = anthropic.Anthropic(
            api_key=VALID_API_KEY or "dummy",
            base_url="http://127.0.0.1:19999",  # non-existent local port
            timeout=httpx.Timeout(connect=0.001, read=0.001,
                                  write=0.001, pool=0.001),
        )
        short_timeout_client.messages.create(
            model=MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )
    except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
        print(f"  Caught: {type(e).__name__}")
        print(f"  Message (truncated): {str(e)[:80]}")
    except Exception as e:
        print(f"  Caught underlying exception: {type(e).__name__}: {str(e)[:80]}")

    print("\n  Best practices:")
    print("  - For streaming responses, set read timeout >= 60s")
    print("  - For batch/long-text tasks, read timeout can go to 120s+")
    print("  - connect timeout: 5-10s is usually sufficient")
    print("  - Catch both APIConnectionError and APITimeoutError for network issues")


# --------------------------------------------------
# 4. Graceful degradation
# --------------------------------------------------

def demo_graceful_degradation():
    """Graceful degradation: return friendly message on error, never crash."""
    print("\n" + "=" * 60)
    print("[4] Graceful Degradation Demo")
    print("=" * 60)

    FALLBACK = "Sorry, the service is temporarily unavailable. Please try again later."

    def safe_ask(client: anthropic.Anthropic, question: str) -> str:
        """
        Production-ready wrapper:
        - Catches exceptions from most specific to most general
        - Returns fallback instead of crashing
        - In production: use logging module for structured logs
        """
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=64,
                messages=[{"role": "user", "content": question}],
            )
            return response.content[0].text

        except anthropic.AuthenticationError as e:
            # Best practice: alert immediately; this should rarely happen in production
            print(f"  [ERROR] AuthenticationError: {e.message}")
            print("          -> Check ANTHROPIC_API_KEY configuration")
            return FALLBACK

        except anthropic.RateLimitError as e:
            # Best practice: reduce concurrency or wait before retrying
            print(f"  [WARN]  RateLimitError: {e.message}")
            print("          -> Reduce request frequency or retry with backoff")
            return FALLBACK

        except anthropic.BadRequestError as e:
            # Best practice: this is a client-side bug; fix the code
            print(f"  [ERROR] BadRequestError (400): {e.message}")
            print("          -> Fix request parameters")
            return FALLBACK

        except anthropic.APIConnectionError as e:
            # Best practice: check network/proxy; can retry after short wait
            print(f"  [ERROR] APIConnectionError: {e}")
            print("          -> Check network or proxy settings")
            return FALLBACK

        except anthropic.APIStatusError as e:
            # Catch-all for other HTTP errors
            print(f"  [ERROR] APIStatusError ({e.status_code}): {e.message}")
            print("          -> 4xx: fix params; 5xx: wait then retry")
            return FALLBACK

        except anthropic.APIError as e:
            # Last-resort catch for unexpected Anthropic exceptions
            print(f"  [ERROR] APIError (unexpected): {e.message}")
            return FALLBACK

    # --- Scenario A: AuthenticationError degradation ---
    print("\n  Scenario A: AuthenticationError -> graceful degradation")
    try:
        raise anthropic.AuthenticationError(
            message="401 Invalid API Key",
            response=_dummy_response(401),
            body=None,
        )
    except anthropic.AuthenticationError as e:
        print(f"  [ERROR] AuthenticationError: {e.message}")
        print(f"  Returned to user: {FALLBACK!r}")
        print("  Program did NOT crash!")

    # --- Scenario B: RateLimitError degradation ---
    print("\n  Scenario B: RateLimitError -> graceful degradation")
    try:
        raise anthropic.RateLimitError(
            message="429 Rate limit exceeded",
            response=_dummy_response(429),
            body=None,
        )
    except anthropic.RateLimitError as e:
        print(f"  [WARN]  RateLimitError: {e.message}")
        print(f"  Returned to user: {FALLBACK!r}")
        print("  Program did NOT crash!")

    # --- Scenario C: real request with valid API key ---
    print("\n  Scenario C: real request with valid API key")
    if VALID_API_KEY:
        real_client = anthropic.Anthropic(api_key=VALID_API_KEY)
        answer = safe_ask(real_client, "Introduce Python in one sentence.")
        print(f"  Claude replied: {answer!r}")
    else:
        print("  [SKIP] ANTHROPIC_API_KEY not set")

    # --- Best practice summary ---
    print("\n  Best Practice Summary:")
    print("  - Catch specific subclasses first; use APIError as last-resort catch")
    print("  - 4xx errors are usually code/config bugs; notify the developer")
    print("  - 5xx and ConnectionError are retriable; use exponential backoff")
    print("  - Show friendly messages to users; never expose internal error details")
    print("  - Use Python logging module with structured output in production")
    print("  - Add monitoring/alerting (e.g. alert when AuthenticationError spikes)")


# --------------------------------------------------
# main
# --------------------------------------------------

def main():
    print("=" * 60)
    print(" Claude API Error Handling Demo")
    print("=" * 60)

    demo_error_types()
    demo_retry_backoff()
    demo_timeout()
    demo_graceful_degradation()

    print("\n" + "=" * 60)
    print(" All demos completed. Program exited normally.")
    print("=" * 60)


if __name__ == "__main__":
    main()
