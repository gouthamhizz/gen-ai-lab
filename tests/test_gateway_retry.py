from collections import Counter

from genai_platform.llm.gateway import (LLMResult, RateLimitError, RetryConfig, TransientError, Usage,
                                        backoff_delay, call_with_retry, parse_retry_after)


def test_backoff_delay_is_bounded_and_nonnegative():
    for attempt in range(6):
        d = backoff_delay(attempt, base=1.0, cap=30.0)
        assert 0 <= d <= 30.0


def test_parse_retry_after_header_seconds():
    assert parse_retry_after({"Retry-After": "3"}) == 3.0


def test_parse_retry_after_openai_style_duration():
    assert parse_retry_after({"x-ratelimit-reset-requests": "1m30s"}) == 90.0


def test_parse_retry_after_gemini_message_hint():
    assert parse_retry_after(None, 'error: "retryDelay": "17s" please retry') == 17.0


def test_call_with_retry_gives_up_after_max_retries():
    sleeps = []
    cfg = RetryConfig(max_retries=2, sleep=sleeps.append)
    stats = Counter()

    def always_fails():
        raise TransientError("boom")

    try:
        call_with_retry(always_fails, cfg, stats)
        assert False, "should have raised"
    except TransientError:
        pass
    assert stats["retries"] == 2
    assert len(sleeps) == 2


def test_call_with_retry_succeeds_after_transient_failures():
    attempts = {"n": 0}
    cfg = RetryConfig(max_retries=3, sleep=lambda s: None)
    stats = Counter()

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimitError("slow down", retry_after=0.01)
        return LLMResult("ok", Usage(1, 1), "test", "m", 0.0)

    res = call_with_retry(flaky, cfg, stats)
    assert res.text == "ok"
    assert stats["rate_limited"] == 2
