import pytest

from genai_platform.agents.tools import calculator, safe_eval


def test_calculator_arithmetic():
    assert calculator("(17*23)+sqrt(144)") == "403"
    assert calculator("2**10") == "1024"


def test_calculator_division_by_zero_is_safe():
    assert "division by zero" in calculator("10/0")


def test_calculator_rejects_code_execution():
    with pytest.raises(Exception):
        safe_eval("__import__('os').system('echo pwned')")


def test_calculator_rejects_huge_exponent():
    assert "too large" in calculator("2**99999")
