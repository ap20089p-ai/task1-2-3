"""Unit tests for Safe Mathematical Calculator.

Ensures reliable arithmetic, operator precedence, division-by-zero handling,
syntax validation, and protection against arbitrary code execution.
Does not require any Gemini API calls.
"""

from __future__ import annotations

import pytest
from tools import calculate, SafeCalculator


class TestCalculatorArithmetic:
    """Tests for basic and complex arithmetic expressions."""

    def test_basic_addition(self) -> None:
        assert calculate("10 + 25") == "35"

    def test_basic_subtraction(self) -> None:
        assert calculate("100 - 37") == "63"

    def test_basic_multiplication(self) -> None:
        assert calculate("25 * 4") == "100"

    def test_basic_division(self) -> None:
        assert calculate("100 / 4") == "25"

    def test_float_division(self) -> None:
        assert calculate("7 / 2") == "3.5"

    def test_floor_division(self) -> None:
        assert calculate("17 // 5") == "3"

    def test_modulo(self) -> None:
        assert calculate("29 % 5") == "4"

    def test_power_operator(self) -> None:
        assert calculate("2 ** 8") == "256"

    def test_power_caret_alias(self) -> None:
        assert calculate("2 ^ 8") == "256"

    def test_canonical_project_query(self) -> None:
        # Canonical query: 1250 multiplied by 18 and then add 500
        assert calculate("1250 * 18 + 500") == "23000"

    def test_operator_precedence_with_parentheses(self) -> None:
        assert calculate("(10 + 5) * 2") == "30"
        assert calculate("10 + (5 * 2)") == "20"
        assert calculate("((5 + 3) * 2) ** 2") == "256"

    def test_unary_operators(self) -> None:
        assert calculate("-50 + 20") == "-30"
        assert calculate("-(5 * 4)") == "-20"
        assert calculate("+15 - -5") == "20"


class TestCalculatorErrorHandling:
    """Tests for division by zero, invalid expressions, and security boundaries."""

    def test_division_by_zero(self) -> None:
        result = calculate("10 / 0")
        assert "Division by zero" in result

    def test_floor_division_by_zero(self) -> None:
        result = calculate("10 // 0")
        assert "Division by zero" in result

    def test_modulo_by_zero(self) -> None:
        result = calculate("10 % 0")
        assert "Division by zero" in result

    def test_empty_expression(self) -> None:
        assert "cannot be empty" in calculate("")
        assert "cannot be empty" in calculate("   ")

    def test_syntax_errors(self) -> None:
        assert "Invalid mathematical syntax" in calculate("2 + * 4")
        assert "Invalid mathematical syntax" in calculate("(10 + 5")
        assert "Invalid mathematical syntax" in calculate("++")

    def test_arbitrary_variable_rejected(self) -> None:
        result = calculate("x + 5")
        assert "Unsupported expression" in result

    def test_code_injection_attempt_import_rejected(self) -> None:
        result = calculate("__import__('os').system('dir')")
        assert "Unsupported expression" in result or "Invalid" in result

    def test_function_call_rejected(self) -> None:
        result = calculate("print('hello')")
        assert "Unsupported expression" in result

    def test_eval_call_rejected(self) -> None:
        result = calculate("eval('2+2')")
        assert "Unsupported expression" in result

    def test_string_literal_rejected(self) -> None:
        result = calculate("'test' + 'string'")
        assert "Unsupported expression" in result or "Unsupported constant" in result

    def test_astronomical_exponent_dos_prevention(self) -> None:
        result = calculate("2 ** 9999999")
        assert "exceeds safe limit" in result
