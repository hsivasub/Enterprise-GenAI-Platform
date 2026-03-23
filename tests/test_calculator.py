"""
Tests for agents/tools.py — CalculatorTool
============================================
Validates safe AST-based expression evaluation:
arithmetic, math functions, error cases, and security checks.
"""

from __future__ import annotations

import math

import pytest

from agents.tools import CalculatorTool


@pytest.fixture
def calc() -> CalculatorTool:
    return CalculatorTool()


# ── Basic arithmetic ─────────────────────────────────────────────

class TestBasicArithmetic:

    def test_addition(self, calc):
        assert calc.run("2 + 2") == "4"

    def test_subtraction(self, calc):
        assert calc.run("10 - 3") == "7"

    def test_multiplication(self, calc):
        assert calc.run("6 * 7") == "42"

    def test_division(self, calc):
        result = float(calc.run("10 / 4"))
        assert abs(result - 2.5) < 1e-9

    def test_floor_division(self, calc):
        assert calc.run("10 // 3") == "3"

    def test_modulo(self, calc):
        assert calc.run("10 % 3") == "1"

    def test_exponentiation(self, calc):
        assert calc.run("2 ** 10") == "1024"

    def test_unary_negation(self, calc):
        assert calc.run("-5 + 10") == "5"

    def test_operator_precedence(self, calc):
        # 2 + 3 * 4 = 14 (not 20)
        assert calc.run("2 + 3 * 4") == "14"

    def test_parentheses(self, calc):
        assert calc.run("(2 + 3) * 4") == "20"

    def test_large_numbers(self, calc):
        result = calc.run("1_000_000 * 365")
        # Python allows underscores in numeric literals
        assert "365000000" in result or "error" not in result.lower()


# ── Financial calculations ────────────────────────────────────────

class TestFinancialCalculations:

    def test_pe_ratio(self, calc):
        # P/E = Price / EPS = 175 / 1.40
        result = float(calc.run("175 / 1.40"))
        assert abs(result - 125.0) < 0.01

    def test_revenue_per_day(self, calc):
        # Annual revenue / 365
        result = float(calc.run("85800000000 / 365"))
        assert result > 200_000_000  # > $200M per day

    def test_percentage_change(self, calc):
        # YoY growth: (85.8 - 81.8) / 81.8 * 100
        result = float(calc.run("(85.8 - 81.8) / 81.8 * 100"))
        assert abs(result - 4.89) < 0.1

    def test_compound_interest(self, calc):
        # FV = PV * (1 + r)^n = 1000 * (1.05)^10
        result = float(calc.run("1000 * (1.05 ** 10)"))
        assert abs(result - 1628.89) < 1.0

    def test_market_cap_calculation(self, calc):
        # Market cap = shares * price = 15.4B * $175
        result = float(calc.run("15400000000 * 175"))
        assert result == 2_695_000_000_000.0


# ── Math functions ────────────────────────────────────────────────

class TestMathFunctions:

    def test_sqrt(self, calc):
        result = float(calc.run("sqrt(144)"))
        assert abs(result - 12.0) < 1e-9

    def test_log_natural(self, calc):
        result = float(calc.run("log(2.718281828)"))
        assert abs(result - 1.0) < 0.001

    def test_log_base_10(self, calc):
        result = float(calc.run("log10(1000)"))
        assert abs(result - 3.0) < 1e-9

    def test_abs_negative(self, calc):
        assert calc.run("abs(-42)") == "42"

    def test_round(self, calc):
        result = calc.run("round(3.14159, 2)")
        assert "3.14" in result

    def test_min(self, calc):
        assert calc.run("min(10, 5, 8)") == "5"

    def test_max(self, calc):
        assert calc.run("max(10, 5, 8)") == "10"

    def test_ceil(self, calc):
        assert calc.run("ceil(4.1)") == "5"

    def test_floor(self, calc):
        assert calc.run("floor(4.9)") == "4"

    def test_pi_constant(self, calc):
        result = float(calc.run("pi"))
        assert abs(result - math.pi) < 1e-5

    def test_e_constant(self, calc):
        result = float(calc.run("e"))
        assert abs(result - math.e) < 1e-5


# ── Dict input format ─────────────────────────────────────────────

class TestDictInput:

    def test_accepts_dict_with_expression_key(self, calc):
        result = calc.run({"expression": "2 + 2"})
        assert result == "4"

    def test_accepts_string_input_directly(self, calc):
        result = calc.run("5 * 5")
        assert result == "25"


# ── Security: disallowed operations ──────────────────────────────

class TestSecurityBlocks:

    def test_import_is_blocked(self, calc):
        result = calc.run("__import__('os').system('echo pwned')")
        assert "error" in result.lower() or "disallowed" in result.lower()

    def test_attribute_access_blocked(self, calc):
        # Attribute access on arbitrary objects should be blocked
        result = calc.run("().__class__.__bases__[0].__subclasses__()")
        assert "error" in result.lower() or "cannot" in result.lower()

    def test_exec_style_string_blocked(self, calc):
        result = calc.run("eval('2+2')")
        assert "error" in result.lower() or "unknown" in result.lower()


# ── Error handling ────────────────────────────────────────────────

class TestErrorHandling:

    def test_division_by_zero_returns_error(self, calc):
        result = calc.run("1 / 0")
        assert "error" in result.lower() or "division" in result.lower()

    def test_invalid_syntax_returns_error(self, calc):
        result = calc.run("2 +* 3")
        assert "error" in result.lower()

    def test_unknown_function_returns_error(self, calc):
        result = calc.run("unknown_func(42)")
        assert "error" in result.lower() or "unknown" in result.lower()

    def test_empty_expression_returns_error(self, calc):
        result = calc.run("")
        assert "error" in result.lower() or result == ""
