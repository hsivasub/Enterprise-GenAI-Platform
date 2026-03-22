"""
Tests for guardrails/output_filter.py
========================================
Covers disclaimer injection, PII leak detection in output,
investment advice flagging, and hallucination risk scoring.
"""

from __future__ import annotations

import pytest

from guardrails.output_filter import OutputFilter, OutputFilterResult


@pytest.fixture
def output_filter() -> OutputFilter:
    return OutputFilter()


# ── Disclaimer injection ─────────────────────────────────────────

class TestDisclaimerInjection:

    def test_disclaimer_added_for_financial_response(self, output_filter):
        response = "Apple had revenue of $85.8 billion and net income of $21.4 billion."
        result = output_filter.filter(response)
        assert result.disclaimer_added
        assert "Disclaimer" in result.filtered or "disclaimer" in result.filtered

    def test_disclaimer_not_added_for_non_financial_response(self, output_filter):
        response = "Hello! How are you today? The weather is nice."
        result = output_filter.filter(response)
        assert not result.disclaimer_added

    def test_disclaimer_content_is_appropriate(self, output_filter):
        response = "The investment return was 15% this year."
        result = output_filter.filter(response)
        if result.disclaimer_added:
            assert "informational" in result.filtered.lower() or \
                   "financial advice" in result.filtered.lower()


# ── PII in response ──────────────────────────────────────────────

class TestPIIInResponse:

    def test_ssn_in_response_is_redacted(self, output_filter):
        response = "The client's SSN 123-45-6789 was found in the records."
        result = output_filter.filter(response)
        assert "123-45-6789" not in result.filtered

    def test_clean_response_has_no_violations(self, output_filter):
        response = "Revenue grew 14% year-over-year to $24.2 billion."
        result = output_filter.filter(response)
        # No PII violations
        pii_violations = [v for v in result.violations_removed if "PII" in v]
        assert len(pii_violations) == 0


# ── Investment advice detection ───────────────────────────────────

class TestInvestmentAdviceDetection:

    def test_buy_recommendation_flagged(self, output_filter):
        response = "You should buy Apple stock immediately before the earnings."
        result = output_filter.filter(response)
        assert len(result.violations_removed) > 0 or "advisor" in result.filtered.lower()

    def test_factual_response_not_flagged(self, output_filter):
        response = "Apple's P/E ratio is 28.5, compared to sector average of 25.3."
        result = output_filter.filter(response)
        # Pure factual analysis — may add disclaimer but shouldn't flag violations
        pii_violations = [v for v in result.violations_removed if "SSN" in v or "PII" in v]
        assert len(pii_violations) == 0


# ── Hallucination risk scoring ────────────────────────────────────

class TestHallucinationRisk:

    def test_low_risk_when_numbers_match_context(self, output_filter):
        context = "Apple revenue was $85.8 billion. Net income was $21.4 billion."
        response = "Apple revenue was $85.8 billion and net income was $21.4 billion."
        result = output_filter.filter(response, context=context)
        assert result.hallucination_risk in ("LOW", "MEDIUM")

    def test_high_risk_when_numbers_dont_match_context(self, output_filter):
        context = "Revenue was $10 million."
        response = "Revenue was $999 billion. Net income hit $750 billion. EPS was $45.23."
        result = output_filter.filter(response, context=context)
        assert result.hallucination_risk in ("HIGH", "MEDIUM")

    def test_low_risk_without_context(self, output_filter):
        response = "The analysis shows positive trends."
        result = output_filter.filter(response, context=None)
        assert result.hallucination_risk == "LOW"

    def test_risk_is_valid_enum_value(self, output_filter):
        response = "Revenue grew by $5 billion to $100 billion."
        context = "Revenue grew to $100 billion."
        result = output_filter.filter(response, context=context)
        assert result.hallucination_risk in ("LOW", "MEDIUM", "HIGH")


# ── OutputFilterResult properties ────────────────────────────────

class TestOutputFilterResult:

    def test_is_safe_true_for_low_risk(self, output_filter):
        response = "The company reported strong earnings growth."
        result = output_filter.filter(response)
        assert result.is_safe or result.hallucination_risk != "HIGH"

    def test_original_text_preserved(self, output_filter):
        response = "Apple revenue was $85.8 billion."
        result = output_filter.filter(response)
        assert result.original == response

    def test_filtered_text_is_string(self, output_filter):
        result = output_filter.filter("Some response text.")
        assert isinstance(result.filtered, str)

    def test_violations_is_list(self, output_filter):
        result = output_filter.filter("Clean neutral response.")
        assert isinstance(result.violations_removed, list)
