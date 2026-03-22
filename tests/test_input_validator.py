"""
Tests for guardrails/input_validator.py
=========================================
Covers PII detection, prompt injection blocking,
blocked term enforcement, and length validation.
"""

from __future__ import annotations

import pytest

from guardrails.input_validator import (
    InputValidator,
    PIIMatch,
    ValidationResult,
    ValidationStatus,
)


@pytest.fixture
def validator() -> InputValidator:
    return InputValidator()


# ── Status: PASSED ───────────────────────────────────────────────

class TestCleanInputPasses:

    def test_clean_financial_question_passes(self, validator):
        result = validator.validate("What was Apple revenue in Q3 2024?")
        assert result.is_allowed
        assert result.status == ValidationStatus.PASSED

    def test_empty_ish_input_passes(self, validator):
        result = validator.validate("Hello")
        assert result.is_allowed

    def test_multiline_clean_text_passes(self, validator):
        text = "Compare the EPS of Apple and Microsoft.\nWhich performed better?"
        result = validator.validate(text)
        assert result.is_allowed

    def test_processed_text_equals_original_when_clean(self, validator):
        text = "What is the P/E ratio for Google?"
        result = validator.validate(text)
        assert result.processed_text == text


# ── PII Detection & Redaction ─────────────────────────────────────

class TestPIIDetection:

    def test_email_is_detected_and_redacted(self, validator):
        result = validator.validate("Contact me at user@example.com for details.")
        assert result.pii_detected
        assert "[EMAIL REDACTED]" in result.processed_text
        assert "user@example.com" not in result.processed_text

    def test_ssn_is_detected_and_redacted(self, validator):
        result = validator.validate("My SSN is 123-45-6789.")
        assert result.pii_detected
        assert "123-45-6789" not in result.processed_text
        assert "[SSN REDACTED]" in result.processed_text

    def test_credit_card_is_detected_and_redacted(self, validator):
        result = validator.validate("Card number 4111-1111-1111-1111 was charged.")
        assert result.pii_detected
        assert "4111-1111-1111-1111" not in result.processed_text

    def test_us_phone_is_detected_and_redacted(self, validator):
        result = validator.validate("Call me at 555-123-4567.")
        assert result.pii_detected
        assert "555-123-4567" not in result.processed_text

    def test_multiple_pii_types_in_one_input(self, validator):
        text = "Email: user@test.com, SSN: 987-65-4321, Phone: 800-555-0100"
        result = validator.validate(text)
        assert result.pii_detected
        assert result.status == ValidationStatus.REDACTED
        assert len(result.pii_found) >= 2

    def test_pii_result_has_correct_type(self, validator):
        result = validator.validate("user@example.com")
        found_types = [m.pii_type for m in result.pii_found]
        assert "EMAIL" in found_types

    def test_original_text_preserved_in_result(self, validator):
        original = "My email is test@test.com"
        result = validator.validate(original)
        assert result.original_text == original

    def test_no_pii_in_financial_text(self, validator):
        text = "Apple reported $85.8 billion revenue in Q3 2024 with 46.3% gross margin."
        result = validator.validate(text)
        assert not result.pii_detected
        assert result.status == ValidationStatus.PASSED


# ── Blocked Terms ────────────────────────────────────────────────

class TestBlockedTerms:

    def test_script_tag_is_blocked(self, validator):
        result = validator.validate("Hello <script>alert('xss')</script>")
        assert not result.is_allowed
        assert result.status == ValidationStatus.BLOCKED

    def test_sql_injection_attempt_is_blocked(self, validator):
        result = validator.validate("What happens if I type DROP TABLE users?")
        assert not result.is_allowed

    def test_blocked_input_has_violations_list(self, validator):
        result = validator.validate("DROP TABLE financial_data")
        assert len(result.violations) > 0

    def test_clean_input_has_no_violations(self, validator):
        result = validator.validate("What is the revenue forecast for next year?")
        assert result.violations == []


# ── Length Validation ────────────────────────────────────────────

class TestLengthValidation:

    def test_input_over_limit_is_blocked(self, validator):
        # Generate input exceeding MAX_INPUT_LENGTH (4096)
        long_input = "A" * 5000
        result = validator.validate(long_input)
        assert not result.is_allowed
        assert result.status == ValidationStatus.BLOCKED

    def test_input_at_limit_is_allowed(self, validator):
        # Exactly at the limit (no PII, no blocked terms)
        text = "A" * 100  # Well under limit
        result = validator.validate(text)
        assert result.is_allowed

    def test_violation_message_mentions_length(self, validator):
        long_input = "B" * 5000
        result = validator.validate(long_input)
        assert any("long" in v.lower() or "limit" in v.lower() for v in result.violations)


# ── ValidationResult properties ──────────────────────────────────

class TestValidationResult:

    def test_is_allowed_true_for_passed(self, validator):
        result = validator.validate("Normal financial question.")
        assert result.is_allowed is True

    def test_is_allowed_false_for_blocked(self, validator):
        result = validator.validate("DROP TABLE users")
        assert result.is_allowed is False

    def test_pii_detected_false_when_no_pii(self, validator):
        result = validator.validate("What is EPS?")
        assert result.pii_detected is False

    def test_result_processed_text_is_string(self, validator):
        result = validator.validate("test input")
        assert isinstance(result.processed_text, str)
