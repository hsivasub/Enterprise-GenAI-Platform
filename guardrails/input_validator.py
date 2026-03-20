"""
Input Validator + PII Detection — Guardrails Layer
====================================================
Runs before any LLM call to:
1. Validate input length and format
2. Detect and redact PII (names, SSNs, credit cards, emails, phones)
3. Block prompt injection attempts
4. Screen for policy violations (blocked terms)

Design philosophy:
- Fast: regex-based, < 5ms per check
- Non-blocking: detects + redacts (doesn't silently drop)
- Transparent: returns a ValidationResult explaining what was found
- Configurable: all patterns and thresholds come from settings
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from config.settings import settings
from observability.logger import get_logger

logger = get_logger(__name__)


class ValidationStatus(str, Enum):
    PASSED = "passed"
    REDACTED = "redacted"       # PII found and removed — proceed with redacted input
    BLOCKED = "blocked"          # Policy violation — reject request
    WARNING = "warning"          # Minor concern — log but proceed


@dataclass
class PIIMatch:
    pii_type: str
    original: str
    redacted: str
    start: int
    end: int


@dataclass
class ValidationResult:
    status: ValidationStatus
    original_text: str
    processed_text: str
    pii_found: List[PIIMatch] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_allowed(self) -> bool:
        return self.status != ValidationStatus.BLOCKED

    @property
    def pii_detected(self) -> bool:
        return len(self.pii_found) > 0


# ─────────────────────────────────────────────────────────────
# PII Patterns
# ─────────────────────────────────────────────────────────────

PII_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    # (pii_type, pattern, replacement)
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[SSN REDACTED]",
    ),
    (
        "CREDIT_CARD",
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "[CARD REDACTED]",
    ),
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[EMAIL REDACTED]",
    ),
    (
        "PHONE_US",
        re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "[PHONE REDACTED]",
    ),
    (
        "IP_ADDRESS",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "[IP REDACTED]",
    ),
    (
        "PASSPORT",
        re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),  # Basic pattern
        "[PASSPORT REDACTED]",
    ),
]

# Prompt injection patterns — common adversarial inputs
INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+(previous|all|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|previous)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(?:an?\s+)?(?!financial|professional|analyst)", re.IGNORECASE),
    re.compile(r"jailbreak|DAN|do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),  # Prompt injection via tags
    re.compile(r"\\n\\n(Human|Assistant|System):", re.IGNORECASE),
]


class InputValidator:
    """
    Validates and sanitizes user input before it reaches the LLM.
    
    Call validate() and check result.is_allowed before proceeding.
    Use result.processed_text as the actual input to the LLM.
    """

    def __init__(self) -> None:
        self._blocked_patterns = [
            re.compile(re.escape(term), re.IGNORECASE)
            for term in settings.BLOCKED_TERMS
        ]

    def validate(self, text: str) -> ValidationResult:
        """
        Run all checks on the input text.
        Returns ValidationResult with potentially redacted text.
        """
        processed = text
        pii_matches: List[PIIMatch] = []
        violations: List[str] = []
        warnings: List[str] = []

        # 1. Length check
        if len(text) > settings.MAX_INPUT_LENGTH:
            return ValidationResult(
                status=ValidationStatus.BLOCKED,
                original_text=text,
                processed_text=text,
                violations=[
                    f"Input too long: {len(text)} chars > {settings.MAX_INPUT_LENGTH} limit"
                ],
            )

        # 2. Blocked terms
        for pattern in self._blocked_patterns:
            if pattern.search(processed):
                violations.append(f"Blocked term detected: {pattern.pattern}")

        if violations:
            logger.warning(
                "Input blocked: policy violation",
                extra={"violations": violations, "text_preview": text[:100]},
            )
            return ValidationResult(
                status=ValidationStatus.BLOCKED,
                original_text=text,
                processed_text=text,
                violations=violations,
            )

        # 3. Prompt injection detection
        for pattern in INJECTION_PATTERNS:
            if pattern.search(processed):
                warnings.append("Potential prompt injection attempt detected")
                logger.warning(
                    "Prompt injection pattern matched",
                    extra={"pattern": pattern.pattern, "text_preview": text[:100]},
                )
                # Warn but don't block — log for human review
                break

        # 4. PII detection and redaction
        if settings.ENABLE_PII_DETECTION:
            processed, pii_matches = self._redact_pii(processed)

        status = ValidationStatus.PASSED
        if pii_matches:
            status = ValidationStatus.REDACTED
            logger.info(
                f"PII redacted from input: {[m.pii_type for m in pii_matches]}"
            )
        elif warnings:
            status = ValidationStatus.WARNING

        return ValidationResult(
            status=status,
            original_text=text,
            processed_text=processed,
            pii_found=pii_matches,
            violations=violations,
            warnings=warnings,
        )

    @staticmethod
    def _redact_pii(text: str) -> Tuple[str, List[PIIMatch]]:
        """Apply all PII patterns and replace with redaction tokens."""
        matches: List[PIIMatch] = []
        offset = 0   # Track index shift after replacements

        for pii_type, pattern, replacement in PII_PATTERNS:
            for m in pattern.finditer(text):
                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        original=m.group(),
                        redacted=replacement,
                        start=m.start(),
                        end=m.end(),
                    )
                )

        # Apply replacements
        for pii_type, pattern, replacement in PII_PATTERNS:
            text = pattern.sub(replacement, text)

        return text, matches
