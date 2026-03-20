"""
Output Filter — Guardrails Layer
==================================
Post-processes LLM responses before returning to user:
1. Content filtering (financial regulations compliance)
2. Hallucination detection placeholder (confidence score)
3. Investment advice disclaimer injection
4. Response quality checks
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from config.settings import settings
from observability.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# Patterns: things that should NEVER appear in responses
# ─────────────────────────────────────────────────────────────

# Financial advice that could constitute unlicensed advice
ADVICE_PATTERNS = [
    re.compile(r"\b(buy|sell|purchase|invest in|put your money in)\s+\w+\s+stock", re.IGNORECASE),
    re.compile(r"\b(you should|i recommend|i advise|you must)\s+(buy|sell|invest|trade)", re.IGNORECASE),
    re.compile(r"\b(guaranteed|certain|definitely will|100%)\s+(?:return|profit|gain)", re.IGNORECASE),
]

# PII that might have leaked into the response
RESPONSE_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                     # SSN
    re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),               # Credit card
]

# Standard financial disclaimer
DISCLAIMER = (
    "\n\n---\n*Disclaimer: This analysis is for informational purposes only and does not "
    "constitute financial advice. Past performance is not indicative of future results. "
    "Consult a licensed financial advisor before making investment decisions.*"
)


@dataclass
class OutputFilterResult:
    original: str
    filtered: str
    violations_removed: List[str]
    disclaimer_added: bool
    hallucination_risk: str    # "LOW" | "MEDIUM" | "HIGH"

    @property
    def is_safe(self) -> bool:
        return self.hallucination_risk != "HIGH"


class OutputFilter:
    """
    Filters and enriches LLM responses for compliance and safety.
    Fast path: run only if ENABLE_CONTENT_FILTER is True.
    """

    def filter(
        self,
        response: str,
        context: Optional[str] = None,
    ) -> OutputFilterResult:
        if not settings.ENABLE_CONTENT_FILTER:
            return OutputFilterResult(
                original=response,
                filtered=response,
                violations_removed=[],
                disclaimer_added=False,
                hallucination_risk="LOW",
            )

        filtered = response
        violations: List[str] = []

        # 1. Check for unlicensed investment advice language
        for pattern in ADVICE_PATTERNS:
            if pattern.search(filtered):
                violations.append(f"Potential investment advice: {pattern.pattern}")
                # Flag but don't silently delete — append warning instead
                filtered = filtered + "\n\n⚠️ *Note: Consult a licensed advisor for investment decisions.*"
                break

        # 2. Check for PII leakage in response
        for pattern in RESPONSE_PII_PATTERNS:
            if pattern.search(filtered):
                violations.append("PII detected in response — redacting")
                filtered = pattern.sub("[REDACTED]", filtered)

        # 3. Add disclaimer for financial responses
        disclaimer_added = False
        if self._should_add_disclaimer(response):
            filtered += DISCLAIMER
            disclaimer_added = True

        # 4. Hallucination risk scoring (lightweight heuristic)
        hallucination_risk = self._score_hallucination_risk(response, context)

        if violations:
            logger.warning(
                "Output filter triggered",
                extra={"violations": violations},
            )

        return OutputFilterResult(
            original=response,
            filtered=filtered,
            violations_removed=violations,
            disclaimer_added=disclaimer_added,
            hallucination_risk=hallucination_risk,
        )

    @staticmethod
    def _should_add_disclaimer(response: str) -> bool:
        """Add disclaimer if response discusses financial topics."""
        financial_terms = [
            "investment", "stock", "portfolio", "return", "revenue",
            "earnings", "profit", "loss", "market cap", "P/E ratio",
        ]
        lower = response.lower()
        return any(term in lower for term in financial_terms)

    @staticmethod
    def _score_hallucination_risk(response: str, context: Optional[str]) -> str:
        """
        Lightweight hallucination risk scoring.
        In production: replace with a dedicated cross-encoder or
        NLI-based fact verification model.
        
        Heuristics:
        - LOW: response cites sources from context
        - MEDIUM: response has numbers not found in context
        - HIGH: response makes definitive claims with no context match
        """
        if not context:
            # Can't verify without context
            return "LOW"

        # Check if key numbers in response appear in context
        response_numbers = re.findall(r"\b\d[\d,\.]+\b", response)
        context_numbers = set(re.findall(r"\b\d[\d,\.]+\b", context))

        unverified = [n for n in response_numbers if n not in context_numbers]

        # Risk based on ratio of unverified numbers
        if not response_numbers:
            return "LOW"
        unverified_ratio = len(unverified) / len(response_numbers)
        if unverified_ratio > 0.7:
            return "HIGH"
        elif unverified_ratio > 0.3:
            return "MEDIUM"
        return "LOW"
