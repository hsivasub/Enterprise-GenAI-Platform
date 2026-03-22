"""
Tests for evaluation/evaluator.py — RAGEvaluator
==================================================
Validates heuristic scoring logic:
context relevance, faithfulness, answer relevance,
composite score, batch evaluation, and summary stats.
"""

from __future__ import annotations

import pytest

from evaluation.evaluator import EvaluationResult, RAGEvaluator


@pytest.fixture
def evaluator() -> RAGEvaluator:
    return RAGEvaluator(llm_client=None)  # Always use heuristic mode


# ── Evaluate returns EvaluationResult ────────────────────────────

class TestReturnType:

    def test_returns_evaluation_result(self, evaluator):
        result = evaluator.evaluate(
            question="What is Apple revenue?",
            answer="Apple revenue was $85.8 billion in Q3 2024.",
            context="Apple reported revenue of $85.8 billion in Q3 FY2024.",
        )
        assert isinstance(result, EvaluationResult)

    def test_all_scores_between_zero_and_one(self, evaluator):
        result = evaluator.evaluate(
            question="What is the EPS?",
            answer="EPS was $1.40.",
            context="Earnings per share was $1.40 for the quarter.",
        )
        assert 0.0 <= result.context_relevance <= 1.0
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.answer_relevance <= 1.0
        assert 0.0 <= result.composite_score <= 1.0

    def test_method_field_is_heuristic(self, evaluator):
        result = evaluator.evaluate(
            question="Test?", answer="Answer.", context="Some context."
        )
        assert result.method == "heuristic"

    def test_latency_is_positive(self, evaluator):
        result = evaluator.evaluate(
            question="Test?", answer="Answer.", context="Context."
        )
        assert result.latency_ms >= 0


# ── Score quality ─────────────────────────────────────────────────

class TestScoringQuality:

    def test_high_faithfulness_when_answer_matches_context(self, evaluator):
        context = "Apple revenue was $85.8 billion. Services grew 14% year over year."
        answer = "Apple revenue was $85.8 billion and services grew 14% year over year."
        result = evaluator.evaluate(
            question="What was Apple revenue?",
            answer=answer,
            context=context,
        )
        assert result.faithfulness > 0.3

    def test_low_faithfulness_when_answer_unrelated_to_context(self, evaluator):
        context = "Apple revenue was $85 billion."
        answer = "Quantum computing will revolutionize cryptocurrency markets by 2030."
        result = evaluator.evaluate(
            question="What is Apple revenue?",
            answer=answer,
            context=context,
        )
        assert result.faithfulness < 0.7

    def test_high_context_relevance_when_context_answers_question(self, evaluator):
        question = "What is Apple revenue?"
        context = "Apple reported revenue of $85 billion for the quarter."
        result = evaluator.evaluate(
            question=question,
            answer="Revenue was $85 billion.",
            context=context,
        )
        assert result.context_relevance > 0.0

    def test_low_answer_relevance_for_short_answer(self, evaluator):
        result = evaluator.evaluate(
            question="What were the key financial highlights for Apple Q3 2024?",
            answer="Yes.",
            context="Apple revenue was $85 billion, EPS $1.40, services $24 billion.",
        )
        # Short answer should have low answer relevance
        assert result.answer_relevance < 0.8

    def test_composite_is_weighted_average(self, evaluator):
        """Composite = 0.3*CR + 0.4*F + 0.3*AR"""
        result = evaluator.evaluate(
            question="Apple revenue?",
            answer="Revenue was $85.8 billion.",
            context="Apple revenue was $85.8 billion.",
        )
        expected = round(
            0.3 * result.context_relevance
            + 0.4 * result.faithfulness
            + 0.3 * result.answer_relevance,
            3,
        )
        assert abs(result.composite_score - expected) < 0.01


# ── Empty / edge cases ────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_context_still_returns_result(self, evaluator):
        result = evaluator.evaluate(
            question="What is EPS?",
            answer="EPS was $1.40.",
            context="",
        )
        assert isinstance(result, EvaluationResult)

    def test_very_long_context_does_not_crash(self, evaluator):
        long_context = "Apple revenue was $85 billion. " * 500
        result = evaluator.evaluate(
            question="Apple revenue?",
            answer="$85 billion.",
            context=long_context,
        )
        assert isinstance(result, EvaluationResult)


# ── Batch evaluation ─────────────────────────────────────────────

class TestBatchEvaluation:

    def test_batch_returns_list_of_results(self, evaluator):
        samples = [
            {
                "question": "What is Apple revenue?",
                "answer": "Revenue was $85.8 billion.",
                "context": "Apple reported revenue of $85.8 billion.",
            },
            {
                "question": "What was EPS?",
                "answer": "EPS was $1.40.",
                "context": "Earnings per share was $1.40.",
            },
        ]
        results = evaluator.batch_evaluate(samples)
        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)

    def test_empty_batch_returns_empty_list(self, evaluator):
        results = evaluator.batch_evaluate([])
        assert results == []


# ── Summary stats ─────────────────────────────────────────────────

class TestSummaryStats:

    def test_summary_stats_returns_dict_with_expected_keys(self, evaluator):
        samples = [
            {
                "question": "Q?",
                "answer": "Apple revenue was high.",
                "context": "Apple revenue was $85 billion.",
            }
        ]
        results = evaluator.batch_evaluate(samples)
        stats = evaluator.summary_stats(results)

        for key in ["n", "avg_composite", "avg_faithfulness",
                    "avg_context_relevance", "avg_answer_relevance",
                    "min_composite", "max_composite"]:
            assert key in stats, f"Missing key: {key}"

    def test_summary_stats_n_equals_sample_count(self, evaluator):
        samples = [
            {"question": "Q?", "answer": f"Answer {i}.", "context": f"Context {i}."}
            for i in range(4)
        ]
        results = evaluator.batch_evaluate(samples)
        stats = evaluator.summary_stats(results)
        assert stats["n"] == 4

    def test_summary_stats_empty_input_returns_empty(self, evaluator):
        stats = evaluator.summary_stats([])
        assert stats == {}

    def test_min_lte_avg_lte_max(self, evaluator):
        samples = [
            {"question": "Q?", "answer": "Apple revenue $85B.", "context": "Apple revenue $85 billion."},
            {"question": "EPS?", "answer": "EPS $1.40.", "context": "Unrelated context about weather."},
        ]
        results = evaluator.batch_evaluate(samples)
        stats = evaluator.summary_stats(results)
        assert stats["min_composite"] <= stats["avg_composite"] <= stats["max_composite"]
