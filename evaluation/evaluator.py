"""
RAG Evaluation Service — Evaluation Layer
==========================================
Evaluates RAG pipeline quality using:
1. Context Relevance — is the retrieved context relevant to the question?
2. Answer Faithfulness — is the answer grounded in the context?
3. Answer Relevance — does the answer address the question?

These metrics mirror the RAGAS framework.
For lightweight evaluation without LLM calls, use the heuristic variants.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Optional

from config.settings import settings
from observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    question: str
    answer: str
    context: str
    context_relevance: float    # [0, 1]
    faithfulness: float          # [0, 1]
    answer_relevance: float      # [0, 1]
    composite_score: float       # Weighted average
    latency_ms: float
    method: str                  # "heuristic" or "llm_judge"


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality.
    Provides two evaluation modes:
    - heuristic: fast, deterministic, no LLM required
    - llm_judge: uses a judge LLM for higher-quality scoring (slower, costs tokens)
    """

    def __init__(self, llm_client: Optional[object] = None) -> None:
        self._llm = llm_client

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        method: str = "heuristic",
    ) -> EvaluationResult:
        start = time.perf_counter()

        if method == "llm_judge" and self._llm:
            scores = self._llm_evaluate(question, answer, context)
        else:
            scores = self._heuristic_evaluate(question, answer, context)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Weighted composite: faithfulness most important (0.4), then each 0.3
        composite = (
            scores["context_relevance"] * 0.3
            + scores["faithfulness"] * 0.4
            + scores["answer_relevance"] * 0.3
        )

        result = EvaluationResult(
            question=question,
            answer=answer,
            context=context,
            context_relevance=round(scores["context_relevance"], 3),
            faithfulness=round(scores["faithfulness"], 3),
            answer_relevance=round(scores["answer_relevance"], 3),
            composite_score=round(composite, 3),
            latency_ms=round(elapsed_ms, 1),
            method=method,
        )
        logger.info(
            "RAG evaluation complete",
            extra={
                "composite_score": result.composite_score,
                "faithfulness": result.faithfulness,
                "method": method,
            },
        )
        return result

    def _heuristic_evaluate(
        self, question: str, answer: str, context: str
    ) -> dict:
        """
        Fast heuristic scoring. No LLM required.
        Works well for continuous monitoring — use LLM judge for spot checks.
        """
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        ctx_words = set(context.lower().split())

        # Context relevance: overlap between question and context
        q_ctx_overlap = len(q_words & ctx_words) / max(len(q_words), 1)
        context_relevance = min(q_ctx_overlap * 3, 1.0)  # Scale up, cap at 1

        # Faithfulness: what fraction of answer content appears in context
        # Use n-gram matching for better accuracy than single-word overlap
        answer_trigrams = self._get_ngrams(answer, n=3)
        context_trigrams = self._get_ngrams(context, n=3)
        if answer_trigrams:
            faith_score = len(answer_trigrams & context_trigrams) / len(answer_trigrams)
            # Boost: if context is very long, expect lower overlap but still faithful
            faithfulness = min(faith_score * 2, 1.0)
        else:
            faithfulness = 0.5  # No ngrams = short answer, assume partially faithful

        # Answer relevance: does the answer address the question's key terms?
        a_q_overlap = len(a_words & q_words) / max(len(q_words), 1)
        # Penalize answers that are too short
        length_penalty = min(len(answer.split()) / 20, 1.0)
        answer_relevance = min(a_q_overlap * 2, 1.0) * length_penalty

        return {
            "context_relevance": context_relevance,
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
        }

    def _llm_evaluate(self, question: str, answer: str, context: str) -> dict:
        """
        Use a judge LLM to score RAG quality.
        More accurate but slower and costs tokens.
        """
        prompt = f"""
You are evaluating a RAG system. Score each dimension from 0.0 to 1.0.

Question: {question}
Context: {context[:1000]}
Answer: {answer[:500]}

Score these dimensions:
1. Context Relevance (0.0-1.0): How relevant is the context to the question?
2. Faithfulness (0.0-1.0): Is the answer grounded in the provided context?
3. Answer Relevance (0.0-1.0): Does the answer address the question?

Respond ONLY with JSON: {{"context_relevance": X.X, "faithfulness": X.X, "answer_relevance": X.X}}
"""
        try:
            response = self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            import json
            scores = json.loads(response.content)
            return {
                "context_relevance": float(scores.get("context_relevance", 0.5)),
                "faithfulness": float(scores.get("faithfulness", 0.5)),
                "answer_relevance": float(scores.get("answer_relevance", 0.5)),
            }
        except Exception as exc:
            logger.warning(f"LLM judge failed: {exc}. Using heuristics.")
            return self._heuristic_evaluate(question, answer, context)

    @staticmethod
    def _get_ngrams(text: str, n: int = 3) -> set:
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        return set(
            " ".join(words[i: i + n])
            for i in range(len(words) - n + 1)
        )

    def batch_evaluate(
        self,
        samples: List[dict],
        method: str = "heuristic",
    ) -> List[EvaluationResult]:
        """Evaluate a batch of Q&A samples."""
        return [
            self.evaluate(
                question=s["question"],
                answer=s["answer"],
                context=s.get("context", ""),
                method=method,
            )
            for s in samples
        ]

    def summary_stats(self, results: List[EvaluationResult]) -> dict:
        """Aggregate stats across a batch evaluation."""
        if not results:
            return {}
        n = len(results)
        return {
            "n": n,
            "avg_composite": round(sum(r.composite_score for r in results) / n, 3),
            "avg_faithfulness": round(sum(r.faithfulness for r in results) / n, 3),
            "avg_context_relevance": round(sum(r.context_relevance for r in results) / n, 3),
            "avg_answer_relevance": round(sum(r.answer_relevance for r in results) / n, 3),
            "min_composite": min(r.composite_score for r in results),
            "max_composite": max(r.composite_score for r in results),
        }
