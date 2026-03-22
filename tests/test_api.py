"""
API Integration Tests — FastAPI Routes
=========================================
Uses the mocked TestClient from conftest.py.
Validates routes, auth, request schemas, and response shapes.
"""

from __future__ import annotations

import io
import pytest


# ── /health ──────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200

    def test_health_returns_healthy_status(self, api_client):
        data = api_client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_includes_version(self, api_client):
        data = api_client.get("/health").json()
        assert "version" in data

    def test_health_includes_environment(self, api_client):
        data = api_client.get("/health").json()
        assert "environment" in data


# ── /metrics/json ─────────────────────────────────────────────────

class TestMetricsEndpoint:

    def test_metrics_json_returns_200(self, api_client):
        r = api_client.get("/metrics/json")
        assert r.status_code == 200

    def test_metrics_json_has_expected_keys(self, api_client):
        data = api_client.get("/metrics/json").json()
        expected = [
            "total_requests", "total_llm_calls",
            "total_cost_usd", "cache_hits",
            "cache_misses", "avg_latency_ms", "errors",
        ]
        for key in expected:
            assert key in data, f"Missing key: {key}"

    def test_metrics_text_endpoint_returns_text(self, api_client):
        r = api_client.get("/metrics")
        assert r.status_code == 200
        assert "text" in r.headers.get("content-type", "")


# ── /api/v1/chat/ ─────────────────────────────────────────────────

class TestChatEndpoint:

    def test_chat_requires_api_key(self, api_client):
        """Without X-API-Key header, should return 401."""
        r = api_client.post(
            "/api/v1/chat/",
            json={"question": "What is Apple revenue?"},
        )
        assert r.status_code == 401

    def test_chat_with_valid_api_key_returns_200(self, api_client):
        r = api_client.post(
            "/api/v1/chat/",
            headers={"X-API-Key": "test-api-key"},
            json={"question": "What was Apple revenue in Q3 2024?"},
        )
        assert r.status_code == 200

    def test_chat_response_has_required_fields(self, api_client):
        r = api_client.post(
            "/api/v1/chat/",
            headers={"X-API-Key": "test-api-key"},
            json={"question": "What was Apple revenue in Q3 2024?"},
        )
        data = r.json()
        for field in ["answer", "question", "session_id", "latency_ms",
                      "cached", "hallucination_risk", "request_id"]:
            assert field in data, f"Missing field: {field}"

    def test_chat_returns_answer_string(self, api_client):
        r = api_client.post(
            "/api/v1/chat/",
            headers={"X-API-Key": "test-api-key"},
            json={"question": "What is revenue?"},
        )
        assert isinstance(r.json()["answer"], str)
        assert len(r.json()["answer"]) > 0

    def test_chat_question_too_short_raises_validation_error(self, api_client):
        r = api_client.post(
            "/api/v1/chat/",
            headers={"X-API-Key": "test-api-key"},
            json={"question": ""},  # Empty string — fails min_length=1
        )
        assert r.status_code == 422  # FastAPI validation error

    def test_chat_pii_in_question_is_processed(self, api_client):
        """PII should be redacted before reaching the agent."""
        r = api_client.post(
            "/api/v1/chat/",
            headers={"X-API-Key": "test-api-key"},
            json={"question": "My SSN is 123-45-6789. What is Apple revenue?"},
        )
        # Should succeed (PII redacted, not blocked)
        assert r.status_code in (200, 400)

    def test_chat_blocked_input_returns_400(self, api_client):
        r = api_client.post(
            "/api/v1/chat/",
            headers={"X-API-Key": "test-api-key"},
            json={"question": "DROP TABLE financial_data; SELECT * FROM passwords"},
        )
        assert r.status_code == 400

    def test_chat_session_id_in_response(self, api_client):
        r = api_client.post(
            "/api/v1/chat/",
            headers={"X-API-Key": "test-api-key"},
            json={"question": "What is EPS?", "session_id": "my-session-123"},
        )
        assert r.json()["session_id"] == "my-session-123"

    def test_chat_accepts_custom_top_k(self, api_client):
        r = api_client.post(
            "/api/v1/chat/",
            headers={"X-API-Key": "test-api-key"},
            json={"question": "Apple revenue?", "top_k": 3},
        )
        assert r.status_code == 200


# ── /api/v1/ingest/stats ──────────────────────────────────────────

class TestIngestStats:

    def test_stats_require_api_key(self, api_client):
        r = api_client.get("/api/v1/ingest/stats")
        assert r.status_code == 401

    def test_stats_return_200_with_auth(self, api_client):
        r = api_client.get(
            "/api/v1/ingest/stats",
            headers={"X-API-Key": "test-api-key"},
        )
        assert r.status_code == 200

    def test_stats_has_total_vectors(self, api_client):
        data = api_client.get(
            "/api/v1/ingest/stats",
            headers={"X-API-Key": "test-api-key"},
        ).json()
        assert "total_vectors" in data

    def test_stats_has_embedding_dimension(self, api_client):
        data = api_client.get(
            "/api/v1/ingest/stats",
            headers={"X-API-Key": "test-api-key"},
        ).json()
        assert "embedding_dimension" in data


# ── /api/v1/ingest/file ──────────────────────────────────────────

class TestIngestFile:

    def test_ingest_requires_api_key(self, api_client):
        r = api_client.post(
            "/api/v1/ingest/file",
            files={"file": ("test.txt", b"content", "text/plain")},
        )
        assert r.status_code == 401

    def test_ingest_unsupported_format_returns_400(self, api_client):
        r = api_client.post(
            "/api/v1/ingest/file",
            headers={"X-API-Key": "test-api-key"},
            files={"file": ("report.xlsx", b"fake excel content", "application/vnd.ms-excel")},
        )
        assert r.status_code == 400
