# 🏦 Enterprise GenAI Platform

> **AI Financial Assistant** — Production-grade Retrieval-Augmented Generation platform built with Python, FastAPI, LangGraph, and FAISS.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](docker-compose.yml)

---

## 📋 Problem Statement

Financial services firms produce and consume vast amounts of unstructured data: SEC filings, earnings reports, research notes, policy documents, and market commentary. Analysts and advisors currently spend 60–70% of their time *finding* information rather than *using* it.

This platform solves that by combining:
- **RAG** (semantic search over documents) for qualitative questions
- **SQL tools** for quantitative metrics from structured data
- **Multi-step agents** for complex multi-part analysis
- **Responsible AI guardrails** for regulatory compliance

**Business Use Case:** An AI Financial Assistant that answers questions from a firm's private document Knowledge Base and financial data warehouse, with full audit logging and cost control.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│         Web App   │   Streamlit Dashboard   │   REST API        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                     NGINX (Reverse Proxy)                       │
│              Rate Limiting │ SSL Termination │ Load Balancing   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                   API GATEWAY (FastAPI)                         │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│ │  /chat       │ │  /ingest     │ │  /metrics  /health       │ │
│ └──────┬───────┘ └──────┬───────┘ └──────────────────────────┘ │
│        │  Guardrails Pipeline                                    │
│        │  ┌─────────────┐  ┌──────────────┐                    │
│        │  │ InputFilter │  │ OutputFilter │                    │
│        │  └─────────────┘  └──────────────┘                    │
└────────┼────────────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────────────┐
│                    AGENT FRAMEWORK (LangGraph)                  │
│                                                                 │
│  ┌─────────┐     ┌──────────────┐     ┌─────────────────────┐ │
│  │  START  │────▶│ Agent (LLM)  │────▶│  Tool Executor      │ │
│  └─────────┘     └──────┬───────┘     │  • DocumentRetrieval│ │
│                         │             │  • SQLQuery         │ │
│                  ┌──────▼───────┐     │  • Calculator       │ │
│                  │   FINISH     │◀────└─────────────────────┘ │
│                  └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
         │                    │
┌────────▼────────┐   ┌───────▼─────────────────────────────────┐
│  RETRIEVAL      │   │  DATA LAYER                             │
│  ┌───────────┐  │   │  ┌──────────────┐  ┌─────────────────┐ │
│  │  FAISS    │  │   │  │  PostgreSQL  │  │    DuckDB       │ │
│  │  Vector   │  │   │  │  (Audit Log, │  │  (Financial     │ │
│  │  Store    │  │   │  │   Documents) │  │   Metrics SQL)  │ │
│  └───────────┘  │   │  └──────────────┘  └─────────────────┘ │
│  ┌───────────┐  │   └─────────────────────────────────────────┘
│  │   BM25    │  │
│  │  (Hybrid) │  │   ┌─────────────────────────────────────────┐
│  └───────────┘  │   │  INFRASTRUCTURE                         │
└─────────────────┘   │  ┌──────────┐ ┌─────────┐ ┌─────────┐ │
                       │  │  Redis   │ │ MLflow  │ │Prometheus│ │
┌─────────────────┐   │  │  (Cache) │ │ (MLOps) │ │ Grafana │ │
│  INGESTION      │   │  └──────────┘ └─────────┘ └─────────┘ │
│  ┌───────────┐  │   └─────────────────────────────────────────┘
│  │  Loader   │  │
│  │  Chunker  │  │
│  │  Embedder │  │
│  └───────────┘  │
└─────────────────┘
```

---

## 📁 Repository Structure

```
enterprise-genai-platform/
├── api/                        # FastAPI gateway
│   ├── main.py                 # App factory, middleware, lifespan
│   ├── dependencies.py         # DI singletons (embedder, retriever, agent)
│   └── routes/
│       ├── chat.py             # POST /chat — agent Q&A
│       └── ingest.py           # POST /ingest/file — document upload
│
├── ingestion/                  # Document → Vector pipeline
│   ├── document_loader.py      # PDF (PyMuPDF), DOCX, TXT/MD loaders
│   ├── chunker.py              # 3 chunking strategies + factory
│   ├── embedder.py             # SentenceTransformers / OpenAI / Azure
│   ├── vector_store.py         # FAISS IndexFlatIP with persistence
│   └── pipeline.py             # Orchestrates load→chunk→embed→store
│
├── retrieval/                  # Semantic search layer
│   └── retriever.py            # Dense + BM25 hybrid + cross-encoder reranker
│
├── agents/                     # Multi-step agent framework
│   ├── tools.py                # DocumentRetrieval, SQL, Calculator tools
│   ├── graph.py                # LangGraph StateGraph (ReAct workflow)
│   ├── agent.py                # Orchestrator with Redis caching
│   └── llm_client.py           # OpenAI / Anthropic / Ollama abstraction
│
├── prompts/                    # Version-controlled prompts
│   └── templates.py            # System prompts v1/v2, RAG, agent templates
│
├── guardrails/                 # Responsible AI
│   ├── input_validator.py      # PII detection, injection blocking, length limits
│   └── output_filter.py        # Content filter, disclaimer injection, hallucination risk
│
├── observability/              # Telemetry
│   ├── logger.py               # JSON structured logging + correlation IDs
│   ├── metrics.py              # Prometheus-compatible metrics + cost tracking
│   └── tracer.py               # Distributed tracing (OpenTelemetry interface)
│
├── mlops/                      # MLOps layer
│   └── experiment_tracker.py   # MLflow integration + PipelineConfig
│
├── evaluation/                 # RAG quality evaluation
│   └── evaluator.py            # RAGAS-style: faithfulness, relevance, context
│
├── config/                     # Central settings
│   └── settings.py             # Pydantic BaseSettings with env var loading
│
├── dashboards/
│   └── streamlit_app.py        # Monitoring dashboard + chat UI
│
├── data/
│   ├── schemas/init.sql         # PostgreSQL schema + sample data
│   └── sample_documents/        # Financial reports for demo
│
├── infra/
│   ├── Dockerfile.api           # Production API container
│   └── nginx.conf               # Reverse proxy + rate limiting
│
├── docker-compose.yml           # Full stack: API, Redis, PostgreSQL, MLflow
├── requirements.txt             # Pinned dependencies
├── .env.example                 # Environment variable template
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.11+
- Docker + Docker Compose (for full stack)
- OpenAI API key (or Anthropic, or local Ollama)

### Option A: Local Development (Python venv)

```bash
# 1. Clone and enter the project
cd enterprise-genai-platform

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and PLATFORM_API_KEY at minimum

# 5. Start infrastructure (Redis + PostgreSQL + MLflow)
docker-compose up redis postgres mlflow -d

# 6. Start the API
python -m uvicorn api.main:app --reload --port 8000

# 7. (Optional) Start the Streamlit dashboard
streamlit run dashboards/streamlit_app.py --server.port 8501
```

### Option B: Full Docker Stack

```bash
# Start all services
docker-compose up -d

# With observability (Prometheus + Grafana)
docker-compose --profile observability up -d

# View logs
docker-compose logs -f api

# Stop everything
docker-compose down
```

---

## 🚀 How to Run — Step by Step

### Step 1: Ingest a Document

```bash
# Ingest the sample Apple earnings report
curl -X POST "http://localhost:8000/api/v1/ingest/file" \
  -H "X-API-Key: dev-secret-key" \
  -F "file=@data/sample_documents/apple_q3_2024_earnings.txt" \
  -F "chunking_strategy=recursive_character"

# Response:
# {
#   "doc_id": "txt-a1b2c3d4",
#   "filename": "apple_q3_2024_earnings.txt",
#   "chunks_created": 18,
#   "chunks_added": 18,
#   "elapsed_seconds": 2.14,
#   "success": true
# }
```

### Step 2: Ask a Financial Question

```bash
curl -X POST "http://localhost:8000/api/v1/chat/" \
  -H "X-API-Key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was Apple revenue in Q3 2024 and how did services perform?",
    "use_cache": true
  }'
```

**Response:**

```json
{
  "answer": "Apple reported total net revenue of **$85.8 billion** in Q3 FY2024...\n\n[Source 1: apple_q3_2024_earnings.txt | Relevance: 0.91]\n\n---\n*Disclaimer: This analysis is for informational purposes only...*",
  "question": "What was Apple revenue in Q3 2024?",
  "session_id": "session-abc123",
  "tool_calls": [{"name": "document_retrieval", "arguments": "{\"query\": \"Apple revenue Q3 2024\"}"}],
  "iterations": 2,
  "latency_ms": 1842.3,
  "cached": false,
  "hallucination_risk": "LOW",
  "disclaimer_added": true,
  "request_id": "req-uuid"
}
```

### Step 3: Query Financial Metrics via SQL

```bash
curl -X POST "http://localhost:8000/api/v1/chat/" \
  -H "X-API-Key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Compare the EPS of Apple and Microsoft for Q3 2024 from the database"
  }'
```

### Step 4: Multi-Step Calculation

```bash
curl -X POST "http://localhost:8000/api/v1/chat/" \
  -H "X-API-Key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "If Apple stock is at $175 and EPS is $1.40, what is the P/E ratio and how does it compare to a sector average of 28?"
  }'
# Agent will: retrieve context → query SQL → use calculator → synthesize answer
```

### Step 5: Check Metrics

```bash
# Prometheus format
curl http://localhost:8000/metrics

# JSON snapshot
curl -H "X-API-Key: dev-secret-key" http://localhost:8000/metrics/json
```

### Step 6: Knowledge Base Stats

```bash
curl -H "X-API-Key: dev-secret-key" \
  http://localhost:8000/api/v1/ingest/stats
```

---

## 📊 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat/` | Ask a financial question (agent flow) |
| `POST` | `/api/v1/ingest/file` | Upload and ingest a document |
| `DELETE` | `/api/v1/ingest/{doc_id}` | Remove a document from knowledge base |
| `GET` | `/api/v1/ingest/stats` | Knowledge base statistics |
| `GET` | `/api/v1/prompts` | List prompt versions |
| `GET` | `/health` | Platform health check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/metrics/json` | JSON metrics snapshot |
| `GET` | `/api/v1/docs` | Swagger UI |

---

## 🔑 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Vector Index** | FAISS IndexFlatIP | Exact cosine similarity; IndexIVFFlat used at >10K vectors |
| **Chunking** | Recursive Character (default) | Sentence-aware; better recall than fixed-size |
| **Embedding** | SentenceTransformers (local) | No API cost; swap to OpenAI for higher accuracy |
| **Hybrid Search** | Dense + BM25 via RRF | Combines semantic and keyword recall |
| **Agent Pattern** | LangGraph ReAct | Explicit state machine; auditable reasoning steps |
| **PII Handling** | Redact before LLM call | Never send PII to third-party APIs |
| **Caching** | Redis with SHA-256 key | Identical questions reuse LLM responses |
| **Config** | Pydantic BaseSettings | Type-safe, env-var driven, zero code changes for env switch |

---

## 📈 Scaling Considerations

### Horizontal Scaling (10K RPS)
- **API**: Deploy multiple uvicorn workers behind Nginx (already configured)
- **Vector Store**: Migrate FAISS to Pinecone or Weaviate for distributed search
- **Embeddings**: Move to GPU instance or OpenAI batch API
- **Caching**: Redis Cluster for distributed cache

### Vertical Scaling (Large Documents)
- Increase `CHUNK_SIZE` for models with larger context windows (GPT-4o: 128K)
- Enable `ENABLE_HYBRID_SEARCH` for better recall on large corpora
- Use `IndexIVFFlat` + GPU-FAISS for million-scale vector stores

### Multi-Tenant Production
- Add user/org scoping to vector store metadata filters
- Implement JWT authentication (replace API key header)
- Namespace Redis keys per tenant
- Row-level security in PostgreSQL

### Cost Optimization
- `gpt-4o-mini` for retrieval-enriched queries (~10x cheaper than gpt-4o)
- Redis response caching eliminates ~30-40% of LLM calls in practice
- Batch embeddings with SentenceTransformers (local, $0 cost)
- Set `MONTHLY_BUDGET_USD` alerts in settings

---

## ⚖️ Tradeoffs

| Tradeoff | Current Choice | Production Alternative |
|----------|---------------|----------------------|
| Exact vs Approximate Search | FAISS Flat (exact) | FAISS IVF / Pinecone (approximate, faster) |
| Local vs API embeddings | SentenceTransformers | OpenAI (higher quality, latency, cost) |
| Sync vs Async ingestion | Synchronous | Celery + Redis queue for async |
| In-memory metrics | Custom MetricsCollector | Prometheus client + push gateway |
| Simple auth | API key header | OAuth 2.0 / JWT + RBAC |
| Single-tenant | Shared FAISS index | Multi-tenant namespaced index |

---

## 🔮 Future Improvements

1. **Multi-modal RAG**: Support images, tables extracted from PDFs via OCR + CLIP embeddings
2. **Streaming responses**: SSE/WebSocket for real-time token streaming
3. **Query routing**: Fast classifier to route simple factual vs complex analytical questions
4. **Automated reingestion**: Airflow DAG to detect and re-index updated documents
5. **Red teaming**: Automated adversarial input testing against guardrails
6. **Fine-tuning**: Domain-specific embedding model fine-tuned on financial corpus
7. **A/B testing**: MLflow + feature flags for controlled prompt version experiments
8. **Memory**: Persistent long-term conversation memory via PostgreSQL
9. **Agentic workflows**: Multi-agent coordination for report generation pipelines
10. **Compliance reports**: Automated PDF reports of guardrail activations for regulatory review

---

## 🧪 Running Tests

```bash
# Unit tests
pytest tests/ -v --cov=. --cov-report=html

# Test the RAG evaluator
python -c "
from evaluation.evaluator import RAGEvaluator
evaluator = RAGEvaluator()
result = evaluator.evaluate(
    question='What is Apple revenue?',
    answer='Apple revenue was 85.8 billion in Q3 2024',
    context='Apple reported revenue of 85.8 billion in Q3 FY2024',
    method='heuristic'
)
print(f'Composite: {result.composite_score}')
print(f'Faithfulness: {result.faithfulness}')
"
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE).

---

> Built with ❤️ using FastAPI, LangChain, LangGraph, FAISS, SentenceTransformers, Redis, PostgreSQL, DuckDB, and MLflow.  
> Designed for production. Tested for enterprise.
