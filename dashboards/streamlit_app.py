"""
Streamlit Monitoring Dashboard — Enterprise GenAI Platform
============================================================
Real-time monitoring of:
- Request volume and latency trends
- LLM cost tracking and budget utilization
- Cache performance
- RAG retrieval quality
- Top questions and responses
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta

import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GenAI Platform Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = st.secrets.get("API_BASE", "http://localhost:8000")
API_KEY = st.secrets.get("API_KEY", "dev-secret-key")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}


# ─────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=5)  # Refresh every 5 seconds
def fetch_metrics():
    try:
        r = requests.get(f"{API_BASE}/metrics/json", headers=HEADERS, timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


@st.cache_data(ttl=30)
def fetch_kb_stats():
    try:
        r = requests.get(f"{API_BASE}/api/v1/ingest/stats", headers=HEADERS, timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def ask_question(question: str) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/api/v1/chat/",
            headers=HEADERS,
            json={"question": question, "use_cache": False},
            timeout=60,
        )
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 GenAI Platform")
    st.caption("Enterprise Financial AI Assistant")
    st.divider()
    
    page = st.radio(
        "Navigation",
        ["📊 Monitoring", "💬 Chat Interface", "📚 Knowledge Base", "⚙️ Settings"],
    )
    
    st.divider()
    st.caption(f"API: {API_BASE}")
    
    if st.button("🔄 Refresh Metrics"):
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────────────────────
# Page: Monitoring
# ─────────────────────────────────────────────────────────────

if page == "📊 Monitoring":
    st.title("Platform Monitoring Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    metrics = fetch_metrics()
    kb_stats = fetch_kb_stats()
    
    if not metrics:
        st.warning("⚠️ Cannot connect to API. Is the platform running?")
    
    # KPI cards row 1
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Requests",
            metrics.get("total_requests", 0),
            help="Total API requests since startup",
        )
    with col2:
        st.metric(
            "LLM Calls",
            metrics.get("total_llm_calls", 0),
            help="Total LLM API calls made",
        )
    with col3:
        cost = metrics.get("total_cost_usd", 0)
        st.metric(
            "Total Cost",
            f"${cost:.4f}",
            delta=f"-${0:.4f}" if cost == 0 else None,
            help="Estimated USD cost of LLM API calls",
        )
    with col4:
        avg_latency = metrics.get("avg_latency_ms", 0)
        color = "normal" if avg_latency < 2000 else "inverse"
        st.metric("Avg Latency", f"{avg_latency:.0f}ms")
    
    # KPI cards row 2
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        hits = metrics.get("cache_hits", 0)
        misses = metrics.get("cache_misses", 0)
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0
        st.metric("Cache Hit Rate", f"{hit_rate:.1f}%", help="Redis cache effectiveness")
    with col6:
        st.metric("Errors", metrics.get("errors", 0), help="Total errors since startup")
    with col7:
        st.metric("KB Vectors", kb_stats.get("total_vectors", 0), help="Total document chunks indexed")
    with col8:
        st.metric("Embedding Dim", kb_stats.get("embedding_dimension", 384))
    
    st.divider()
    
    # Cost gauge chart
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("💰 Cost Budget Utilization")
        budget = 100.0  # Monthly budget from settings
        used = cost
        pct = min(used / budget * 100, 100)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            title={"text": f"${used:.4f} / ${budget:.2f}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "lightgreen"},
                    {"range": [50, 80], "color": "yellow"},
                    {"range": [80, 100], "color": "red"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 80,
                },
            },
            number={"suffix": "%"},
        ))
        fig.update_layout(height=300, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("📈 Cache Performance")
        hits = metrics.get("cache_hits", 0)
        misses = metrics.get("cache_misses", 0)
        
        if hits + misses > 0:
            fig2 = go.Figure(go.Pie(
                labels=["Cache Hits", "Cache Misses"],
                values=[hits, misses],
                hole=0.5,
                marker_colors=["#00CC96", "#EF553B"],
            ))
            fig2.update_layout(height=300, margin=dict(t=30, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No cache data yet. Send some queries to populate.")


# ─────────────────────────────────────────────────────────────
# Page: Chat Interface
# ─────────────────────────────────────────────────────────────

elif page == "💬 Chat Interface":
    st.title("💬 AI Financial Assistant")
    st.caption("Ask financial questions powered by RAG + Multi-step Agent")
    
    # Example questions
    st.subheader("Example Questions")
    examples = [
        "What was Apple's revenue in the most recent quarter?",
        "Compare the EPS of AAPL and MSFT for Q3 2024",
        "What are the key risk factors for technology sector stocks?",
        "Calculate the P/E ratio if stock price is $175 and EPS is $6.50",
    ]
    
    cols = st.columns(2)
    clicked_question = None
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"📋 {example}", key=f"ex_{i}"):
                clicked_question = example
    
    st.divider()
    
    # Chat input
    user_question = st.text_area(
        "Your Question",
        value=clicked_question or "",
        placeholder="Ask anything about financial data, documents, or analysis...",
        height=100,
    )
    
    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        submit = st.button("🚀 Ask", type="primary")
    with col_btn2:
        use_cache = st.checkbox("Use response cache", value=True)
    
    if submit and user_question.strip():
        with st.spinner("🤔 Agent is reasoning..."):
            start = time.time()
            response = ask_question(user_question)
            elapsed = time.time() - start
        
        if "error" in response:
            st.error(f"❌ Error: {response['error']}")
        else:
            st.success("✅ Answer generated")
            
            # Main answer
            st.subheader("🤖 Answer")
            st.markdown(response.get("answer", "No answer"))
            
            # Metadata
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Latency", f"{response.get('latency_ms', 0):.0f}ms")
            with col_m2:
                st.metric("Iterations", response.get("iterations", 0))
            with col_m3:
                st.metric("Cached", "Yes" if response.get("cached") else "No")
            with col_m4:
                risk = response.get("hallucination_risk", "N/A")
                color_map = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
                st.metric("Hallucination Risk", f"{color_map.get(risk, '⚪')} {risk}")
            
            # Tool calls
            tool_calls = response.get("tool_calls", [])
            if tool_calls:
                with st.expander(f"🔧 Tool Calls ({len(tool_calls)})"):
                    for tc in tool_calls:
                        st.code(f"{tc.get('name')}: {tc.get('arguments')}", language="json")
            
            # Session ID
            st.caption(f"Session: {response.get('session_id', 'N/A')} | Request: {response.get('request_id', 'N/A')}")


# ─────────────────────────────────────────────────────────────
# Page: Knowledge Base
# ─────────────────────────────────────────────────────────────

elif page == "📚 Knowledge Base":
    st.title("📚 Knowledge Base Management")
    
    kb_stats = fetch_kb_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Vectors", kb_stats.get("total_vectors", 0))
    with col2:
        st.metric("Embedding Model", "all-MiniLM-L6-v2")
    with col3:
        st.metric("Dimension", kb_stats.get("embedding_dimension", 384))
    
    st.divider()
    st.subheader("📁 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Upload a document to add to the knowledge base",
        type=["pdf", "txt", "md", "docx"],
    )
    
    chunking_strategy = st.selectbox(
        "Chunking Strategy",
        ["recursive_character", "sentence_aware", "fixed_size"],
    )
    
    if uploaded_file and st.button("📤 Ingest Document", type="primary"):
        with st.spinner("Ingesting document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {"chunking_strategy": chunking_strategy}
                r = requests.post(
                    f"{API_BASE}/api/v1/ingest/file",
                    headers={"X-API-Key": API_KEY},
                    files=files,
                    params=params,
                    timeout=120,
                )
                if r.status_code == 200:
                    result = r.json()
                    st.success(f"✅ Ingested! {result['chunks_added']} chunks added.")
                    st.json(result)
                    st.cache_data.clear()
                else:
                    st.error(f"Error: {r.text}")
            except Exception as e:
                st.error(f"Upload failed: {e}")


# ─────────────────────────────────────────────────────────────
# Page: Settings
# ─────────────────────────────────────────────────────────────

elif page == "⚙️ Settings":
    st.title("⚙️ Platform Settings")
    
    st.subheader("API Configuration")
    api_base = st.text_input("API Base URL", value=API_BASE)
    api_key = st.text_input("API Key", value=API_KEY, type="password")
    
    st.divider()
    st.subheader("Prompt Versions")
    
    try:
        r = requests.get(f"{API_BASE}/api/v1/prompts", headers=HEADERS, timeout=5)
        if r.status_code == 200:
            prompts = r.json()
            for prompt_type, versions in prompts.items():
                st.caption(f"**{prompt_type}**")
                for v in versions:
                    status = "✅ Active" if v["active"] else "⚪ Inactive"
                    st.write(f"  {v['version']}: {v['description']} — {status}")
        else:
            st.warning("Could not fetch prompt versions")
    except Exception as e:
        st.warning(f"Could not connect to API: {e}")
    
    st.divider()
    st.info("💡 Tip: Configure environment variables in .env file. Restart the API after changes.")
