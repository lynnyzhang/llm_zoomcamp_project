"""Streamlit chat interface with agent transparency for LLM Zoomcamp capstone.

Features:
- Chat UI with input box and message history
- Answer display with confidence score
- Feedback buttons (thumbs up/down)
- Source documents display
- Agent process visualization: search iterations, reformulation steps, search history
- Sidebar with settings (num_results, search_type, max_iterations)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load .env explicitly — do not rely on Streamlit's implicit CWD-based dotenv
# (only runs at startup and misses .env created later / different CWD).
load_dotenv(project_root / ".env")

import streamlit as st
from src.llm import create_client, get_model
from src.rag.agent import RAGAgent
from src.search.hybrid import HybridSearch


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LLM Zoomcamp Assistant",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    num_results = st.slider(
        "Number of search results",
        min_value=1,
        max_value=10,
        value=5,
        help="How many documents to retrieve per search iteration",
    )

    search_type = st.selectbox(
        "Search type",
        options=["hybrid", "keyword", "vector"],
        index=0,
        help="Hybrid combines keyword + vector with RRF fusion",
    )

    max_iterations = st.slider(
        "Max agent iterations",
        min_value=1,
        max_value=5,
        value=3,
        help="Maximum number of search-reformulate cycles",
    )

    st.divider()
    st.markdown(f"**Model:** `{get_model()}`")
    st.markdown("**Search:** Hybrid (keyword + vector)")


# ---------------------------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "feedback" not in st.session_state:
    st.session_state.feedback = {}


# ---------------------------------------------------------------------------
# Initialize agent (lazy loading)
# ---------------------------------------------------------------------------

def get_agent() -> RAGAgent:
    """Get or create the RAG agent."""
    if st.session_state.agent is None:
        with st.spinner("Loading search index and LLM..."):
            search_index = HybridSearch()
            st.session_state.agent = RAGAgent(
                search_index=search_index,
                llm_client=create_client(),
                max_iterations=max_iterations,
            )
    return st.session_state.agent


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_confidence(searches: list) -> float:
    """Compute confidence score from search analyses.
    
    High confidence if any iteration found sufficient results.
    Score increases with more iterations finding sufficient results.
    """
    if not searches:
        return 0.0

    sufficient_count = sum(
        1 for s in searches
        if s.analysis and s.analysis.get("sufficient", False)
    )

    # Base confidence from proportion of sufficient iterations
    base_confidence = sufficient_count / len(searches)

    # Bonus for finding results in fewer iterations (more efficient)
    efficiency_bonus = 1.0 / len(searches)

    # Combine: 70% base + 30% efficiency
    confidence = 0.7 * base_confidence + 0.3 * efficiency_bonus

    return min(confidence, 1.0)


def display_search_iteration(idx: int, search) -> None:
    """Display a single search iteration with full transparency."""
    with st.expander(f"🔍 Search Iteration {idx + 1}", expanded=(idx == 0)):
        # Query used
        st.markdown(f"**Query:** `{search.query}`")

        # Results count
        st.markdown(f"**Results found:** {len(search.results)}")

        # Analysis (if available)
        if search.analysis:
            st.markdown("**Analysis:**")
            col1, col2 = st.columns(2)
            with col1:
                sufficient = search.analysis.get("sufficient", False)
                icon = "✅" if sufficient else "❌"
                st.markdown(f"{icon} Sufficient: **{sufficient}**")
            with col2:
                reason = search.analysis.get("reason", "N/A")
                st.markdown(f"Reason: {reason}")

            # Show reformulation if not sufficient
            if not sufficient:
                reformulated = search.analysis.get("reformulated_query", "")
                if reformulated:
                    st.info(f"🔄 Reformulated query: `{reformulated}`")

        # Display top results
        if search.results:
            st.markdown("**Top Results:**")
            for i, doc in enumerate(search.results[:3], 1):
                title = doc.get("title", "Untitled")
                section = doc.get("section", "")
                score = doc.get("score", 0)
                st.markdown(f"{i}. **{title}** (score: {score:.4f})")
                if section:
                    st.caption(f"Section: {section}")


def display_source_documents(searches: list) -> None:
    """Display all source documents from search iterations."""
    # Collect unique documents
    seen_ids = set()
    unique_docs = []

    for search in searches:
        for doc in search.results:
            doc_id = doc.get("id", "")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

    if not unique_docs:
        st.info("No source documents found.")
        return

    st.subheader("📚 Source Documents")

    for i, doc in enumerate(unique_docs, 1):
        with st.expander(f"{i}. {doc.get('title', 'Untitled')[:80]}"):
            st.markdown(f"**ID:** `{doc.get('id', 'N/A')}`")
            st.markdown(f"**Section:** {doc.get('section', 'N/A')}")
            st.markdown(f"**Score:** {doc.get('score', 0):.4f}")

            content = doc.get("content", "")
            if content:
                st.markdown("**Content:**")
                st.text_area(
                    "Document content",
                    value=content[:500] + ("..." if len(content) > 500 else ""),
                    disabled=True,
                    height=150,
                    key=f"doc_content_{i}",
                )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("🤖 LLM Zoomcamp Assistant")
st.caption("Agentic RAG with full transparency into search and reasoning")

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show agent process for assistant messages
        if message["role"] == "assistant" and "agent_result" in message:
            result = message["agent_result"]

            # Confidence score
            confidence = compute_confidence(result.get("searches", []))
            st.progress(confidence, text=f"Confidence: {confidence:.0%}")

            # Agent process visualization
            st.divider()
            st.subheader("🔄 Agent Process")

            iterations = result.get("iterations", 0)
            st.markdown(f"**Total iterations:** {iterations}")

            # Display each search iteration
            for idx, search in enumerate(result.get("searches", [])):
                display_search_iteration(idx, search)

            # Source documents
            display_source_documents(result.get("searches", []))

            # Feedback buttons
            st.divider()
            msg_id = message.get("msg_id", "")
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("👍", key=f"up_{msg_id}"):
                    st.session_state.feedback[msg_id] = "positive"
                    st.toast("Thanks for the feedback!")
            with col2:
                if st.button("👎", key=f"down_{msg_id}"):
                    st.session_state.feedback[msg_id] = "negative"
                    st.toast("Thanks for the feedback!")

            # Show feedback status
            if msg_id in st.session_state.feedback:
                feedback = st.session_state.feedback[msg_id]
                if feedback == "positive":
                    st.success("👍 Positive feedback recorded")
                else:
                    st.warning("👎 Negative feedback recorded")


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about the course..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent = get_agent()
                result = agent.run(prompt)

                answer = result.get("answer", "No answer generated.")
                st.markdown(answer)

                # Compute and display confidence
                confidence = compute_confidence(result.get("searches", []))
                st.progress(confidence, text=f"Confidence: {confidence:.0%}")

                # Agent process visualization
                st.divider()
                st.subheader("🔄 Agent Process")

                iterations = result.get("iterations", 0)
                st.markdown(f"**Total iterations:** {iterations}")

                # Display each search iteration
                for idx, search in enumerate(result.get("searches", [])):
                    display_search_iteration(idx, search)

                # Source documents
                display_source_documents(result.get("searches", []))

                # Feedback buttons
                st.divider()
                msg_id = f"msg_{len(st.session_state.messages)}"
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button("👍", key=f"up_{msg_id}"):
                        st.session_state.feedback[msg_id] = "positive"
                        st.toast("Thanks for the feedback!")
                with col2:
                    if st.button("👎", key=f"down_{msg_id}"):
                        st.session_state.feedback[msg_id] = "negative"
                        st.toast("Thanks for the feedback!")

                # Store message with agent result
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "msg_id": msg_id,
                    "agent_result": {
                        "answer": answer,
                        "searches": result.get("searches", []),
                        "iterations": iterations,
                    },
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
