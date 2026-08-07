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

import contextlib
import re
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
    st.markdown(f"**Search:** `{search_type}`")


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
                search_type=search_type,
            )
    else:
        # Re-wire the sidebar selection onto the cached agent (dispatch is read
        # at call time) so switching search_type skips rebuilding the ONNX index.
        st.session_state.agent.rag.search_type = search_type
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


def _unique_docs(searches: list) -> list[dict]:
    """Collect unique documents across search iterations (dedup by id)."""
    seen_ids: set[str] = set()
    unique_docs: list[dict] = []

    for search in searches:
        for doc in search.results:
            doc_id = doc.get("id", "")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

    return unique_docs


def display_source_documents(searches: list) -> None:
    """Display all source documents from search iterations."""
    unique_docs = _unique_docs(searches)

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


def _doc_artwork_url(doc: dict) -> str:
    """Official PokeAPI artwork URL for a doc's Pokémon id (pure digits)."""
    doc_id = str(doc.get("id", ""))
    pokemon_id = re.sub(r"\D", "", doc_id)
    if not pokemon_id:
        return ""
    return (
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
        f"sprites/pokemon/other/official-artwork/{pokemon_id}.png"
    )


def _stats_excerpt(content: str, limit: int = 200) -> str:
    """Short stats excerpt from doc content: the 'Stats:' line when present."""
    if not content:
        return ""
    stats_idx = content.lower().find("stats:")
    if stats_idx != -1:
        return content[stats_idx:stats_idx + limit].strip()
    return content.strip()[:limit]


def _pokemon_card_grid(docs: list[dict]) -> None:
    """Render retrieved Pokémon as artwork cards in a 4-per-row grid."""
    st.subheader("Pokémon Cards")

    for row_start in range(0, len(docs), 4):
        row_docs = docs[row_start:row_start + 4]
        columns = st.columns(4)
        for col, doc in zip(columns, row_docs):
            with col:
                title = doc.get("title", "Untitled")
                artwork_url = _doc_artwork_url(doc)
                if artwork_url:
                    # Broken/404 artwork must not break the card — the title
                    # and stats below still render without the image.
                    with contextlib.suppress(Exception):
                        st.image(artwork_url, width="stretch")
                st.markdown(f"**{title}**")
                section = doc.get("section", "")
                if section:
                    st.caption(f"Section: {section}")
                excerpt = _stats_excerpt(doc.get("content", ""))
                if excerpt:
                    st.caption(excerpt)


def render_message(msg: dict) -> None:
    """Render one chat message — the single path for history and live replies.

    Assistant messages carrying an ``agent_result`` render the answer (or a
    rejection banner), Pokémon cards, confidence, agent-process transparency,
    source documents, and the feedback buttons; all other messages render
    their content as plain markdown.
    """
    with st.chat_message(msg["role"]):
        if msg["role"] != "assistant" or "agent_result" not in msg:
            st.markdown(msg["content"])
            return

        result = msg["agent_result"]
        answer = result.get("answer", msg.get("content", ""))
        searches = result.get("searches", [])
        msg_id = msg.get("msg_id", "")

        if result.get("rejected", False):
            st.warning(answer)
        else:
            st.markdown(answer)

            unique_docs = _unique_docs(searches)
            if unique_docs:
                _pokemon_card_grid(unique_docs)

            # Confidence score
            confidence = compute_confidence(searches)
            st.progress(confidence, text=f"Confidence: {confidence:.0%}")

            # Agent process visualization
            st.divider()
            st.subheader("🔄 Agent Process")

            iterations = result.get("iterations", 0)
            st.markdown(f"**Total iterations:** {iterations}")

            for idx, search in enumerate(searches):
                display_search_iteration(idx, search)

            display_source_documents(searches)

        # Feedback buttons
        st.divider()
        col1, col2, _ = st.columns([1, 1, 8])
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
# Main UI
# ---------------------------------------------------------------------------

st.title("🤖 LLM Zoomcamp Assistant")
st.caption("Agentic RAG with full transparency into search and reasoning")

# Chat history
for message in st.session_state.messages:
    render_message(message)


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about the course..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            agent = get_agent()
            result = agent.run(prompt)

            answer = result.get("answer", "No answer generated.")

            msg_id = f"msg_{len(st.session_state.messages)}"
            message = {
                "role": "assistant",
                "content": answer,
                "msg_id": msg_id,
                "agent_result": result,
            }
            render_message(message)
            st.session_state.messages.append(message)

        except Exception as e:  # noqa: BLE001 — UI error boundary: any failure surfaces as a chat error
            error_msg = f"Error: {e!s}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })
