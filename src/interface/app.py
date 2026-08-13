import argparse
import contextlib
import logging
import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load .env explicitly — do not rely on Streamlit's implicit CWD-based dotenv
# (only runs at startup and misses .env created later / different CWD).
load_dotenv(project_root / ".env")

import streamlit as st

from src.llm import LLMClient
from src.rag.agent import RAGAgent
from src.search.hybrid import HybridSearch


def _parse_cli_flags(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--show-confidence", action="store_true", default=False)
    args, _ = parser.parse_known_args(argv)
    return args.show_confidence


# Read once at import; --show-confidence toggles the LLM-judge confidence bar.
SHOW_CONFIDENCE = _parse_cli_flags(sys.argv[1:])

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pokémon Assistant",
    page_icon="🤖",
    layout="wide",
)


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

def _make_agent():
    search_index = HybridSearch()
    return RAGAgent(
        search_index=search_index,
        llm_client=LLMClient.get(),
    )


def _maybe_trace(agent):
    # Fall back to the plain agent when tracing is disabled or the tracer
    # cannot be initialized (an unwritable data/ must never break the app).
    try:
        from monitoring.tracer import TracedRAGAgent, tracing_enabled

        if tracing_enabled():
            return TracedRAGAgent(agent)
    except Exception:
        logging.getLogger(__name__).warning(
            "Tracing disabled — falling back to plain agent", exc_info=True
        )
    return agent


def get_agent():
    if st.session_state.agent is None:
        with st.spinner("Loading search index and LLM..."):
            st.session_state.agent = _maybe_trace(_make_agent())
    return st.session_state.agent


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _unique_docs(searches):
    seen_ids = set()
    unique_docs = []

    for search in searches:
        for doc in search.results:
            # Only doc-shaped local results render as cards; web results
            # (title/url/snippet) carry no "id" and must never be shown.
            if not isinstance(doc, dict) or "id" not in doc:
                continue
            doc_id = doc.get("id", "")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

    return unique_docs


def _filter_docs_by_question(unique_docs, question):
    if not question:
        return []

    question_lower = question.lower()
    matched = []

    for doc in unique_docs:
        name = doc.get("name", "")
        if name and name.lower() in question_lower:
            matched.append(doc)

    return matched


def _doc_artwork_url(doc):
    sprite_url = doc.get("sprite_url")
    if sprite_url:
        return sprite_url
    # Fallback: build the PokeAPI official-artwork URL from the numeric part of
    # the id (used when a doc has no sprite_url, e.g. type-chart docs).
    doc_id = str(doc.get("id", ""))
    pokemon_id = re.sub(r"\D", "", doc_id)
    if not pokemon_id:
        return ""
    return (
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
        f"sprites/pokemon/other/official-artwork/{pokemon_id}.png"
    )


def _stats_excerpt(doc, limit=200):
    # Type-chart docs carry no stats.
    if doc.get("kind") == "type_chart":
        return ""
    stats = doc.get("stats") or {}
    if not stats:
        return ""
    parts = [
        f"hp {stats.get('hp', 0)}",
        f"attack {stats.get('attack', 0)}",
        f"defense {stats.get('defense', 0)}",
        f"sp. attack {stats.get('sp_attack', 0)}",
        f"sp. defense {stats.get('sp_defense', 0)}",
        f"speed {stats.get('speed', 0)}",
    ]
    return ", ".join(parts)[:limit]


def _card_title(doc):
    if doc.get("kind") == "type_chart":
        return f"{doc.get('type') or 'Unknown'} type chart"
    name = doc.get("name", "Untitled")
    doc_id = doc.get("id")
    if isinstance(doc_id, int):
        return f"{name} (#{doc_id})"
    return name


def _card_caption(doc):
    if doc.get("kind") == "type_chart":
        return "Type chart"
    types = doc.get("types") or []
    if types:
        return " + ".join(types)
    return "unknown"


def _pokemon_card_grid(docs):
    st.subheader("Pokémon Cards")

    for row_start in range(0, len(docs), 4):
        row_docs = docs[row_start:row_start + 4]
        columns = st.columns(4)
        for col, doc in zip(columns, row_docs):
            with col:
                title = _card_title(doc)
                artwork_url = _doc_artwork_url(doc)
                if artwork_url:
                    # Broken/404 artwork must not break the card — the title
                    # and stats below still render without the image.
                    with contextlib.suppress(Exception):
                        st.image(artwork_url, width="stretch")
                st.markdown(f"**{title}**")
                caption = _card_caption(doc)
                if caption:
                    st.caption(caption)
                excerpt = _stats_excerpt(doc)
                if excerpt:
                    st.caption(excerpt)


def _record_feedback(span_id, feedback):
    # Persist feedback for a message's span; never crash the UI. Messages
    # without a span (untraced agent or pre-tracing history) are skipped;
    # write failures (unwritable database) are logged and swallowed.
    if not span_id:
        return
    try:
        from monitoring.tracer import record_feedback

        record_feedback(span_id, feedback)
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to record feedback", exc_info=True
        )


# ---------------------------------------------------------------------------
# Render helpers — separated so the live path can render inside one bubble
# ---------------------------------------------------------------------------

def render_message_body(msg):
    """Render the *inner* content of an assistant message (no outer bubble)."""
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

        source = result.get("source")
        if source == "local":
            st.caption("Source: Local knowledge base")
        elif source == "web":
            st.caption("Source: Bulbapedia (web)")

        unique_docs = _unique_docs(searches)
        question = msg.get("question", "")
        matched_docs = _filter_docs_by_question(unique_docs, question) if question else []
        if matched_docs:
            _pokemon_card_grid(matched_docs)

        if SHOW_CONFIDENCE:
            confidence = result.get("confidence")
            if confidence is not None:
                st.progress(confidence, text=f"Confidence: {confidence:.0%}")

    # Feedback buttons
    st.divider()
    col1, col2, _ = st.columns([1, 1, 8])
    with col1:
        if st.button("👍", key=f"up_{msg_id}"):
            st.session_state.feedback[msg_id] = "positive"
            _record_feedback(msg.get("span_id"), "positive")
            st.toast("Thanks for the feedback!")
    with col2:
        if st.button("👎", key=f"down_{msg_id}"):
            st.session_state.feedback[msg_id] = "negative"
            _record_feedback(msg.get("span_id"), "negative")
            st.toast("Thanks for the feedback!")

    # Show feedback status
    if msg_id in st.session_state.feedback:
        feedback = st.session_state.feedback[msg_id]
        if feedback == "positive":
            st.success("👍 Positive feedback recorded")
        else:
            st.warning("👎 Negative feedback recorded")


def render_message(msg):
    with st.chat_message(msg["role"]):
        render_message_body(msg)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("🤖 Pokémon Assistant")
st.caption("Agentic RAG assistant for Pokémon knowledge")

# Chat history
for message in st.session_state.messages:
    render_message(message)


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about Pokémon..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent = get_agent()
                if hasattr(agent, "run_with_feedback"):
                    result, span_id = agent.run_with_feedback(prompt)
                else:
                    result = agent.run(prompt)
                    span_id = None

                answer = result.get("answer", "No answer generated.")

                msg_id = f"msg_{len(st.session_state.messages)}"
                message = {
                    "role": "assistant",
                    "content": answer,
                    "msg_id": msg_id,
                    "agent_result": result,
                    "span_id": span_id,
                    "question": prompt,
                }

            except Exception as e:  # noqa: BLE001 — UI error boundary: any failure surfaces as a chat error
                error_msg = f"Error: {e!s}"
                st.error(error_msg)
                message = {
                    "role": "assistant",
                    "content": error_msg,
                }

        # Render inside the SAME bubble — after spinner finishes
        if "agent_result" in message:
            render_message_body(message)
            st.session_state.messages.append(message)
        elif message.get("content", "").startswith("Error:"):
            # Error was already shown via st.error() above; still record in history.
            st.session_state.messages.append(message)
