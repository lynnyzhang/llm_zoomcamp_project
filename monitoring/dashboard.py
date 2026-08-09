import sqlite3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import streamlit as st

from .tracer import get_traces_db_path

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LLM Zoomcamp Monitoring",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def load_dataframe(query):
    # Opens a fresh connection per call: SQLite connections are thread-bound,
    # so a cached one would break reruns from a different thread.
    conn = sqlite3.connect(str(get_traces_db_path()))
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Check if database exists
# ---------------------------------------------------------------------------

db_path = get_traces_db_path()
if not db_path.exists():
    st.warning(
        f"No traces database found at `{db_path}`. "
        "Run the tracer first to generate trace data."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Dashboard layout
# ---------------------------------------------------------------------------

st.title("📊 LLM Zoomcamp Monitoring Dashboard")
st.caption("Analytics from OpenTelemetry traces stored in SQLite")

# Summary metrics
col1, col2, col3, col4 = st.columns(4)

df_all = load_dataframe("SELECT * FROM spans")
total_traces = len(df_all)
total_agent_runs = (
    len(df_all[df_all["name"] == "agent.run"]) if "name" in df_all.columns else 0
)
total_cost = df_all["cost"].sum() if "cost" in df_all.columns else 0
avg_latency = 0.0
if "start_time" in df_all.columns and "end_time" in df_all.columns:
    mask = df_all["end_time"].notna() & df_all["start_time"].notna()
    if mask.any():
        durations = (df_all.loc[mask, "end_time"] - df_all.loc[mask, "start_time"]) / 1e9
        avg_latency = durations.mean()

col1.metric("Total Traces", total_traces)
col2.metric("Agent Runs", total_agent_runs)
col3.metric("Total Cost", f"${total_cost:.4f}")
col4.metric("Avg Latency", f"{avg_latency:.2f}s")

st.divider()

# ---------------------------------------------------------------------------
# Chart 1: Queries over time
# ---------------------------------------------------------------------------

st.subheader("📈 Queries Over Time")

df_queries = load_dataframe("""
    SELECT
        start_time,
        name,
        query
    FROM spans
    WHERE name = 'agent.run' AND start_time IS NOT NULL
    ORDER BY start_time
""")

if not df_queries.empty:
    # Convert nanoseconds to datetime
    df_queries["timestamp"] = pd.to_datetime(df_queries["start_time"], unit="ns")

    # Group by time bucket (minute or hour depending on data density)
    time_range = df_queries["timestamp"].max() - df_queries["timestamp"].min()
    if time_range.total_seconds() < 3600:
        df_queries["time_bucket"] = df_queries["timestamp"].dt.floor("1min")
    else:
        df_queries["time_bucket"] = df_queries["timestamp"].dt.floor("1h")

    queries_over_time = df_queries.groupby("time_bucket").size().reset_index(name="count")

    st.line_chart(
        queries_over_time.set_index("time_bucket")["count"],
        use_container_width=True,
    )
else:
    st.info("No agent.run traces yet.")

st.divider()

# ---------------------------------------------------------------------------
# Chart 2: Feedback distribution
# ---------------------------------------------------------------------------

st.subheader("👍👎 Feedback Distribution")

df_feedback = load_dataframe("""
    SELECT feedback, COUNT(*) as count
    FROM spans
    WHERE feedback IS NOT NULL
    GROUP BY feedback
""")

if not df_feedback.empty:
    # Create pie chart data
    feedback_data = dict(zip(df_feedback["feedback"], df_feedback["count"]))

    col1, col2 = st.columns([1, 2])
    with col1:
        st.bar_chart(df_feedback.set_index("feedback")["count"])
    with col2:
        total = sum(feedback_data.values())
        positive = feedback_data.get("positive", 0)
        negative = feedback_data.get("negative", 0)
        st.metric("Positive", positive, f"{positive/total*100:.0f}%")
        st.metric("Negative", negative, f"{negative/total*100:.0f}%")
else:
    st.info("No feedback recorded yet.")

st.divider()

# ---------------------------------------------------------------------------
# Chart 3: Latency distribution
# ---------------------------------------------------------------------------

st.subheader("⏱️ Latency Distribution")

df_latency = load_dataframe("""
    SELECT name, start_time, end_time
    FROM spans
    WHERE start_time IS NOT NULL AND end_time IS NOT NULL
      AND name != 'agent.run'
""")

if not df_latency.empty:
    df_latency["duration_s"] = (
        (df_latency["end_time"] - df_latency["start_time"]) / 1e9
    )

    # Histogram of durations
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Duration by span type**")
        latency_by_type = df_latency.groupby("name")["duration_s"].agg(
            ["mean", "min", "max", "count"]
        ).reset_index()
        st.dataframe(latency_by_type, use_container_width=True)

    with col2:
        st.markdown("**Duration histogram (seconds)**")
        # Create histogram bins
        bins = pd.cut(df_latency["duration_s"], bins=20)
        hist = df_latency.groupby(bins).size().reset_index(name="count")
        hist["label"] = hist["duration_s"].apply(lambda x: f"{x.left:.1f}-{x.right:.1f}")
        st.bar_chart(hist.set_index("label")["count"])
else:
    st.info("No latency data yet.")

st.divider()

# ---------------------------------------------------------------------------
# Chart 4: Token usage
# ---------------------------------------------------------------------------

st.subheader("🔤 Token Usage")

df_tokens = load_dataframe("""
    SELECT name, input_tokens, output_tokens
    FROM spans
    WHERE input_tokens IS NOT NULL
""")

if not df_tokens.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Input vs Output tokens per span**")
        st.bar_chart(
            df_tokens.set_index("name")[["input_tokens", "output_tokens"]],
            use_container_width=True,
        )

    with col2:
        st.markdown("**Token totals**")
        total_input = df_tokens["input_tokens"].sum()
        total_output = df_tokens["output_tokens"].sum()
        st.metric("Total Input Tokens", f"{total_input:,}")
        st.metric("Total Output Tokens", f"{total_output:,}")
        st.metric("Total Tokens", f"{(total_input + total_output):,}")
else:
    st.info("No token usage data yet.")

st.divider()

# ---------------------------------------------------------------------------
# Chart 5: Popular topics (top queries)
# ---------------------------------------------------------------------------

st.subheader("🔥 Popular Topics")

df_topics = load_dataframe("""
    SELECT query, COUNT(*) as count
    FROM spans
    WHERE query IS NOT NULL AND name = 'agent.run'
    GROUP BY query
    ORDER BY count DESC
    LIMIT 20
""")

if not df_topics.empty:
    st.bar_chart(
        df_topics.set_index("query")["count"],
        use_container_width=True,
    )
else:
    st.info("No query data yet.")

st.divider()

# ---------------------------------------------------------------------------
# Chart 6: Agent search patterns
# ---------------------------------------------------------------------------

st.subheader("🔄 Agent Search Patterns")

df_iterations = load_dataframe("""
    SELECT query, agent_iterations, search_queries
    FROM spans
    WHERE name = 'agent.run' AND agent_iterations IS NOT NULL
    ORDER BY start_time
""")

if not df_iterations.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Iterations per query**")
        # Truncate query for display
        df_iterations["short_query"] = df_iterations["query"].str[:40] + "..."
        st.bar_chart(
            df_iterations.set_index("short_query")["agent_iterations"],
            use_container_width=True,
        )

    with col2:
        st.markdown("**Iteration distribution**")
        iter_counts = df_iterations["agent_iterations"].value_counts().sort_index()
        st.bar_chart(iter_counts)

    # Show recent queries with search patterns
    st.markdown("**Recent search patterns:**")
    for _, row in df_iterations.tail(5).iterrows():
        with st.expander(f"Query: {row['query'][:60]}..."):
            st.markdown(f"**Iterations:** {row['agent_iterations']}")
            try:
                queries = pd.read_json(row["search_queries"])
                for i, q in enumerate(queries, 1):
                    st.markdown(f"  {i}. `{q}`")
            except (ValueError, TypeError):
                st.markdown(f"  Search queries: {row['search_queries']}")
else:
    st.info("No agent iteration data yet.")

st.divider()

# ---------------------------------------------------------------------------
# Raw data explorer
# ---------------------------------------------------------------------------

st.subheader("📋 Raw Trace Data")

with st.expander("Show all traces"):
    st.dataframe(df_all, use_container_width=True, height=300)

# Manual refresh from the sidebar (st.auto_refresh is not a Streamlit API)
with st.sidebar:
    st.caption("Data is loaded at script run time.")
    if st.button("🔄 Refresh Data"):
        st.rerun()
