import json

import pandas as pd
import streamlit as st


class TraceDetailCharts:
    def __init__(self, load_dataframe):
        self.load_dataframe = load_dataframe

    def latency_distribution(self):
        st.subheader("⏱️ Latency Distribution")
        df_latency = self.load_dataframe("""
            SELECT name, start_time, end_time
            FROM spans
            WHERE start_time IS NOT NULL AND end_time IS NOT NULL
              AND name != 'agent.run'
        """)
        if not df_latency.empty:
            df_latency["duration_s"] = (
                (df_latency["end_time"] - df_latency["start_time"]) / 1e9
            )
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Duration by span type**")
                latency_by_type = df_latency.groupby("name")["duration_s"].agg(
                    ["mean", "min", "max", "count"]
                ).reset_index()
                st.dataframe(latency_by_type, use_container_width=True)
            with col2:
                st.markdown("**Duration histogram (seconds)**")
                bins = pd.cut(df_latency["duration_s"], bins=20)
                hist = df_latency.groupby(bins).size().reset_index(name="count")
                hist["label"] = hist["duration_s"].apply(lambda x: f"{x.left:.1f}-{x.right:.1f}")
                st.bar_chart(hist.set_index("label")["count"])
        else:
            st.info("No latency data yet.")

    def token_usage(self):
        st.subheader("🔤 Token Usage")
        df_tokens = self.load_dataframe("""
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

    def popular_topics(self):
        st.subheader("🔥 Popular Topics")
        df_topics = self.load_dataframe("""
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

    def agent_search_patterns(self):
        st.subheader("🔄 Agent Search Patterns")
        df_iterations = self.load_dataframe("""
            SELECT query, agent_iterations, search_queries
            FROM spans
            WHERE name = 'agent.run' AND agent_iterations IS NOT NULL
            ORDER BY start_time
        """)
        if not df_iterations.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Iterations per query**")
                df_iterations["short_query"] = df_iterations["query"].str[:40] + "..."
                st.bar_chart(
                    df_iterations.set_index("short_query")["agent_iterations"],
                    use_container_width=True,
                )
            with col2:
                st.markdown("**Iteration distribution**")
                iter_counts = df_iterations["agent_iterations"].value_counts().sort_index()
                st.bar_chart(iter_counts)
            st.markdown("**Recent search patterns:**")
            for _, row in df_iterations.tail(5).iterrows():
                with st.expander(f"Query: {row['query'][:60]}..."):
                    st.markdown(f"**Iterations:** {row['agent_iterations']}")
                    try:
                        # search_queries is a JSON array of strings — pandas read_json
                        # would treat it as a file path and raise FileNotFoundError.
                        queries = json.loads(row["search_queries"])
                        for i, q in enumerate(queries, 1):
                            st.markdown(f"  {i}. `{q}`")
                    except (ValueError, TypeError):
                        st.markdown(f"  Search queries: {row['search_queries']}")
        else:
            st.info("No agent iteration data yet.")
