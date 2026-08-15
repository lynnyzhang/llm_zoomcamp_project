import pandas as pd
import streamlit as st

from monitoring.db_stats import get_user_feedback_stats


class SpanCharts:
    def __init__(self, load_dataframe):
        self.load_dataframe = load_dataframe

    def summary_metrics(self):
        col1, col2, col3, col4 = st.columns(4)
        df_all = self.load_dataframe("SELECT * FROM spans")
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

    def queries_over_time(self):
        st.subheader("📈 Queries Over Time")
        df_queries = self.load_dataframe("""
            SELECT
                start_time,
                name,
                query
            FROM spans
            WHERE name = 'agent.run' AND start_time IS NOT NULL
            ORDER BY start_time
        """)
        if not df_queries.empty:
            df_queries["timestamp"] = pd.to_datetime(df_queries["start_time"], unit="ns")
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

    def feedback_distribution(self):
        st.subheader("👍👎 Feedback Distribution")
        df_feedback = self.load_dataframe("""
            SELECT feedback, COUNT(*) as count
            FROM spans
            WHERE feedback IS NOT NULL
            GROUP BY feedback
        """)
        if not df_feedback.empty:
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
        st.subheader("User feedback")
        thumbs_up, thumbs_down = get_user_feedback_stats()
        col1, col2 = st.columns(2)
        col1.metric("👍 Thumbs up", thumbs_up)
        col2.metric("👎 Thumbs down", thumbs_down)
