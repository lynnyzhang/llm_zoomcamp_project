import pandas as pd
import streamlit as st

from monitoring.db_query import get_conversations


class HistoryCharts:
    def __init__(self, load_dataframe):
        self.load_dataframe = load_dataframe

    def recent_conversations(self):
        st.subheader("💬 Recent Conversations")
        records = get_conversations(limit=20)
        if not records:
            st.info("No conversations yet. Ask something in the chat app first.")
        else:
            rows = []
            for record in records:
                question = getattr(record, "question", None) or record.prompt or ""
                rows.append({
                    "question": question[:200],
                    "answer": record.answer[:200] + "...",
                    "model": record.model,
                    "tokens": record.total_tokens,
                    "response_time": round(record.response_time or 0, 2),
                    "cost": round(record.cost or 0, 4),
                    "source": record.source,
                    "rejected": record.rejected,
                    "error": getattr(record, "error", None) or "",
                    "timestamp": record.timestamp,
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
            )

    def gated_query_rate(self):
        st.subheader("🛡️ Gated Query Rate")
        df_gated = self.load_dataframe("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE rejected = 1 OR error IS NOT NULL) AS gated,
                COUNT(*) FILTER (WHERE error IS NOT NULL) AS errors
            FROM conversations
        """)
        if not df_gated.empty and df_gated.iloc[0]["total"] > 0:
            total = df_gated.iloc[0]["total"]
            gated = df_gated.iloc[0]["gated"]
            errors = df_gated.iloc[0]["errors"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total turns", total)
            col2.metric("Gated turns", gated)
            col3.metric("Gated rate", f"{gated / total * 100:.1f}%")
            col4.metric("Error turns", errors)
            st.caption("Gated = out-of-scope rejections and failed turns — the guardrail layer.")
            df_gated_rate = self.load_dataframe("""
                SELECT date_trunc('hour', timestamp) AS bucket,
                       COUNT(*) FILTER (WHERE rejected = 1 OR error IS NOT NULL)::float
                           / NULLIF(COUNT(*), 0) AS gated_rate
                FROM conversations
                GROUP BY 1
                ORDER BY 1
            """)
            if not df_gated_rate.empty:
                st.line_chart(
                    df_gated_rate.set_index("bucket")["gated_rate"],
                    use_container_width=True,
                )
        else:
            st.info("No conversations yet.")

    def answer_path_mix(self):
        st.subheader("🔀 Answer Path Mix")
        df_path = self.load_dataframe("""
            SELECT source, COUNT(*) AS count
            FROM conversations
            WHERE source IS NOT NULL
            GROUP BY source
        """)
        df_rejected = self.load_dataframe("""
            SELECT COUNT(*) AS rejected
            FROM conversations
            WHERE source IS NULL
        """)
        if not df_path.empty or (not df_rejected.empty and df_rejected.iloc[0]["rejected"] > 0):
            local_count = 0
            web_count = 0
            if not df_path.empty:
                source_counts = dict(zip(df_path["source"], df_path["count"]))
                local_count = source_counts.get("local", 0)
                web_count = source_counts.get("web", 0)
            rejected_count = (
                df_rejected.iloc[0]["rejected"] if not df_rejected.empty else 0
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Hybrid only (local)", local_count)
            col2.metric("Local + web", web_count)
            col3.metric("Rejected", rejected_count)
            st.caption("source='local' means the local hybrid search alone answered; source='web' means the agent used local + Bulbapedia.")
            path_data = {"Hybrid only": local_count, "Local + web": web_count}
            if rejected_count:
                path_data["Rejected"] = rejected_count
            if path_data:
                st.bar_chart(pd.DataFrame(list(path_data.items()), columns=["path", "count"]).set_index("path")["count"])
        else:
            st.info("No conversations yet.")

    def raw_trace_data(self):
        st.subheader("📋 Raw Trace Data")
        df_all = self.load_dataframe("SELECT * FROM spans")
        with st.expander("Show all traces"):
            st.dataframe(df_all, use_container_width=True, height=300)

    def refresh_sidebar(self):
        # Manual refresh from the sidebar (st.auto_refresh is not a Streamlit API)
        with st.sidebar:
            st.caption("Data is loaded at script run time.")
            if st.button("🔄 Refresh Data"):
                st.rerun()
