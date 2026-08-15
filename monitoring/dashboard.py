import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from monitoring.dashboard_utils import load_dataframe, postgres_reachable
from monitoring.history_panels import HistoryCharts
from monitoring.span_panels import SpanCharts
from monitoring.trace_detail_panels import TraceDetailCharts

st.set_page_config(
    page_title="Pokémon Monitoring",
    page_icon="📊",
    layout="wide",
)


class MonitoringDashboard:
    def __init__(self):
        self.span_charts = SpanCharts(load_dataframe)
        self.trace_detail_charts = TraceDetailCharts(load_dataframe)
        self.history_charts = HistoryCharts(load_dataframe)

    def run(self):
        st.title("📊 Pokémon Monitoring Dashboard")
        st.caption("Analytics from OpenTelemetry traces stored in Postgres")

        self.span_charts.summary_metrics()
        st.divider()
        self.span_charts.queries_over_time()
        st.divider()
        self.span_charts.feedback_distribution()
        st.divider()
        self.trace_detail_charts.latency_distribution()
        st.divider()
        self.trace_detail_charts.token_usage()
        st.divider()
        self.trace_detail_charts.popular_topics()
        st.divider()
        self.trace_detail_charts.agent_search_patterns()
        st.divider()
        self.history_charts.recent_conversations()
        st.divider()
        self.history_charts.gated_query_rate()
        st.divider()
        self.history_charts.answer_path_mix()
        st.divider()
        self.history_charts.raw_trace_data()
        self.history_charts.refresh_sidebar()


if not postgres_reachable():
    st.warning("Postgres is unreachable — start it (`docker-compose up postgres` or a local server) and retry.")
    st.stop()

MonitoringDashboard().run()
