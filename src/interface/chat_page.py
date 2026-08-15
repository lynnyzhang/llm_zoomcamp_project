import logging
import time

import streamlit as st

from src.llm import LLMClient
from src.rag.agent import RAGAgent
from src.search.hybrid import HybridSearch


class ChatPage:
    def __init__(self, cards, messages, saver):
        self.cards = cards
        self.messages = messages
        self.saver = saver

    def make_agent(self):
        search_index = HybridSearch()
        return RAGAgent(search_index=search_index, llm_client=LLMClient.get())

    def maybe_trace(self, agent):
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

    def get_agent(self):
        if st.session_state.agent is None:
            with st.spinner("Loading search index and LLM..."):
                st.session_state.agent = self.maybe_trace(self.make_agent())
        return st.session_state.agent

    def load_history(self):
        # Restore the last conversations from Postgres so a browser refresh
        # keeps the chat (the store is the source of truth, session state is not).
        if st.session_state.messages:
            return
        try:
            from monitoring.db_query import get_conversations, get_feedback_for_conversations
            records = [
                r for r in reversed(get_conversations(
                    limit=20, session_id=st.session_state.session_id
                )) if r.error is None
            ]
            ids = [r.id for r in records if r.id]
            feedback = get_feedback_for_conversations(ids) if ids else {}
            for record in records:
                msg_id = f"hist_{record.id}"
                question = getattr(record, "question", None) or record.prompt or ""
                st.session_state.messages.append({"role": "user", "content": question})
                message = {
                    "role": "assistant",
                    "content": record.answer,
                    "msg_id": msg_id,
                    "span_id": record.span_id,
                    "conversation_id": record.id,
                }
                if record.rejected:
                    message["agent_result"] = {"rejected": True}
                st.session_state.messages.append(message)
                score = feedback.get(record.id)
                if score:
                    st.session_state.feedback[msg_id] = (
                        "positive" if score > 0 else "negative"
                    )
        except Exception:
            logging.getLogger(__name__).warning("Failed to load chat history", exc_info=True)

    def render(self, message):
        self.messages.render_message(message)

    def run_turn(self, prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    start = time.perf_counter()
                    agent = self.get_agent()
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
                    message["conversation_id"] = self.saver.save_turn(
                        agent, result, span_id, prompt, st.session_state.session_id)
                except Exception as e:  # noqa: BLE001 — UI error boundary
                    error_msg = f"Error: {e!s}"
                    st.error(error_msg)
                    message = {"role": "assistant", "content": error_msg}
                    self.saver.save_error(prompt, error_msg,
                                          time.perf_counter() - start,
                                          st.session_state.session_id)
            # Render inside the SAME bubble — after spinner finishes
            if "agent_result" in message:
                self.messages.render_message_body(message)
                st.session_state.messages.append(message)
            elif message.get("content", "").startswith("Error:"):
                # Error was already shown via st.error() above; still record it.
                st.session_state.messages.append(message)
