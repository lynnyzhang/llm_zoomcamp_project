import logging
import time

import streamlit as st

from src.interface.chat_message import ChatMessage
from src.llm_client import LLMClient
from src.rag.rag_agent import RAGAgent
from src.rag.scoring import AgentResult
from src.search.hybrid_search import HybridSearch


class ChatPage:
    def __init__(self, cards, messages, saver):
        self.cards = cards
        self.messages = messages
        self.saver = saver

    def make_agent(self) -> RAGAgent:
        search_index = HybridSearch()
        return RAGAgent(search_index=search_index, llm_client=LLMClient.get())

    def maybe_trace(self, agent: RAGAgent) -> RAGAgent:
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

    def get_agent(self) -> RAGAgent:
        if st.session_state.agent is None:
            with st.spinner("Loading search index and LLM..."):
                st.session_state.agent = self.maybe_trace(self.make_agent())
        return st.session_state.agent

    def render(self, message: ChatMessage):
        self.messages.render_message(message)

    def handle_prompt(self, prompt: str):
        st.session_state.messages.append(ChatMessage.user(prompt))
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
                    answer = result.answer
                    msg_id = f"msg_{len(st.session_state.messages)}"
                    message = ChatMessage.assistant(
                        answer, msg_id, result, span_id, prompt
                    )
                    message.conversation_id = self.saver.save_agent_loop(
                        agent, result, span_id, prompt, st.session_state.session_id
                    )
                except Exception as e:  # noqa: BLE001 — UI error boundary
                    error_msg = f"Error: {e!s}"
                    st.error(error_msg)
                    message = ChatMessage.assistant(content=error_msg)
                    self.saver.save_error(
                        prompt,
                        error_msg,
                        time.perf_counter() - start,
                        st.session_state.session_id,
                    )
            # Render inside the SAME bubble — after spinner finishes
            if message.agent_result is not None:
                self.messages.render_message_body(message)
                st.session_state.messages.append(message)
            elif message.content.startswith("Error:"):
                # Error was already shown via st.error() above; still record it.
                st.session_state.messages.append(message)
