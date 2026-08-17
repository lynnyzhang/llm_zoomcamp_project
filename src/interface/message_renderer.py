import streamlit as st

from src.interface.agent_loop_saver import AgentLoopSaver
from src.interface.card_renderer import CardRenderer
from src.interface.chat_message import ChatMessage


class MessageRenderer:
    def __init__(
        self, cards: CardRenderer, saver: AgentLoopSaver, show_confidence: bool = False
    ):
        self.cards = cards
        self.saver = saver
        self.show_confidence = show_confidence

    def render_message_body(self, msg: ChatMessage):
        """Render the *inner* content of an assistant message (no outer bubble)."""
        if msg.role != "assistant" or msg.agent_result is None:
            st.markdown(msg.content)
            return

        result = msg.agent_result
        answer = result.answer
        searches = result.searches
        msg_id = msg.msg_id

        if result.rejected:
            st.warning(answer)
        else:
            st.markdown(answer)

            source = result.source
            if source == "local":
                st.caption("Source: Local knowledge base")
            elif source == "web":
                st.caption("Source: Bulbapedia (web)")
            elif source == "local+web":
                st.caption("Source: Local knowledge base + Bulbapedia (web)")

            docs = self.cards.pokemon_doc(searches, msg.question)
            if docs:
                self.cards.pokemon_card_grid(docs)

            if self.show_confidence:
                confidence = result.confidence
                if confidence is not None:
                    label = f"Confidence: {confidence:.0%}"
                    relevance = result.relevance
                    if relevance is not None:
                        label += f" · Relevance: {relevance:.0%}"
                    st.progress(confidence, text=label)

        # Feedback buttons
        st.divider()
        col1, col2, _ = st.columns([1, 1, 8])
        with col1:
            if st.button("👍", key=f"up_{msg_id}"):
                st.session_state.feedback[msg_id] = "positive"
                self.saver.save_feedback_status(
                    msg.span_id,
                    msg.conversation_id,
                    1,
                )
                st.toast("Thanks for the feedback!")
        with col2:
            if st.button("👎", key=f"down_{msg_id}"):
                st.session_state.feedback[msg_id] = "negative"
                self.saver.save_feedback_status(
                    msg.span_id,
                    msg.conversation_id,
                    -1,
                )
                st.toast("Thanks for the feedback!")

        # Show feedback status
        if msg_id in st.session_state.feedback:
            feedback = st.session_state.feedback[msg_id]
            if feedback == "positive":
                st.success("👍 Positive feedback recorded")
            else:
                st.warning("👎 Negative feedback recorded")

    def render_message(self, msg: ChatMessage):
        with st.chat_message(msg.role):
            self.render_message_body(msg)
