import streamlit as st


class MessageRenderer:
    def __init__(self, cards, saver, show_confidence=False):
        self.cards = cards
        self.saver = saver
        self.show_confidence = show_confidence

    def render_message_body(self, msg):
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

            docs = self.cards.pokemon_doc(searches, msg.get("question", ""))
            if docs:
                self.cards.pokemon_card_grid(docs)

            if self.show_confidence:
                confidence = result.get("confidence")
                if confidence is not None:
                    st.progress(confidence, text=f"Confidence: {confidence:.0%}")

        # Feedback buttons
        st.divider()
        col1, col2, _ = st.columns([1, 1, 8])
        with col1:
            if st.button("👍", key=f"up_{msg_id}"):
                st.session_state.feedback[msg_id] = "positive"
                self.saver.save_feedback_status(
                    msg.get("span_id"), msg.get("conversation_id"), 1,
                )
                st.toast("Thanks for the feedback!")
        with col2:
            if st.button("👎", key=f"down_{msg_id}"):
                st.session_state.feedback[msg_id] = "negative"
                self.saver.save_feedback_status(
                    msg.get("span_id"), msg.get("conversation_id"), -1,
                )
                st.toast("Thanks for the feedback!")

        # Show feedback status
        if msg_id in st.session_state.feedback:
            feedback = st.session_state.feedback[msg_id]
            if feedback == "positive":
                st.success("👍 Positive feedback recorded")
            else:
                st.warning("👎 Negative feedback recorded")

    def render_message(self, msg):
        with st.chat_message(msg["role"]):
            self.render_message_body(msg)
