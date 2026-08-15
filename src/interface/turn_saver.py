import logging

from src.llm import LLMClient


class TurnSaver:
    # All persistence lives here; every write is guarded so a failure never
    # breaks the chat.

    def save_turn(self, agent, result, span_id, question, session_id):
        try:
            from monitoring.db_save import (
                save_conversation,
                save_llm_call,
                save_search,
            )

            record = agent.turn_record
            conversation_id = save_conversation(
                record, question, "llm-zoomcamp", session_id=session_id,
            )
            if conversation_id:
                for search in result.get("searches", []):
                    save_search(
                        conversation_id, span_id,
                        search.query, search.search_query,
                        search.source, search.payload,
                    )
                for call in agent.calls:
                    save_llm_call(
                        conversation_id, span_id,
                        call.model, call.prompt_tokens,
                        call.completion_tokens, call.total_tokens,
                        call.response_time, call.error,
                    )
            return conversation_id
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to save conversation", exc_info=True
            )
            return None

    def save_error(self, question, error_msg, response_time, session_id):
        try:
            from monitoring.db_save import save_conversation
            from src.rag.metrics import LLMCallRecord

            record = LLMCallRecord.failed(
                model=LLMClient.get_model(),
                answer=error_msg,
                error=error_msg,
                response_time=response_time,
            )
            save_conversation(
                record, question, "llm-zoomcamp", session_id=session_id,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to save error conversation", exc_info=True
            )

    def save_feedback_status(self, span_id, conversation_id, score):
        label = "positive" if score > 0 else "negative"
        if span_id:
            try:
                from monitoring.span_store import record_feedback

                record_feedback(span_id, label)
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to record feedback", exc_info=True
                )
        if conversation_id:
            try:
                from monitoring.db_feedback import save_feedback

                save_feedback(conversation_id, "user", score=score)
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to save feedback", exc_info=True
                )
