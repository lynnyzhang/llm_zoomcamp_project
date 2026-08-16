import logging

from src.llm_client import LLMClient
from src.rag.rag_agent import RAGAgent
from src.rag.scoring import AgentResult


class AgentLoopSaver:
    # All persistence lives here; every write is guarded so a failure never
    # breaks the chat.

    def save_agent_loop(
        self,
        agent: RAGAgent,
        result: AgentResult,
        span_id: str | None,
        question: str,
        session_id: str,
    ) -> int | None:
        record = agent.agent_loop_record
        if record is None:
            return None
        try:
            from monitoring.db_save import save_conversation, save_llm_call, save_search

            conversation_id = save_conversation(
                record,
                question,
                "llm-zoomcamp",
                session_id=session_id,
            )
            if conversation_id:
                for search in result.searches:
                    save_search(
                        conversation_id,
                        span_id,
                        search.query,
                        search.search_query,
                        search.source,
                        search.payload,
                    )
                for call in agent.calls:
                    save_llm_call(
                        conversation_id,
                        span_id,
                        call.model,
                        call.prompt_tokens,
                        call.completion_tokens,
                        call.total_tokens,
                        call.response_time,
                        call.error,
                    )
            return conversation_id
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to save conversation", exc_info=True
            )
            return None

    def save_error(
        self, question: str, error_msg: str, response_time: float, session_id: str
    ):
        try:
            from monitoring.db_save import save_conversation
            from src.rag.llm_call_record import LLMCallRecord

            record = LLMCallRecord.failed(
                model=LLMClient.get_model(),
                answer=error_msg,
                error=error_msg,
                response_time=response_time,
            )
            save_conversation(
                record,
                question,
                "llm-zoomcamp",
                session_id=session_id,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to save error conversation", exc_info=True
            )

    def save_feedback_status(
        self, span_id: str | None, conversation_id: int | None, score: int
    ):
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
