from dataclasses import dataclass

from src.rag.scoring import AgentResult


@dataclass
class ChatMessage:
    role: str
    content: str
    msg_id: str = ""
    agent_result: AgentResult | None = None
    span_id: str | None = None
    conversation_id: int | None = None
    question: str = ""

    @classmethod
    def user(cls, content):
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content, msg_id="", agent_result=None, span_id=None, question=""
    ):
        return cls(
            role="assistant",
            content=content,
            msg_id=msg_id,
            agent_result=agent_result,
            span_id=span_id,
            question=question,
        )
