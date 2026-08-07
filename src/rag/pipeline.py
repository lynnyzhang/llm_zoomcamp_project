"""RAG pipeline base class.

Adapted from 4-Evaluation/rag_helper.py RAGBase pattern, modified to use
HybridSearch (keyword + vector) instead of minsearch, and to work with
documents that have 'content'/'title'/'section' fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai import OpenAI

from src.llm import create_client, get_model

if TYPE_CHECKING:
    from src.search.hybrid import HybridSearch


INSTRUCTIONS = """\
You are a helpful assistant that answers questions based on provided context.
Use ONLY the information in the context to answer. If the answer is not found
in the context, respond with "I don't know."
"""

PROMPT_TEMPLATE = """\
QUESTION: {question}

CONTEXT:
{context}
"""


class RAGBase:
    """Base RAG pipeline: search → build context → prompt → LLM → answer."""

    def __init__(
        self,
        search_index: HybridSearch,
        llm_client: OpenAI | None = None,
        instructions: str = INSTRUCTIONS,
        prompt_template: str = PROMPT_TEMPLATE,
        model: str | None = None,
    ):
        self.search_index = search_index
        self.llm_client = llm_client or create_client()
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model or get_model()

    def search(self, query: str, num_results: int = 5) -> list[dict]:
        """Run hybrid search and return top results."""
        return self.search_index.search(query, num_results=num_results)

    def build_context(self, search_results: list[dict]) -> str:
        """Format search results into a text context block."""
        lines: list[str] = []
        for doc in search_results:
            section = doc.get("section", "")
            title = doc.get("title", "")
            content = doc.get("content", "")
            if section:
                lines.append(f"SECTION: {section}")
            if title:
                lines.append(f"TITLE: {title}")
            lines.append(f"CONTENT: {content}")
            lines.append("")
        return "\n".join(lines).strip()

    def build_prompt(self, query: str, search_results: list[dict]) -> str:
        """Combine query and context into the final prompt."""
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt: str) -> str:
        """Call the LLM via OpenAI Responses API and return the answer text."""
        messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        response = self.llm_client.responses.create(
            model=self.model,
            input=messages,
        )
        return response.output_text

    def rag(self, query: str) -> str:
        """Full RAG pipeline: search → prompt → LLM answer."""
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        return self.llm(prompt)
