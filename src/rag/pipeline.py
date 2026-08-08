# Adapted from 4-Evaluation/rag_helper.py RAGBase pattern; modified to use
# HybridSearch (keyword + vector) instead of minsearch, and to work with
# documents that have 'content'/'title'/'section' fields.

from src.llm import create_client, get_model

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

    def __init__(
        self,
        search_index,
        llm_client=None,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model=None,
        search_type="hybrid",
    ):
        self.search_index = search_index
        self.llm_client = llm_client or create_client()
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model or get_model()
        self.search_type = search_type

    def search(self, query, num_results=5):
        if self.search_type == "keyword":
            return self.search_index.keyword_search(query, num_results=num_results)
        if self.search_type == "vector":
            return self.search_index.vector_search(query, num_results=num_results)
        return self.search_index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []
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

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt):
        messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        response = self.llm_client.responses.create(
            model=self.model,
            input=messages,
        )
        return response.output_text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        return self.llm(prompt)
