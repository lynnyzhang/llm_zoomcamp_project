# RAGBase: search → build context → prompt → LLM answer, using HybridSearch
# (keyword + vector) over documents with 'search_text' fields.

from src.llm import LLMClient

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
    ):
        self.search_index = search_index
        self.llm_client = llm_client or LLMClient.get()
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model or LLMClient.get_model()

    def search(self, query, num_results=5):
        return self.search_index.search(query, num_results=num_results)

    def build_context(self, search_results):
        blocks = []
        for doc in search_results:
            blocks.append(doc.get("search_text", ""))
        return "\n\n".join(b for b in blocks if b).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def call_llm(self, messages, tools=None, temperature=None):
        # The single LLM request site shared by the plain rag() path and the
        # agent loop (which wraps this with per-call recording).
        return self.llm_client.client.responses.create(
            model=self.model,
            input=messages,
            tools=tools,
            temperature=(
                temperature if temperature is not None
                else LLMClient.get_answer_temperature()
            ),
        )

    def llm(self, prompt):
        messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        response = self.call_llm(messages)
        return response.output_text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        return self.llm(prompt)
