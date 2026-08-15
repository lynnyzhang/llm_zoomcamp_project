JUDGE_PROMPTS = {
    "simple": {
        "instructions": "You are an AI judge. Rate the answer on a 1-5 scale.",
        "template": """\
Question: {question}
Context: {context}
Answer: {answer}
Ground Truth: {ground_truth}

Rate faithfulness (context support), relevance (addresses question), coherence (clarity). 1-5 each.
Respond with a single JSON object in exactly this format (no markdown, no extra text):
{{"faithfulness": 1-5, "relevance": 1-5, "coherence": 1-5, "explanation": "brief reasoning"}}""",
    },
    "detailed": {
        "instructions": """\
You are an expert evaluator for question-answering systems.
Rate the generated answer on three dimensions using a 1-5 scale:
- Faithfulness (1-5): How well is the answer supported by the provided context? 5 = fully grounded, 1 = contradicts context.
- Relevance (1-5): Does the answer actually address the question asked? 5 = directly answers, 1 = off-topic.
- Coherence (1-5): Is the answer well-structured and clear? 5 = excellent, 1 = incoherent.
Provide a brief explanation for your ratings.""",
        "template": """\
Question: {question}

Retrieved Context:
{context}

Generated Answer: {answer}

Ground Truth Answer: {ground_truth}

Evaluate the generated answer against the context and ground truth.
Provide scores (1-5) for faithfulness, relevance, and coherence.
Respond with a single JSON object in exactly this format (no markdown, no extra text):
{{"faithfulness": 1-5, "relevance": 1-5, "coherence": 1-5, "explanation": "brief reasoning"}}""",
    },
    "with_examples": {
        "instructions": """\
You are an expert evaluator for question-answering systems.

Examples of ratings:

Example 1:
Q: "What type is Pikachu?"
Context: "Pikachu is an electric-type Pokémon."
Answer: "Pikachu is an electric type."
Ground Truth: "Pikachu is an electric type."
Rating: faithfulness=5, relevance=5, coherence=5

Example 2:
Q: "What type is Pikachu?"
Context: "Pikachu is an electric-type Pokémon."
Answer: "Pikachu is a grass type."
Ground Truth: "Pikachu is an electric type."
Rating: faithfulness=1, relevance=2, coherence=4

Now evaluate the following:""",
        "template": """\
Question: {question}

Retrieved Context:
{context}

Generated Answer: {answer}

Ground Truth Answer: {ground_truth}

Rate faithfulness (1-5), relevance (1-5), coherence (1-5). Explain briefly.
Respond with a single JSON object in exactly this format (no markdown, no extra text):
{{"faithfulness": 1-5, "relevance": 1-5, "coherence": 1-5, "explanation": "brief reasoning"}}""",
    },
}
