import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from src.llm_client import LLMClient
from src.rag.rag_base import RAGBase
from src.rag.rag_agent import RAGAgent
from src.search.hybrid_search import HybridSearch


def setup():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(PROJECT_ROOT / ".env")

    from evaluation.document_index import load_document_index
    from evaluation.retrieval_metrics import load_qa_pairs

    qa_path = PROJECT_ROOT / "evaluation" / "data" / "qa.jsonl"
    results_dir = PROJECT_ROOT / "evaluation" / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loading Q&A pairs...")
    qa_pairs = load_qa_pairs(str(qa_path))
    print(f"Loaded {len(qa_pairs)} pairs")

    print("Loading document index (ground-truth answers)...")
    doc_idx = load_document_index()
    print(f"Loaded {len(doc_idx)} documents")

    base_url = LLMClient.get_base_url()
    llm_available = False
    client = None
    try:
        urllib.request.urlopen(base_url + "/models", timeout=3)
        client = LLMClient.get()
        llm_available = True
        print(f"LLM API available at {base_url}")
    except Exception:
        print(f"LLM API not available at {base_url} — running retrieval-only evaluation")

    print("Initializing search index...")
    t0 = time.time()
    search_index = HybridSearch()
    print(f"Search index ready in {time.time() - t0:.1f}s")

    rag_base = RAGBase(search_index=search_index, llm_client=client)
    agent = RAGAgent(search_index=search_index, llm_client=client)

    return qa_pairs, doc_idx, client, llm_available, rag_base, agent
