"""STEP 10 - RAGAS: quantitative evaluation of the RAG pipeline.

Four metrics, each answering a different failure mode:
  Context Precision : of the passages retrieved, how many were actually relevant?      (retrieval, ranking quality)
  Context Recall     : of the passages that WERE needed, how many did we retrieve?      (retrieval, coverage)
  Faithfulness       : is every claim in the answer supported by the retrieved context? (generation, hallucination)
  Answer Relevancy   : does the answer actually address the question asked?            (generation, on-topic-ness)
All are in [0, 1]; higher is better. RAGAS itself uses an LLM-as-judge internally, wired via `judge_llm()`
to our own gateway/embedder (Gemini or local Ollama) - never a hard OpenAI dependency.

COMPATIBILITY SHIM: every RAGAS release we tested (0.2-0.4) unconditionally imports
`langchain_community.chat_models.vertexai.ChatVertexAI` at package import time, but current
`langchain-community` no longer ships it (moved to the separate `langchain-google-vertexai`
package). We do not use Vertex AI, so we register a harmless stub module BEFORE importing ragas.
This is applied at module import time, once, and is a no-op if a real module is already present.
"""
from __future__ import annotations

import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401  (already installed - do nothing)
    except ModuleNotFoundError:
        _stub = types.ModuleType("langchain_community.chat_models.vertexai")

        class _ChatVertexAIStub:  # pragma: no cover - never instantiated, we don't use Vertex AI
            def __init__(self, *a, **k):
                raise RuntimeError("langchain_community.chat_models.vertexai is a compatibility stub "
                                  "(this project does not use Google Vertex AI).")

        _stub.ChatVertexAI = _ChatVertexAIStub
        sys.modules["langchain_community.chat_models.vertexai"] = _stub

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from genai_platform.config import ARTIFACTS, get_settings
from ragas import evaluate
from ragas.run_config import RunConfig

@dataclass
class EvalCase:
    question: str
    ground_truth: str = ""  # optional; only Context Recall needs a reference answer


DEFAULT_EVAL_SET: list[EvalCase] = [
    EvalCase("What is the refund policy for cancellations made within 24 hours?",
            "Cancellations within 24 hours of purchase are fully refunded."),
    EvalCase("How long is the standard product warranty?", "The standard warranty is 12 months from the purchase date."),
    EvalCase("What are the steps to reset a user's password?",
            "Go to the login page, click 'Forgot password', enter the email, then follow the reset link sent by email."),
    EvalCase("Which file formats does the ingestion pipeline support?", "PDF and HTML files (also plain text and markdown)."),
    EvalCase("What happens if a payment fails twice in a row?",
            "The account is flagged for manual review and the customer is notified by email."),
]


def judge_llm():
    """RAGAS's own judge model, wrapped for LangChain. Defaults to local Ollama; falls back to Gemini
    when LLM_PROVIDER=gemini, so a laptop with no GPU can still run a (slower, API-billed) evaluation."""
    from ragas.llms import LangchainLLMWrapper

    from genai_platform.llm.lc_model import GatewayChatModel

    return LangchainLLMWrapper(GatewayChatModel())


def judge_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_core.embeddings import Embeddings

    from genai_platform.embeddings.embedder import get_embedder

    emb = get_embedder()

    class _Adapter(Embeddings):  # our Embedder -> LangChain's Embeddings interface
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return emb.embed_documents(texts).tolist()

        def embed_query(self, text: str) -> list[float]:
            return emb.embed_query(text).tolist()

    return LangchainEmbeddingsWrapper(_Adapter())


def build_dataset(pipeline, cases: list[EvalCase], k: int = 5):
    """Run the live RAG pipeline over each case to collect (question, contexts, answer, ground_truth)."""
    from ragas import EvaluationDataset

    rows = []
    for c in cases:
        ans = pipeline.answer(c.question, k=k)
        rows.append({"user_input": c.question, "retrieved_contexts": [h.text for h in ans.hits],
                    "response": ans.answer, "reference": c.ground_truth or ans.answer})
    return EvaluationDataset.from_list(rows), rows


def run_ragas(pipeline, cases: Optional[list[EvalCase]] = None, k: int = 5, save: bool = True) -> dict:
    from ragas import evaluate
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, LLMContextRecall, ResponseRelevancy

    dataset, rows = build_dataset(pipeline, cases or DEFAULT_EVAL_SET, k)
    llm, emb = judge_llm(), judge_embeddings()
    metrics = [LLMContextPrecisionWithoutReference(llm=llm), LLMContextRecall(llm=llm),
               Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=emb)]
    result = evaluate(dataset=dataset, metrics=metrics, run_config=RunConfig(max_workers=2, timeout=300, max_retries=3))
    scores = result.to_pandas()
    summary = {m: round(float(scores[m].mean()), 4) for m in
              ["llm_context_precision_without_reference", "context_recall", "faithfulness", "answer_relevancy"]
              if m in scores.columns}
    out = {"summary": summary, "n_cases": len(rows), "per_case": scores.to_dict(orient="records")}
    if save:
        import json

        (ARTIFACTS / "ragas_report.json").write_text(json.dumps(out, indent=2, default=str))
    return out
