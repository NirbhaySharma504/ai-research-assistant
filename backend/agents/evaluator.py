"""Evaluator node: RAGAS scoring (ragas 0.2 EvaluationDataset API).

Uses the Groq judge when available (reliable structured output); falls back to
Ollama. NaN scores are coerced to None so they never crash downstream / the UI.
"""

import math

from backend.graph.state import ResearchState
from backend.llm import get_judge_llm
from backend.tools.vector_store import _get_model  # reuse loaded MiniLM


def _clean(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 3)


def evaluator_node(state: ResearchState) -> dict:
    if not state.get("final_answer") or not state.get("retrieved_content"):
        return {"ragas_scores": None, "status": "done"}

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            faithfulness,
        )
        # Reference-free precision: we have no ground-truth answer, so the standard
        # context_precision (which needs a `reference` column) can't be used.
        from ragas.metrics import LLMContextPrecisionWithoutReference
        from langchain_core.embeddings import Embeddings

        context_precision = LLMContextPrecisionWithoutReference()

        # Evaluate against the chunks actually fed to the synthesizer (focused,
        # semantically-retrieved units) rather than whole scraped pages -- this is
        # what RAG context precision is meant to measure. Fall back to pages only if
        # the synthesizer didn't record its contexts.
        contexts = state.get("synthesis_contexts") or [
            c["content"] for c in state["retrieved_content"][:10]
        ]
        dataset = EvaluationDataset.from_list([
            {
                "user_input": state["query"],
                "response": state["final_answer"],
                "retrieved_contexts": contexts,
            }
        ])

        # wrap the local MiniLM model as a LangChain Embeddings for ragas
        class _MiniLM(Embeddings):
            def embed_documents(self, texts):
                return _get_model().encode(texts).tolist()

            def embed_query(self, text):
                return _get_model().encode([text]).tolist()[0]

        judge = LangchainLLMWrapper(get_judge_llm())
        embeddings = LangchainEmbeddingsWrapper(_MiniLM())

        # Serialize judge calls + back off on 429s. RAGAS otherwise fires many
        # concurrent calls per metric, which blows hosted free-tier rate limits
        # (Groq TPM) and silently NaNs every score.
        from ragas.run_config import RunConfig

        run_config = RunConfig(max_workers=1, max_retries=3, max_wait=20, timeout=120)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=judge,
            embeddings=embeddings,
            run_config=run_config,
        )
        df = result.to_pandas()

        # Metric column names vary by ragas version / metric class (e.g.
        # LLMContextPrecisionWithoutReference -> "llm_context_precision_..."),
        # so resolve each score by substring match instead of hardcoding.
        def _by(substr: str):
            for col in df.columns:
                if substr in col:
                    return _clean(df[col].iloc[0])
            return None

        scores = {
            "faithfulness": _by("faithfulness"),
            "answer_relevancy": _by("answer_relevancy"),
            "context_precision": _by("context_precision"),
        }
        return {"ragas_scores": scores, "status": "done"}
    except Exception as e:  # noqa: BLE001 - evaluation must never fail the run
        return {
            "ragas_scores": None,
            "errors": state.get("errors", []) + [f"evaluator: {e}"],
            "status": "done",
        }
