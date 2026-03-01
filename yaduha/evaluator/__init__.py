from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from yaduha.logger import Logger

if TYPE_CHECKING:
    from yaduha.translator import Translation


class Evaluator(BaseModel, ABC):
    name: str
    logger: Logger = Field(default_factory=Logger)

    def evaluate(self, source: str, target: str) -> float:
        raise NotImplementedError("Subclasses must implement the evaluate method.")


class OpenAIEvaluator(Evaluator):
    name: str = "openai_embedding"
    model: Literal["text-embedding-3-small", "text-embedding-3-large"] = "text-embedding-3-small"
    api_key: str | None = None

    def evaluate(self, source: str, target: str) -> float:
        from openai import OpenAI

        if not source or not target:
            self.logger.log(
                data={"source": source, "target": target, "similarity_score": 0.0, "skipped": True}
            )
            return 0.0
        client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
        response = client.embeddings.create(model=self.model, input=[source, target])
        source_embedding = response.data[0].embedding
        target_embedding = response.data[1].embedding

        # Compute cosine similarity
        dot_product = sum(s * t for s, t in zip(source_embedding, target_embedding))
        source_norm = sum(s**2 for s in source_embedding) ** 0.5
        target_norm = sum(t**2 for t in target_embedding) ** 0.5
        similarity = dot_product / (source_norm * target_norm)
        self.logger.log(data={"source": source, "target": target, "similarity_score": similarity})
        return similarity


def batch_evaluate(translations: list[Translation], evaluator: Evaluator) -> list[Translation]:
    """Evaluate translations and return copies with updated evaluations dict.

    Only evaluates translations that have a back_translation. Scores are computed
    between the source text and the back-translated text.

    Args:
        translations: List of Translation objects to evaluate.
        evaluator: Evaluator to compute scores with.

    Returns:
        New list of Translation objects with evaluator scores added to evaluations.
    """
    from yaduha.translator import Translation as Translation  # noqa: F811

    results = []
    for t in translations:
        if t.back_translation is not None:
            score = evaluator.evaluate(t.source, t.back_translation.source)
            updated_evals = {**t.evaluations, evaluator.name: score}
            results.append(t.model_copy(update={"evaluations": updated_evals}))
        else:
            results.append(t)
    return results


try:
    from yaduha.evaluator.bleu import BleuEvaluator as BleuEvaluator
    from yaduha.evaluator.chrf import ChrfEvaluator as ChrfEvaluator
except ImportError:
    pass

try:
    from yaduha.evaluator.bertscore import BertScoreEvaluator as BertScoreEvaluator
except ImportError:
    pass

try:
    from yaduha.evaluator.comet import CometEvaluator as CometEvaluator
except ImportError:
    pass
