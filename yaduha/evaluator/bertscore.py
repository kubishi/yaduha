from yaduha.evaluator import Evaluator


class BertScoreEvaluator(Evaluator):
    """BERTScore evaluator using contextual embeddings."""

    name: str = "bertscore"
    model_type: str = "microsoft/deberta-xlarge-mnli"

    def evaluate(self, source: str, target: str) -> float:
        from bert_score import score  # type: ignore[import-untyped]

        P, R, F1 = score(
            [target],
            [source],
            model_type=self.model_type,
            lang="en",
            verbose=False,
        )
        result = F1.item()
        self.logger.log(
            data={
                "event": "evaluation",
                "evaluator": self.name,
                "source": source,
                "target": target,
                "score": result,
                "precision": P.item(),
                "recall": R.item(),
            }
        )
        return result
