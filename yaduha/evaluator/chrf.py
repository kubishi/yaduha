from yaduha.evaluator import Evaluator


class ChrfEvaluator(Evaluator):
    """Character n-gram F-score evaluator using sacrebleu."""

    name: str = "chrf"

    def evaluate(self, source: str, target: str) -> float:
        import sacrebleu

        score = sacrebleu.sentence_chrf(hypothesis=target, references=[source])
        normalized = score.score / 100.0
        self.logger.log(
            data={
                "event": "evaluation",
                "evaluator": self.name,
                "source": source,
                "target": target,
                "score": normalized,
            }
        )
        return normalized
