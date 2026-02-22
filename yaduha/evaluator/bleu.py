from yaduha.evaluator import Evaluator


class BleuEvaluator(Evaluator):
    """BLEU score evaluator using sacrebleu."""

    name: str = "bleu"

    def evaluate(self, source: str, target: str) -> float:
        import sacrebleu

        score = sacrebleu.sentence_bleu(hypothesis=target, references=[source])
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
