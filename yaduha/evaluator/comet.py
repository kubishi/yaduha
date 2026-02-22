from typing import Any

from yaduha.evaluator import Evaluator


class CometEvaluator(Evaluator):
    """COMET evaluator for translation quality estimation.

    Uses the reference-based COMET model by default. For round-trip evaluation
    (source vs. back-translation), the source is used as both src and ref.
    """

    name: str = "comet"
    model_path: str = "Unbabel/wmt22-comet-da"
    _model: Any | None = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_model(self) -> Any:
        if self._model is None:
            from comet import download_model, load_from_checkpoint  # type: ignore[import-untyped]

            path = download_model(self.model_path)
            self._model = load_from_checkpoint(path)
        return self._model

    def evaluate(self, source: str, target: str) -> float:
        model = self._get_model()
        data = [{"src": source, "mt": target, "ref": source}]
        output = model.predict(data, batch_size=1, gpus=0)
        result = output.scores[0]
        self.logger.log(
            data={
                "event": "evaluation",
                "evaluator": self.name,
                "source": source,
                "target": target,
                "score": result,
            }
        )
        return result
