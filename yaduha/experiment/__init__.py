from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pathlib

from yaduha.agent import Agent
from yaduha.translator import Translator, Translation
from yaduha.logger import Logger



class InputSentence(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TranslationEvaluation(BaseModel):
    sentence: InputSentence
    translation: Translation
    score: float = Field(..., description="The evaluation score for the translation.")

class SemanticSimilarityEvaluator(BaseModel):
    name: str
    model: str

    def evaluate(self, translation: Translation) -> TranslationEvaluation:
        # Placeholder for semantic similarity evaluation logic
        source = translation.source
        if not translation.back_translation:
            raise ValueError("Back translation is required for semantic similarity evaluation.")
        target = translation.back_translation.target

        # TODO: Implement the actual semantic similarity evaluation using the specified model.
        semantic_similarity_score = 0.0

        evaluation = TranslationEvaluation(
            sentence=InputSentence(text=source, metadata={"category": "semantic_similarity"}),
            translation=translation,
            score=semantic_similarity_score,
        )
        return evaluation

class Experiment(BaseModel):
    name: str
    agents: List[Agent]
    translators: List[Translator]
    logger: Logger
    sentences: List[InputSentence]

    savedir: pathlib.Path

    def run(self, overwrite: bool = False):
        if self.savedir.exists() and not overwrite:
            raise FileExistsError(f"Directory {self.savedir} already exists. Use overwrite=True to overwrite it.")
        self.savedir.mkdir(parents=True, exist_ok=overwrite)

        # TODO: Implement the logic to run the experiment, including translating sentences and evaluating translations.

