from abc import ABC

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal
import pathlib
from openai import OpenAI

from yaduha.agent import Agent
from yaduha.translator import Translator, Translation
from yaduha.logger import Logger

    

class Evaluator(BaseModel, ABC):
    name: str
    logger: Logger = Field(default_factory=Logger)

    def evaluate(self, source: str, target: str) -> float:
        raise NotImplementedError("Subclasses must implement the evaluate method.")

class OpenAIEvaluator(Evaluator):
    name: str = "openai_embedding_evaluator"
    model: Literal["text-embedding-3-small", "text-embedding-3-large"] = "text-embedding-3-small"

    def evaluate(self, source: str, target: str) -> float:
        client = OpenAI()
        response = client.embeddings.create(
            model=self.model,
            input=[source, target]
        )
        source_embedding = response.data[0].embedding
        target_embedding = response.data[1].embedding

        # Compute cosine similarity
        dot_product = sum(s * t for s, t in zip(source_embedding, target_embedding))
        source_norm = sum(s ** 2 for s in source_embedding) ** 0.5
        target_norm = sum(t ** 2 for t in target_embedding) ** 0.5
        similarity = dot_product / (source_norm * target_norm)
        self.logger.log(data={
            "source": source,
            "target": target,
            "similarity_score": similarity
        })
        return similarity

