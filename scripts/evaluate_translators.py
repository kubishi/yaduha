import json
from typing import Dict, List, Set, Tuple, Type
from yaduha.base import Translation, Translator
from yaduha.forward import (
    PipelineTranslation, PipelineTranslator,
    InstructionsTranslator, AgenticTranslator
)
import pandas as pd
import pathlib
from itertools import product

from pydantic import BaseModel
import dotenv

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.resolve()

class Result(BaseModel):
    translator: str
    model: str
    sentence_type: str
    translation: Translation

class Results(BaseModel):
    results: List[Result]

def main():
    models = ["gpt-4o-mini", "gpt-4o"]
    sentences = pd.read_csv(thisdir / 'data/evaluation_sentences.csv')
    translators: Dict[str, Type[Translator]] = {
        'pipeline': PipelineTranslator,
        'instructions': InstructionsTranslator,
        'agentic': AgenticTranslator
    }

    # load results from disk
    resultspath = thisdir / 'results/evaluation_results.json'
    resultspath.parent.mkdir(exist_ok=True, parents=True)
    results = Results(results=[])
    if resultspath.exists():
        results = Results.model_validate_json(resultspath.read_text())
    finished = {
        (r.translator, r.model, r.sentence_type)
        for r in results.results
    }
    combos = list(product(models, sentences.itertuples(), translators.items()))
    for model, (_, sentence, sentence_type), (translator_name, TranslatorClass) in combos:
        if (translator_name, model, sentence_type) in finished:
            continue
        translator = TranslatorClass(model=model)
        translation = translator.translate(sentence)
        result = Result(
            translator=translator_name,
            model=model,
            sentence_type=sentence_type,
            translation=translation
        )
        results.results.append(result)
        finished.add((translator_name, model, sentence_type))
        resultspath.write_text(results.model_dump_json())




if __name__ == '__main__':
    main()