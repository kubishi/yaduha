import pathlib
import json
import pandas as pd
import pytest

from kubishi_sentences import segment, semantic_models

thisdir = pathlib.Path(__file__).parent.parent.absolute()


@pytest.fixture
def example_sentences():
    # Load example sentences from a fixture or test file
    return json.loads((thisdir / "data" / "semantic_sentences.json").read_text())

@pytest.fixture
def similarity_funcs():
    return {
        "spacy": semantic_models.semantic_similarity_spacy,
        # "bert": semantic_similarity_bert,
        # "all-MiniLM-L6-v2": functools.partial(semantic_similarity_sentence_transformers, model="all-MiniLM-L6-v2"),
        # "paraphrase-MiniLM-L6-v2": functools.partial(
        #     semantic_similarity_sentence_transformers, model="paraphrase-MiniLM-L6-v2"
        # ),
        # # "SFR-Embedding-Mistral": functools.partial(semantic_similarity_sentence_transformers, model='Salesforce/SFR-Embedding-Mistral'),
        # "text-embedding-3-large": functools.partial(semantic_similarity_openai, model="text-embedding-3-large"),
        # "text-embedding-3-small": functools.partial(semantic_similarity_openai, model="text-embedding-3-small"),
        # "text-embedding-ada-002": functools.partial(semantic_similarity_openai, model="text-embedding-ada-002"),
    }

def test_similarity_metrics(example_sentences, similarity_funcs):
    
    rows = segment.calculate_similarity_metrics(example_sentences[:1], similarity_funcs)
    df = pd.DataFrame(rows)

    # Check if DataFrame has been created
    assert not df.empty

    # Check statistical calculations
    stats = df.groupby("similarity_func").agg({"avg_displacement": ["mean", "std"], "rbo": ["mean", "std"]})
    assert not stats.empty

    # Ensure stats DataFrame is not None
    assert stats is not None

# def main() -> None:  # pylint: disable=missing-function-docstring
#     """
#     Driver function showing how to run segment.py
#     Workflow:
#         1. Define a set of english sentences
#         1. For a single sentence, split it into a schema defined in object `SENTENCE_SCHEMA`
#         1. Create a simple sentence using the schema from previous step
#         1. Compute semantic similarity between original sentence and simple sentence created in previous step
#     """
#     source_sentences = [
#         "The dog fell.",
#         "The dog fell yesterday.",
#         "The dog was running yesterday and fell.",
#         "The dog was running yesterday and fell while chasing a cat.",
#         "The dog sat in the house.",
#         "I gave him bread.",
#         "The dog and the cat were running.",
#     ]

#     for source_sentence in source_sentences:
#         simple_sentences = segment.split_sentence(source_sentence, model=os.environ["OPENAI_MODEL"])
#         simple_nl_sentence = ". ".join([segment.make_sentence(sentence) for sentence in simple_sentences]) + "."

#         similarity = semantic_models.semantic_similarity_spacy(source_sentence, simple_nl_sentence)
#         print(f"Source sentence: {source_sentence}")
#         print(f"Simple sentences: {simple_nl_sentence}")

#         print(f"Semantic similarity: {similarity:0.3f}")
#         print()