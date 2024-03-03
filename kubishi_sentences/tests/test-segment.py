import json
import functools
import numpy as np
import pandas as pd
import pytest
from kubishi_sentences.segment import (
    semantic_similarity_spacy,
    semantic_similarity_bert,
    semantic_similarity_sentence_transformers,
    semantic_similarity_openai,
)
from ranking_metrics import RankingSimilarity


def load_sentences(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


def calculate_similarity_metrics(sentences, similarity_funcs):
    rows = []
    for sentence in sentences:
        base_sentence = sentence["base"]
        sentence_list = sentence["sentences"]
        for similarity_func_name, similarity_func in similarity_funcs.items():
            similarities = np.array([similarity_func(base_sentence, s) for s in sentence_list])
            dist = np.mean(np.abs(np.argsort(-similarities) - np.arange(len(similarities))))

            sorted_sentences = [sentence_list[i] for i in np.argsort(-similarities)]
            rbo_similarity = RankingSimilarity(sorted_sentences, sentence_list).rbo()

            rows.append([base_sentence, similarity_func_name, dist, rbo_similarity])

    return rows


def create_dataframe(rows):
    return pd.DataFrame(rows, columns=["sentence", "similarity_func", "avg_displacement", "rbo"])


def compute_stats(df):
    stats = df.groupby("similarity_func").agg({"avg_displacement": ["mean", "std"], "rbo": ["mean", "std"]})
    return stats.sort_values(by=("rbo", "mean"), ascending=False)


def test_load_sentences():
    sentences = load_sentences("data/semantic_sentences.json")
    assert isinstance(sentences, list)
    assert len(sentences) > 0


def test_calculate_similarity_metrics():
    sentences = [
        {"base": "sentence1", "sentences": ["sentence2", "sentence3"]},
        {"base": "sentence2", "sentences": ["sentence3", "sentence4"]},
    ]
    similarity_funcs = {"test": lambda x, y: 0.5}  # Dummy similarity function
    rows = calculate_similarity_metrics(sentences, similarity_funcs)
    assert len(rows) == 4


def test_create_dataframe():
    rows = [["sentence1", "test", 0.5, 0.7], ["sentence2", "test", 0.4, 0.6]]
    df = create_dataframe(rows)
    assert len(df) == 2
    assert set(df.columns) == {"sentence", "similarity_func", "avg_displacement", "rbo"}


def test_compute_stats():
    rows = [["sentence1", "test", 0.5, 0.7], ["sentence2", "test", 0.4, 0.6]]
    df = create_dataframe(rows)
    stats = compute_stats(df)
    assert len(stats) == 1
    assert set(stats.index) == {"test"}


# if __name__ == "__main__":
#     sentences = load_sentences("data/semantic_sentences.json")
#     similarity_funcs = {
#         "spacy": semantic_similarity_spacy,
#         "bert": semantic_similarity_bert,
#         "all-MiniLM-L6-v2": functools.partial(semantic_similarity_sentence_transformers, model="all-MiniLM-L6-v2"),
#         "paraphrase-MiniLM-L6-v2": functools.partial(
#             semantic_similarity_sentence_transformers, model="paraphrase-MiniLM-L6-v2"
#         ),
#         "text-embedding-3-large": functools.partial(semantic_similarity_openai, model="text-embedding-3-large"),
#         "text-embedding-3-small": functools.partial(semantic_similarity_openai, model="text-embedding-3-small"),
#         "text-embedding-ada-002": functools.partial(semantic_similarity_openai, model="text-embedding-ada-002"),
#     }

#     rows = calculate_similarity_metrics(sentences, similarity_funcs)
#     df = create_dataframe(rows)
#     print(df)

#     stats = compute_stats(df)
#     print(stats.round(3))
#     print(stats.to_latex(float_format="%.3f", bold_rows=True, column_format="lcccc"))