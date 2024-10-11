import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from functools import lru_cache
from yaduha.segment import make_sentence, semantic_similarity_sentence_transformers as semantic_similarity
from yaduha.segment import (
    split_sentence, 
)
from yaduha.forward.pipeline import split_sentence, comparator_sentence

thisdir = pathlib.Path(__file__).parent.resolve()

FILETYPE = 'png'
# set up for latex fonts
if FILETYPE == 'pdf':
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

TRANSLATOR_NAMES = {
    'instructions': 'Instructions-Based',
    'finetuned-simple': 'Finetuned',
    'pipeline': 'Pipeline',
    'agentic': 'Builder',
}

CATEGORY_ORDERS = {
    'sentence_type': [
        'subject-verb',
        'subject-verb-object',
        'two-verb',
        'two-clause',
        'complex',
        'nominalization',
    ],
    'translator': [n for n in TRANSLATOR_NAMES.values()],
    'models': ['gpt-4o-mini', 'gpt-4o'],
}

COLORS = ['#7b3294', '#c2a5cf', '#a6dba0', '#008837']

@lru_cache(maxsize=None)
def load_data(do_save: bool = True,
              overwrite: bool = False,
              compute_semantic_similarity: bool = False,
              skip_errors: bool = False) -> pd.DataFrame:
    file_path = pathlib.Path('./results/evaluation_results.json')
    data = json.loads(file_path.read_text())

    if compute_semantic_similarity:
        print(f"Computing semantic similarities for {len(data['results'])} sentences...")
        for i, result in enumerate(data['results'], start=1):
            print(f"Computing semantic similarity for sentence {i}/{len(data['results'])} ({i/len(data['results'])*100:.2f}%)", end='\r')
            back_translation = result['translation']['back_translation']
            has_changed = False
            if not back_translation:
                if not skip_errors:
                    raise ValueError(f"Missing back translation for the following result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                logging.warning(f"Missing back translation for the following result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                result['semantic_similarity'] = 0
                result['semantic_similarity_comparator'] = 0
            elif back_translation == "N/A":
                result['semantic_similarity'] = 0
                result['semantic_similarity_comparator'] = 0
            else:
                if overwrite or 'semantic_similarity' not in result:
                    result['semantic_similarity'] = semantic_similarity(
                        result['translation']['source'],
                        back_translation,
                        model="all-MiniLM-L6-v2"
                    )
                    has_changed = True

                if overwrite or 'semantic_similarity_comparator' not in result:
                    simple_sentences = split_sentence(
                        back_translation,
                        model='gpt-4o-mini'
                    )
                    comparator_sentences = [
                        comparator_sentence(sentence)
                        for sentence in simple_sentences
                    ]
                    comparator = ". ".join([
                        make_sentence(sentence, model='gpt-4o-mini')
                        for sentence in comparator_sentences
                    ]) + '.'
                    result['comparator'] = comparator

                    result['semantic_similarity_comparator'] = semantic_similarity(
                        result['translation']['source'],
                        comparator,
                        model="all-MiniLM-L6-v2"
                    )
                    has_changed = True

            if 'semantic_similarity' not in result:
                result['semantic_similarity'] = 0
    
            if do_save and has_changed:
                file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        
        print(" " * 100, end='\r')
        print("Semantic similarities computed successfully!")

    df = pd.json_normalize(data['results'], sep='_')
    # rename translator names
    # print(df['translator'])
    df['translator'] = df['translator'].apply(lambda x: TRANSLATOR_NAMES[x])
    return df

def plot_translation_time():
    df = load_data(compute_semantic_similarity=False)

    df_analysis = df[['translator', 'model', 'sentence_type', 'translation_translation_time']]
    grouped_data = df_analysis.groupby(['translator', 'model', 'sentence_type']).agg(
        median_translation_time=('translation_translation_time', 'median'),
        q1_translation_time=('translation_translation_time', lambda x: x.quantile(0.25)),
        q3_translation_time=('translation_translation_time', lambda x: x.quantile(0.75))
    ).reset_index()
    grouped_data['error_y_plus'] = grouped_data['q3_translation_time'] - grouped_data['median_translation_time']
    grouped_data['error_y_minus'] = grouped_data['median_translation_time'] - grouped_data['q1_translation_time']

    bar_width = 0.2  # Width of each bar
    x_positions = np.arange(len(CATEGORY_ORDERS['sentence_type']))  # X-axis positions for the sentence types

    for model in grouped_data['model'].unique():
        plt.figure(figsize=(10, 6))
        subset = grouped_data[grouped_data['model'] == model]
        
        for i, translator in enumerate(CATEGORY_ORDERS['translator']):
            data = subset[subset['translator'] == translator]
            if not data.empty:
                # Offset each translator's bars by its index position
                plt.bar(
                    x_positions + i * bar_width,
                    data['median_translation_time'],
                    width=bar_width,
                    yerr=[data['error_y_minus'], data['error_y_plus']],
                    label=translator,
                    capsize=5,
                    color=COLORS[i % len(COLORS)]  # Use colors from the list
                )

        plt.title(f'Translation Time by Sentence Type and Translator ({model})')
        plt.xlabel('Sentence Type')
        plt.ylabel('Median Translation Time (s)')
        plt.xticks(x_positions + bar_width * (len(CATEGORY_ORDERS['translator']) - 1) / 2,
                   CATEGORY_ORDERS['sentence_type'], rotation=45)
        plt.legend(title='Translator')
        plt.tight_layout()

        savepath = thisdir / f'plots/translation_time_{model}.{FILETYPE}'
        savepath.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(savepath)
        plt.close()

import matplotlib.pyplot as plt
import numpy as np

def plot_semantic_similarity():
    df = load_data(compute_semantic_similarity=True)
    bar_width = 0.2  # Adjust the width of each bar
    x_positions = np.arange(len(CATEGORY_ORDERS['sentence_type']))  # Create fixed positions for sentence types

    plots = [
        {
            'model': 'gpt-4o-mini',
            'yval': 'semantic_similarity_comparator',
            'title': 'Semantic Similarity w/ Comparator (gpt-4o-mini)',
        },
        {
            'model': 'gpt-4o',
            'yval': 'semantic_similarity_comparator',
            'title': 'Semantic Similarity w/ Comparator (gpt-4o)',
        },
        {
            'model': 'gpt-4o-mini',
            'yval': 'semantic_similarity',
            'title': 'Semantic Similarity w/ Backwards Translation (gpt-4o-mini)',
        },
        {
            'model': 'gpt-4o',
            'yval': 'semantic_similarity',
            'title': 'Semantic Similarity w/ Backwards Translation (gpt-4o)',
        },
    ]

    for plot in plots:
        model = plot['model']
        yval = plot['yval']
        title = plot['title']

        df_model = df[df['model'] == model]
        df_similarity = df_model[['translator', 'model', 'sentence_type', yval]]
        similarity_data = df_similarity.groupby(['translator', 'model', 'sentence_type']).agg(
            median_similarity=(yval, 'median'),
            q1_similarity=(yval, lambda x: x.quantile(0.25)),
            q3_similarity=(yval, lambda x: x.quantile(0.75))
        ).reset_index()

        similarity_data['error_y_plus'] = similarity_data['q3_similarity'] - similarity_data['median_similarity']
        similarity_data['error_y_minus'] = similarity_data['median_similarity'] - similarity_data['q1_similarity']

        plt.figure(figsize=(10, 6))
        for i, translator in enumerate(CATEGORY_ORDERS['translator']):
            data = similarity_data[similarity_data['translator'] == translator]
            if not data.empty:
                plt.bar(
                    x_positions + i * bar_width,
                    data['median_similarity'],
                    width=bar_width,
                    yerr=[data['error_y_minus'], data['error_y_plus']],
                    label=translator,
                    capsize=5,
                    color=COLORS[i % len(COLORS)]  # Use colors from the list
                )

        plt.title(title)
        plt.xlabel('Sentence Type')
        plt.ylabel('Median Semantic Similarity')
        plt.xticks(x_positions + bar_width * (len(CATEGORY_ORDERS['translator']) - 1) / 2,
                    CATEGORY_ORDERS['sentence_type'], rotation=45)
        plt.legend(title='Translator')
        plt.tight_layout()

        savepath = thisdir / f'plots/{yval}_{model}.{FILETYPE}'
        savepath.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(savepath)
        plt.close()



def plot_cost():
    df = load_data(compute_semantic_similarity=False)

    model_prices = {
        "gpt-4o": {"prompt": 2.50, "completion": 10.00},
        "gpt-4o-mini": {"prompt": 0.150, "completion": 0.600}
    }

    def get_cost(row, type: str):
        model = row['model']
        cost = (
            row[f'translation_{type}_prompt_tokens'] / 1_000_000 * model_prices[model]["prompt"] +
            row[f'translation_{type}_completion_tokens'] / 1_000_000 * model_prices[model]["completion"]
        )
        return cost
    
    df['translation_cost'] = df.apply(lambda row: get_cost(row, 'translation'), axis=1)
    df['back_translation_cost'] = df.apply(lambda row: get_cost(row, 'back_translation'), axis=1)
    df['total_cost'] = df['translation_cost'] + df['back_translation_cost']

    df_cost = df[['translator', 'model', 'sentence_type', 'total_cost']]
    grouped_data = df_cost.groupby(['translator', 'model', 'sentence_type']).agg(
        median_cost=('total_cost', 'median'),
        q1_cost=('total_cost', lambda x: x.quantile(0.25)),
        q3_cost=('total_cost', lambda x: x.quantile(0.75))
    ).reset_index()
    grouped_data['error_y_plus'] = grouped_data['q3_cost'] - grouped_data['median_cost']
    grouped_data['error_y_minus'] = grouped_data['median_cost'] - grouped_data['q1_cost']

    bar_width = 0.2  # Width of each bar
    x_positions = np.arange(len(CATEGORY_ORDERS['sentence_type']))  # X-axis positions for the sentence types

    for model in grouped_data['model'].unique():
        plt.figure(figsize=(10, 6))
        subset = grouped_data[grouped_data['model'] == model]

        for i, translator in enumerate(CATEGORY_ORDERS['translator']):
            data = subset[subset['translator'] == translator]
            if not data.empty:
                # Offset each translator's bars by its index position
                plt.bar(
                    x_positions + i * bar_width,
                    data['median_cost'],
                    width=bar_width,
                    yerr=[data['error_y_minus'], data['error_y_plus']],
                    label=translator,
                    capsize=5,
                    color=COLORS[i % len(COLORS)]  # Use colors from the list
                )

        plt.title(f'Cost by Sentence Type and Translator ({model})')
        plt.xlabel('Sentence Type')
        plt.ylabel('Median Cost ($)')
        plt.xticks(x_positions + bar_width * (len(CATEGORY_ORDERS['translator']) - 1) / 2,
                   CATEGORY_ORDERS['sentence_type'], rotation=45)
        plt.legend(title='Translator')
        plt.tight_layout()

        savepath = thisdir / f'plots/cost_{model}.{FILETYPE}'
        savepath.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(savepath)
        plt.close()


def main():
    plot_translation_time()
    plot_semantic_similarity()
    plot_cost()

if __name__ == '__main__':
    main()
