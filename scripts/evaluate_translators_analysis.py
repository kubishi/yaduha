import json
import logging
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from itertools import combinations
from functools import lru_cache
from yaduha.segment import make_sentence, semantic_similarity_sentence_transformers as semantic_similarity
from yaduha.segment import (
    split_sentence, 
)
from yaduha.forward.pipeline import split_sentence, comparator_sentence

thisdir = pathlib.Path(__file__).parent.resolve()

FILETYPE = 'pdf'
# set up for latex fonts
if FILETYPE == 'pdf':
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
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

    OFFSET = 0.04

    for plot in plots:
        model = plot['model']
        yval = plot['yval']
        title = plot['title']

        df_model = df[df['model'] == model]
        df_similarity = df_model[['translator', 'model', 'sentence_type', yval]]
        
        df_similarity.loc[:, yval] += OFFSET

        similarity_data = df_similarity.groupby(['translator', 'model', 'sentence_type']).agg(
            median_similarity=(yval, 'median'),
            q1_similarity=(yval, lambda x: x.quantile(0.25)),
            q3_similarity=(yval, lambda x: x.quantile(0.75))
        ).reset_index()

        similarity_data['error_y_plus'] = similarity_data['q3_similarity'] - similarity_data['median_similarity']
        similarity_data['error_y_minus'] = similarity_data['median_similarity'] - similarity_data['q1_similarity']


        plt.figure(figsize=(10, 4))
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
                    bottom=-OFFSET,
                    color=COLORS[i % len(COLORS)]  # Use colors from the list
                )

        ss_mean, ss_std = semantic_similarity_baseline_analysis()
        # add a horizontal line for the baseline
        plt.axhline(
            y=ss_mean,
            color='black',
            linestyle='--',
            label=None
        )
        
        plt.axhline(
            y=ss_mean + 3*ss_std,
            color='black',
            linestyle='--',
            label='$\mu$ and $\mu + 3\sigma$'
        )

        # make y-axis start at -0.02
        plt.ylim(-OFFSET, 1+OFFSET)

        plt.title(title)
        plt.xlabel('Sentence Type')
        plt.ylabel('Median Semantic Similarity')
        plt.xticks(x_positions + bar_width * (len(CATEGORY_ORDERS['translator']) - 1) / 2,
                    CATEGORY_ORDERS['sentence_type'], rotation=45)
        plt.legend(title='Translator') #, loc='upper left')
        # legend outside of plot
        plt.legend(title='Translator', bbox_to_anchor=(1.05, 1), loc='upper left')
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
        plt.ylabel('Median Cost (\$)')
        plt.xticks(x_positions + bar_width * (len(CATEGORY_ORDERS['translator']) - 1) / 2,
                   CATEGORY_ORDERS['sentence_type'], rotation=45)
        plt.legend(title='Translator')
        plt.tight_layout()

        savepath = thisdir / f'plots/cost_{model}.{FILETYPE}'
        savepath.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(savepath)
        plt.close()

def generate_cost_latex_table():
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

    # Grouping data to compute average and std for costs by model and translator only
    df_cost = df[['translator', 'model', 'total_cost']]
    grouped_data = df_cost.groupby(['translator', 'model']).agg(
        average_cost=('total_cost', 'mean'),
        std_cost=('total_cost', 'std')
    ).reset_index()

    # Generating the LaTeX table
    latex_table = grouped_data.to_latex(index=False, 
                                        columns=['translator', 'model', 'average_cost', 'std_cost'],
                                        header=['Translator', 'Model', 'Average Cost', 'Standard Deviation'],
                                        float_format="%.3f",
                                        caption='Summary of Average Costs and Standard Deviation by Model and Translator',
                                        label='tab:average_cost_summary')

    with open('results/average_cost_summary_table.tex', 'w') as file:
        file.write(latex_table)


def generate_translation_time_latex_table():
    df = load_data(compute_semantic_similarity=False)

    # Selecting relevant columns
    df_analysis = df[['translator', 'model', 'translation_translation_time']]

    # Grouping data to compute average and standard deviation for translation times by model and translator
    grouped_data = df_analysis.groupby(['translator', 'model']).agg(
        average_translation_time=('translation_translation_time', 'mean'),
        std_translation_time=('translation_translation_time', 'std')
    ).reset_index()

    # Generating the LaTeX table
    latex_table = grouped_data.to_latex(index=False, 
                                        columns=['translator', 'model', 'average_translation_time', 'std_translation_time'],
                                        header=['Translator', 'Model', 'Average Translation Time', 'Standard Deviation'],
                                        float_format="%.3f",
                                        caption='Summary of Average Translation Time and Standard Deviation by Model and Translator',
                                        label='tab:average_translation_time_summary')

    with open('results/average_translation_time_summary_table.tex', 'w') as file:
        file.write(latex_table)

@lru_cache(maxsize=None)
def semantic_similarity_baseline_analysis(overwrite: bool = False) -> Tuple[float, float]:
    """Runs semantic similirity on pairs of sentences to determine baseline similarity"""

    savepath = thisdir / 'results/semantic_similarity_baseline.json'
    savepath.parent.mkdir(exist_ok=True, parents=True)

    if overwrite or not savepath.exists():
        df = pd.read_csv(thisdir / 'data/evaluation_sentences.csv')
        sentences = df['sentence'].tolist()
        # for all pairs of sentences
        similarities = []
        total_pairs = len(sentences) * (len(sentences) - 1) // 2
        for i, (s1, s2) in enumerate(combinations(sentences, 2), start=1):
            print(f"Computing semantic similarity for sentence pair {i}/{total_pairs} ({i/total_pairs*100:.2f}%)", end='\r')
            similarity = semantic_similarity(s1, s2, model="all-MiniLM-L6-v2")
            similarities.append(similarity)
        print(" " * 100, end='\r')
        print("Semantic similarities computed successfully!")

        # save similarities to file
        savepath.write_text(json.dumps(similarities, indent=2, ensure_ascii=False))

    similarities = json.loads(savepath.read_text())

    mean = np.mean(similarities)
    std = np.std(similarities)

    print(f"Baseline semantic similarity: {mean:.3f} ± {std:.3f}")

    # Plot histogram
    plt.figure(figsize=(6, 6))
    plt.hist(similarities, bins=20, color=COLORS[0], edgecolor='black')
    # set font size for all elements to 24
    plt.title('Semantic Similarity Distribution', fontsize=24)
    plt.xlabel('Semantic Similarity', fontsize=24)
    plt.ylabel('Frequency', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)


    plt.tight_layout()

    savepath = thisdir / f'plots/semantic_similarity_baseline.{FILETYPE}'
    savepath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(savepath)
    plt.close()

    return mean, std

def get_interesting_examples():
    # get example where Builder comparator is better than Pipeline
    print(f"=== Examples where Builder is better than Pipeline ===")
    df = load_data(compute_semantic_similarity=True)
    print(df.columns)
    df_1 = df[df['translator'].isin(['Pipeline', 'Builder'])]
    df_1 = df_1[df_1['model'] == 'gpt-4o']
    df_1['diff'] = df_1.groupby('translation_source')['semantic_similarity_comparator'].diff()
    # get the best 5 examples
    best_examples = df_1.sort_values('diff', ascending=False).head(5)

    for i, best_example in best_examples.iterrows():
        print(f"Example {i+1}")
        print(f"Translator: {best_example['translator']}")
        print(f"Model: {best_example['model']}")
        print(f"Source: {best_example['translation_source']}")
        print(f"Target: {best_example['translation_target']}")
        print(f"Backwards: {best_example['translation_back_translation']}")
        print(f"Builder: {best_example['comparator']}")
        print(f"Semantic Similarity: {best_example['semantic_similarity']:.3f}")
        print(f"Semantic Similarity Comparator: {best_example['semantic_similarity_comparator']:.3f}")
        print()

    # Get example for Builder/gpt-4o where the back translation and comparator semantic similiarity are high
    print(f"=== Examples where Instructions-Based/gpt-4o has high semantic similarity with back translation and comparator ===")
    df_2 = df[df['translator'] == 'Pipeline']
    df_2 = df_2[df_2['model'] == 'gpt-4o']
    # get top 5 semnatic similarity + comparator
    df_2['sum'] = df_2['semantic_similarity'] + df_2['semantic_similarity_comparator']
    best_examples = df_2.sort_values('sum', ascending=False).head(5)

    for i, best_example in best_examples.iterrows():
        print(f"Example {i+1}")
        print(f"Translator: {best_example['translator']}")
        print(f"Model: {best_example['model']}")
        print(f"Source: {best_example['translation_source']}")
        print(f"Target: {best_example['translation_target']}")
        print(f"Backwards: {best_example['translation_back_translation']}")
        print(f"Builder: {best_example['comparator']}")
        print(f"Semantic Similarity: {best_example['semantic_similarity']:.3f}")
        print(f"Semantic Similarity Comparator: {best_example['semantic_similarity_comparator']:.3f}")
        print()

    # when the backwards translation is good but the comparator translation is bad.
    print(f"=== Examples where the back translation is good but the comparator translation is bad ===")
    df_3 = df[df['translator'] == 'Pipeline']
    df_3 = df_3[df_3['model'] == 'gpt-4o']
    # get high semantic similarity
    df_3 = df_3[df_3['semantic_similarity'] > 0.95]
    # get min 5 semnatic similarity + comparator
    best_examples = df_3.sort_values('semantic_similarity_comparator', ascending=True).head(5)

    for i, best_example in best_examples.iterrows():
        print(f"Example {i+1}")
        print(f"Translator: {best_example['translator']}")
        print(f"Model: {best_example['model']}")
        print(f"Source: {best_example['translation_source']}")
        print(f"Target: {best_example['translation_target']}")
        print(f"Backwards: {best_example['translation_back_translation']}")
        print(f"Builder: {best_example['comparator']}")
        print(f"Semantic Similarity: {best_example['semantic_similarity']:.3f}")
        print(f"Semantic Similarity Comparator: {best_example['semantic_similarity_comparator']:.3f}")
        print()

    # example of a bad translation overall
    print(f"=== Examples of bad translations overall ===")
    
    df_4 = df[df['translator'] == 'Pipeline']
    df_4 = df_4[df_4['model'] == 'gpt-4o']
    df_4 = df_4[df_4['sentence_type'] == 'complex']
    # get min 5 semnatic similarity
    best_examples = df_4.sort_values('semantic_similarity', ascending=True).head(5)

    for i, best_example in best_examples.iterrows():
        print(f"Example {i+1}")
        print(f"Translator: {best_example['translator']}")
        print(f"Model: {best_example['model']}")
        print(f"Source: {best_example['translation_source']}")
        print(f"Target: {best_example['translation_target']}")
        print(f"Backwards: {best_example['translation_back_translation']}")
        print(f"Builder: {best_example['comparator']}")
        print(f"Semantic Similarity: {best_example['semantic_similarity']:.3f}")
        print(f"Semantic Similarity Comparator: {best_example['semantic_similarity_comparator']:.3f}")
        print()

    # get examples where back_translation is "N/A"
    print(f"=== Examples where back translation is N/A ===")
    df_5 = df[(df['translation_back_translation'] == "N/A") & 
              (df['translator'] == 'Instructions-Based') & 
              (df['model'] == 'gpt-4o-mini')]
    df_5 = df_5.sort_values('semantic_similarity', ascending=False).head(5)
    for i, example in df_5.iterrows():
        print(f"Example {i+1}")
        print(f"Translator: {example['translator']}")
        print(f"Model: {example['model']}")
        print(f"Source: {example['translation_source']}")
        print(f"Target: {example['translation_target']}")
        print(f"Backwards: {example['translation_back_translation']}")
        print(f"Comparator: {example['comparator']}")
        print(f"Semantic Similarity: {example['semantic_similarity']:.3f}")
        print(f"Semantic Similarity Comparator: {example['semantic_similarity_comparator']:.3f}")
        print()


def main():
    plot_semantic_similarity()
    plot_translation_time()
    plot_cost()
    generate_cost_latex_table()
    generate_translation_time_latex_table()
    semantic_similarity_baseline_analysis()
    get_interesting_examples()

if __name__ == '__main__':
    main()
