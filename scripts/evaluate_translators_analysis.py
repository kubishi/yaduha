import json
import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from itertools import combinations
from functools import lru_cache, partial
import dotenv
from yaduha.evaluate.semantic_similarity import semantic_similarity_sentence_transformers as semantic_similarity
from yaduha.translate.pipeline import split_sentence, comparator_sentence, make_sentence

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sacrebleu

from yaduha.evaluate.bert_score import get_bertscore
from yaduha.evaluate.comet import get_comet_score_batch
from yaduha.translate.pipeline_syntax import SentenceList # pip install sacrebleu
# Ensure the NLTK package is ready
# nltk.download(quiet=True)

dotenv.load_dotenv()

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
    'instructions': 'Instructions',
    'finetuned': 'Fine-tuned',
    # 'pipeline': 'Pipeline',
    'agentic': 'Builder',
    'pipeline-new': 'Pipeline',
    'rag': 'RAG',
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
    'translator': [
        'Instructions', 
        'Fine-tuned', 
        'Pipeline', 
        # 'Pipeline V2', 
        'Builder', 
        'RAG'
    ],
    'models': ['gpt-4o-mini', 'gpt-4o'],
}

COLORS = ['#7b3294', '#c2a5cf', '#a6dba0', '#008837', '#d95f0e', '#fdae61']

def compute_chrf(reference: str, candidate: str, word_order: int = 2):
    """
    Compute the chrF++ score between a reference sentence and a candidate sentence.
    
    Parameters:
    - reference (str): The reference sentence (ground truth).
    - candidate (str): The candidate sentence (generated sentence).
    
    Returns:
    - float: chrF++ score
    """
    # Sacrebleu expects reference to be a list of references
    references = [reference]
        
    # Compute chrF++ score
    chrf_score = sacrebleu.metrics.chrf.CHRF(word_order = word_order).sentence_score(candidate, references)
    return chrf_score.score

# Function to compute BLEU score
def compute_bleu(reference: str, candidate: str) -> float:
    """
    Compute the BLEU score between a reference sentence and a candidate sentence.
    
    Parameters:
    - reference (str): The reference sentence (ground truth).
    - candidate (str): The candidate sentence (generated sentence).
    
    Returns:
    - float: BLEU score
    """
    # Tokenize the sentences
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    
    # BLEU score calculation
    smooth = SmoothingFunction().method1
    score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smooth)
    return score

@lru_cache(maxsize=None)
def load_data(do_save: bool = True,
              overwrite: bool = False,
              compute_scores: bool = False,
              skip_errors: bool = False) -> pd.DataFrame:
    file_path = thisdir / 'results' / 'evaluation_results.json'
    save_path = thisdir / 'results' / 'evaluation_results_evaluated.json'
    data = json.loads(file_path.read_text())
    data_evaluated = json.loads(save_path.read_text()) if save_path.exists() else {'results': []}

    all_results = set()
    results = []
    # Add completed results to the list
    for result in data_evaluated['results']:
        # remove metadata if present
        if 'metadata' in result['translation']:
            del result['translation']['metadata']
        results.append(result)
        all_results.add((result['translator'], result['model'], result['translation']['source']))

    # Add new results to the list
    for result in data['results']:
        # remove metadata if present
        if 'metadata' in result['translation']:
            del result['translation']['metadata']
        if (result['translator'], result['model'], result['translation']['source']) not in all_results:
            results.append(result)
            all_results.add((result['translator'], result['model'], result['translation']['source']))

    data['results'] = results

    if compute_scores:
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
                if overwrite or 'semantic_similarity' not in result or result['semantic_similarity'] == 0:
                    result['semantic_similarity'] = semantic_similarity(
                        result['translation']['source'],
                        back_translation,
                        model="all-MiniLM-L6-v2"
                    )
                    has_changed = True

                if overwrite or 'semantic_similarity_comparator' not in result or result['semantic_similarity_comparator'] == 0:
                    simple_sentences = split_sentence(
                        back_translation,
                        model='gpt-4o-mini'
                    )
                    comparator_sentences = SentenceList(
                        sentences=[
                            comparator_sentence(sentence)
                            for sentence in simple_sentences.sentences
                        ]
                    )
                    comparator = make_sentence(comparator_sentences, model='gpt-4o-mini')
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
                save_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        
        print(" " * 100, end='\r')
        print("Semantic similarities computed successfully!")


        print(f"Computing BLEU scores for {len(data['results'])} sentences...")
        for i, result in enumerate(data['results'], start=1):
            print(f"Computing BLEU score for sentence {i}/{len(data['results'])} ({i/len(data['results'])*100:.2f}%)", end='\r')
            # if bleu score is already computed, skip
            if 'bleu_score' in result and 'bleu_score_comparator' in result:
                continue
            back_translation = result['translation']['back_translation']
            if not back_translation:
                if not skip_errors:
                    raise ValueError(f"Missing back translation for the following result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                logging.warning(f"Missing back translation for the following result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                result['bleu_score'] = 0
            elif back_translation == "N/A":
                result['bleu_score'] = 0
            else:
                result['bleu_score'] = compute_bleu(result['translation']['source'], back_translation)

            comparator = 'N/A' if back_translation == "N/A" else result['comparator']
            if not comparator:
                if not skip_errors:
                    raise ValueError(f"Missing comparator for the following result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                logging.warning(f"Missing comparator for the following result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                result['bleu_score_comparator'] = 0
            elif comparator == "N/A":
                result['bleu_score_comparator'] = 0
            else:
                result['bleu_score_comparator'] = compute_bleu(result['translation']['source'], comparator)

            if do_save:
                save_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(" " * 100, end='\r')
        print("BLEU scores computed successfully!")

        print(f"Computing chrF++ scores for {len(data['results'])} sentences...")
        for i, result in enumerate(data['results'], start=1):
            print(f"Computing chrF++ score for sentence {i}/{len(data['results'])} ({i/len(data['results'])*100:.2f}%)", end='\r')
            # if bleu score is already computed, skip
            if 'chrf_score' in result and 'chrf_score_comparator' in result:
                continue
            back_translation = result['translation']['back_translation']
            if not back_translation:
                if not skip_errors:
                    raise ValueError(f"Missing back translation for the following result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                logging.warning(f"Missing back translation for the following result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                result['chrf_score'] = 0
            elif back_translation == "N/A":
                result['chrf_score'] = 0
            else:
                result['chrf_score'] = compute_chrf(result['translation']['source'], back_translation)

            comparator = 'N/A' if back_translation == "N/A" else result['comparator']
            if not comparator:
                if not skip_errors:
                    raise ValueError(f"Missing comparator for the following result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                logging.warning(f"Missing comparator for the following result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                result['chrf_score_comparator'] = 0
            elif comparator == "N/A":
                result['chrf_score_comparator'] = 0
            else:
                result['chrf_score_comparator'] = compute_chrf(result['translation']['source'], comparator)
                
            if do_save:
                save_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(" " * 100, end='\r')
        print("chrF++ scores computed successfully!")

    
        print(f"Computing BERT scores for {len(data['results'])} sentences...")
        for i, result in enumerate(data['results'], start=1):
            print(f"Computing BERT score for sentence {i}/{len(data['results'])} ({i/len(data['results'])*100:.2f}%)", end='\r')
            # if bleu score is already computed, skip
            if 'bert_score' in result and 'bert_score_comparator' in result:
                continue
            back_translation = result['translation']['back_translation']
            if not back_translation:
                if not skip_errors:
                    raise ValueError(f"Missing back translation for the following result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                logging.warning(f"Missing back translation for the following result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                result['bert_score'] = 0
            elif back_translation == "N/A":
                result['bert_score'] = 0
            else:
                result['bert_score'] = get_bertscore(result['translation']['source'], back_translation)["f1"]

            comparator = 'N/A' if back_translation == "N/A" else result['comparator']
            if not comparator:
                if not skip_errors:
                    raise ValueError(f"Missing comparator for the following result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                logging.warning(f"Missing comparator for the following result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
                result['bert_score_comparator'] = 0
            elif comparator == "N/A":
                result['bert_score_comparator'] = 0
            else:
                result['bert_score_comparator'] = get_bertscore(result['translation']['source'], comparator)["f1"]
                
            if do_save:
                save_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(" " * 100, end='\r')
        print("BERT scores computed successfully!")

        print(f"Computing COMET scores for {len(data['results'])} sentences...")
        missing_any_comet_scores = any(
            'comet_score' not in result or 'comet_score_comparator' not in result
            for result in data['results']
        )
        if missing_any_comet_scores:
            predictions = [result['translation']['source'] for result in data['results']]
            references = [result['translation']['back_translation'] for result in data['results']]
            print(f"Computing COMET scores for {len(predictions)} sentences...")
            scores = get_comet_score_batch(
                predictions=predictions,
                references=references,
                sources=predictions
            )
            
            references_comparator = [result.get('comparator', ' ') for result in data['results']]
            print(f"Computing COMET scores for {len(predictions)} sentences...")
            scores_comparator = get_comet_score_batch(
                predictions=predictions,
                references=references_comparator,
                sources=predictions
            )

            for i, result in enumerate(data['results']):
                if result['translation']['back_translation'] == "N/A":
                    result['comet_score'] = 0
                    result['comet_score_comparator'] = 0
                else:
                    result['comet_score'] = scores[i]
                    if 'comparator' in result:
                        result['comet_score_comparator'] = scores_comparator[i]
                    else:
                        result['comet_score_comparator'] = 0

            if do_save:
                save_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(" " * 100, end='\r')
        print("COMET scores computed successfully!")

    df = pd.json_normalize(data['results'], sep='_')
    # drop translators not in TRANSLATOR_NAMES.keys()
    df = df[df['translator'].isin(TRANSLATOR_NAMES.keys())]
    # rename translators
    df['translator'] = df['translator'].apply(lambda x: TRANSLATOR_NAMES[x])
    return df

def plot_translation_time():
    df = load_data(compute_scores=False)

    df_analysis = df[['translator', 'model', 'sentence_type', 'translation_translation_time']]
    grouped_data = df_analysis.groupby(['translator', 'model', 'sentence_type']).agg(
        median_translation_time=('translation_translation_time', 'median'),
        q1_translation_time=('translation_translation_time', lambda x: x.quantile(0.25)),
        q3_translation_time=('translation_translation_time', lambda x: x.quantile(0.75))
    ).reset_index()
    grouped_data['error_y_plus'] = grouped_data['q3_translation_time'] - grouped_data['median_translation_time']
    grouped_data['error_y_minus'] = grouped_data['median_translation_time'] - grouped_data['q1_translation_time']

    bar_width = 0.175  # Width of each bar
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
    df = load_data(compute_scores=True)
    bar_width = 0.175  # Adjust the width of each bar
    fontsize = 16
    x_positions = np.arange(len(CATEGORY_ORDERS['sentence_type']))  # Create fixed positions for sentence types

    plots = [
        {
            'yval': 'semantic_similarity_comparator',
            'ylabel': 'Median Semantic Similarity',
            'rulers': 'semantic_similarity',
            'offset': 0.04,
        },
        {
            'yval': 'semantic_similarity',
            'ylabel': 'Median Semantic Similarity',
            'rulers': 'semantic_similarity',
            'offset': 0.04,
        },
        {
            'yval': 'bleu_score_comparator',
            'ylabel': 'Median BLEU Score',
            'rulers': 'bleu_score',
            'offset': 0.04,
        },
        {
            'yval': 'bleu_score',
            'ylabel': 'Median BLEU Score',
            'rulers': 'bleu_score',
            'offset': 0.04,
        },
        {
            'yval': 'chrf_score_comparator',
            'ylabel': 'Median chrF++ Score',
            'rulers': 'chrf_score',
            'offset': 4.0,
        },
        {
            'yval': 'chrf_score',
            'ylabel': 'Median chrF++ Score',
            'rulers': 'chrf_score',
            'offset': 4.0,
        },
        {
            'yval': 'bert_score_comparator',
            'ylabel': 'Median BERT Score',
            'rulers': 'bert_score',
            'offset': 0.04,
        },
        {
            'yval': 'bert_score',
            'ylabel': 'Median BERT Score',
            'rulers': 'bert_score',
            'offset': 0.04,
        },
        {
            'yval': 'comet_score_comparator',
            'ylabel': 'Median COMET Score',
            'rulers': 'comet_score',
            'offset': 0.04,
        },
        {
            'yval': 'comet_score',
            'ylabel': 'Median COMET Score',
            'rulers': 'comet_score',
            'offset': 0.04,
        },
    ]

    models = ['gpt-4o-mini', 'gpt-4o']  # Define models to iterate over

    for plot in plots:
        yval = plot['yval']
        offset = plot['offset']

        # Create a figure with 2 subplots (one on top of the other), sharing the x-axis
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        handles, labels = [], []  # To collect legend handles and labels

        for i, model in enumerate(models):
            ax: plt.Axes = axes[i]  # Get the corresponding subplot
            df_model = df[df['model'] == model]
            df_similarity = df_model[['translator', 'model', 'sentence_type', yval]]

            df_similarity.loc[:, yval] += offset

            similarity_data = df_similarity.groupby(['translator', 'model', 'sentence_type']).agg(
                median_similarity=(yval, 'median'),
                q1_similarity=(yval, lambda x: x.quantile(0.25)),
                q3_similarity=(yval, lambda x: x.quantile(0.75))
            ).reset_index()

            similarity_data['error_y_plus'] = similarity_data['q3_similarity'] - similarity_data['median_similarity']
            similarity_data['error_y_minus'] = similarity_data['median_similarity'] - similarity_data['q1_similarity']

            # Plot bars with error bars
            for j, translator in enumerate(CATEGORY_ORDERS['translator']):
                print(f"Translator {j}: {translator}")
                data = similarity_data[similarity_data['translator'] == translator]
                if not data.empty:
                    bars = ax.bar(
                        x_positions + j * bar_width,
                        data['median_similarity'],
                        width=bar_width,
                        yerr=[data['error_y_minus'], data['error_y_plus']],
                        capsize=5,
                        bottom=-offset,
                        color=COLORS[j % len(COLORS)]  # Use colors from the list
                    )

                    # Add the handles and labels from the last plot (avoiding duplicates)
                    if i == len(models) - 1:
                        handle = bars[0]
                        handles.append(handle)
                        labels.append(translator)

            # Add baseline horizontal line for the semantic similarity baseline analysis
            if plot['rulers']:
                ss_mean, ss_std = similarity_baseline_analysis(metric=plot['rulers'])
                ax.axhline(
                    y=ss_mean,
                    color='black',
                    linestyle='--',
                    label=None
                )
                ax.axhline(
                    y=ss_mean + 3 * ss_std,
                    color='black',
                    linestyle='--',
                    label='$\mu$ and $\mu + 3\sigma$'
                )

            # Set plot title and axis labels
            ax.set_ylim(-offset, df_similarity[yval].max() + offset)
            ax.set_title(f'Model: {model}', fontsize=fontsize)
            ax.set_ylabel(plot['ylabel'], fontsize=fontsize)

        # Configure shared x-axis labels and ticks
        axes[-1].set_xlabel('Sentence Type', fontsize=fontsize)
        plt.xticks(x_positions + bar_width * (len(CATEGORY_ORDERS['translator']) - 1) / 2,
                   CATEGORY_ORDERS['sentence_type'], rotation=45, fontsize=fontsize)
        # plt.legend(title='Translator', bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()

        # Adjust layout and add the figure-wide legend
        fig.tight_layout()
        fig.legend(
            handles, labels,
            title='Translator',
            bbox_to_anchor=(1.0, 0.95),
            loc='upper left',
            fontsize=fontsize,
            title_fontsize=fontsize
        )

        savepath = thisdir / f'plots/{yval}.{FILETYPE}'
        savepath.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(savepath, bbox_inches='tight')  # Ensure everything fits including the legend
        plt.close()

def plot_cost():
    df = load_data(compute_scores=False)

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

    bar_width = 0.175  # Width of each bar
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
    df = load_data(compute_scores=False)

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
    df = load_data(compute_scores=False)

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


METRICS = {
    "semantic_similarity": partial(semantic_similarity, model="all-MiniLM-L6-v2"),
    "bleu_score": compute_bleu,
    "chrf_score": partial(compute_chrf, word_order=2),
    "bert_score": lambda reference, candidate: get_bertscore(reference, candidate)["f1"],
    "comet_score": get_comet_score_batch
}

@lru_cache(maxsize=None)
def similarity_baseline_analysis(metric: str, overwrite: bool = False) -> Tuple[float, float]:
    """Runs semantic similirity on pairs of sentences to determine baseline similarity"""
    if metric not in METRICS:
        raise ValueError(f"Invalid metric: {metric}. Choose from {list(METRICS.keys())}")
    
    eval_func = METRICS[metric]

    savepath = thisdir / f'results/{metric}_baseline.json'
    savepath.parent.mkdir(exist_ok=True, parents=True)

    if overwrite or not savepath.exists():
        df = pd.read_csv(thisdir / 'data/evaluation_sentences.csv')
        sentences = df['sentence'].tolist()
        # for all pairs of sentences
        similarities = []
        total_pairs = len(sentences) * (len(sentences) - 1) // 2
        if metric == "comet_score": # batch comet_score function
            combos = list(combinations(sentences, 2))
            similarities = eval_func(
                predictions=[s1 for s1, _ in combos],
                references=[s2 for _, s2 in combos],
                sources=[s1 for s1, _ in combos]
            )
        else:
            for i, (s1, s2) in enumerate(combinations(sentences, 2), start=1):
                print(f"Computing similarity for sentence pair {i}/{total_pairs} ({i/total_pairs*100:.2f}%)", end='\r')
                similarity = eval_func(s1, s2)
                similarities.append(similarity)
        print(" " * 100, end='\r')
        print("Similarity computed successfully!")

        # save similarities to file
        savepath.write_text(json.dumps(similarities, indent=2, ensure_ascii=False))

    similarities = json.loads(savepath.read_text())

    mean = np.mean(similarities)
    std = np.std(similarities)

    print(f"Baseline similarity: {mean:.3f} ± {std:.3f}")

    # Plot histogram
    plt.figure(figsize=(6, 6))
    plt.hist(similarities, bins=20, color=COLORS[0], edgecolor='black')
    # set font size for all elements to 24
    plt.title('Similarity Distribution', fontsize=24)
    plt.xlabel(metric, fontsize=24)
    plt.ylabel('Frequency', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.tight_layout()

    savepath = thisdir / f'plots/{metric}_baseline.{FILETYPE}'
    savepath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(savepath)
    plt.close()

    return mean, std

def similarity_baseline_latex_table():
    # create table with the following columns:
    # metric, mean, std
    rows = []
    for metric in METRICS.keys():
        mean, std = similarity_baseline_analysis(metric=metric)
        rows.append([metric, f"{mean:.3f}", f"{std:.3f}"])
    df = pd.DataFrame(rows, columns=["Metric", "Mean", "Standard Deviation"])
    latex_table = df.to_latex(index=False,
                              columns=["Metric", "Mean", "Standard Deviation"],
                              header=["Metric", "Mean", "Standard Deviation"],
                              float_format="%.3f",
                              caption='Summary of Baseline Similarity Analysis',
                              label='tab:baseline_similarity_summary')
    
    savepath = thisdir / 'results/baseline_similarity_summary_table.tex'
    savepath.parent.mkdir(exist_ok=True, parents=True)
    savepath.write_text(latex_table)


def get_interesting_examples():
    # get example where Builder comparator is better than Pipeline
    print(f"=== Examples where Builder is better than Pipeline ===")
    df = load_data(compute_scores=True)
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
              (df['translator'] == 'Instructions') & 
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
    # plot_semantic_similarity()
    # plot_translation_time()
    # plot_cost()
    # generate_cost_latex_table()
    # generate_translation_time_latex_table()
    similarity_baseline_latex_table()
    # semantic_similarity_baseline_analysis()
    # get_interesting_examples()

if __name__ == '__main__':
    main()
