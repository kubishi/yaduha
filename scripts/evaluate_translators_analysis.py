import json
import logging
import pandas as pd
import plotly.express as px
import pathlib
from functools import lru_cache
from yaduha.segment import make_sentence, semantic_similarity_sentence_transformers as semantic_similarity
from yaduha.segment import (
    split_sentence, 
)
from yaduha.forward.pipeline import split_sentence, comparator_sentence

thisdir = pathlib.Path(__file__).parent.resolve()

CATEGORY_ORDERS = {
    'sentence_type': [
        'subject-verb',
        'subject-verb-object',
        'two-verb',
        'two-clause',
        'complex',
        'nominalization',
    ],
    'translator': ['instructions', 'finetuned-simple', 'pipeline', 'agentic'],
    'models': ['gpt-4o-mini', 'gpt-4o'],
}

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
            if not back_translation: # translation cannot be provided because target sentence is grammatically incorrect
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

            if 'semantic_similarity' not in result:
                result['semantic_similarity'] = 0
    
            if do_save:
                file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        
        print(" " * 100, end='\r')
        print("Semantic similarities computed successfully!")

    # Step 2: Normalize the JSON data into a DataFrame
    df = pd.json_normalize(data['results'], sep='_')
    return df

def plot_translation_time():
    df = load_data(compute_semantic_similarity=False)

    # Translation Time Analysis
    df_analysis = df[['translator', 'model', 'sentence_type', 'translation_translation_time']]
    grouped_data = df_analysis.groupby(['translator', 'model', 'sentence_type']).agg(
        median_translation_time=('translation_translation_time', 'median'),
        q1_translation_time=('translation_translation_time', lambda x: x.quantile(0.25)),
        q3_translation_time=('translation_translation_time', lambda x: x.quantile(0.75))
    ).reset_index()
    grouped_data['error_y_plus'] = grouped_data['q3_translation_time'] - grouped_data['median_translation_time']
    grouped_data['error_y_minus'] = grouped_data['median_translation_time'] - grouped_data['q1_translation_time']

    fig = px.bar(
        grouped_data,
        x='sentence_type',
        y='median_translation_time',
        color='translator',
        barmode='group',
        facet_col='model',
        error_y='error_y_plus',  # Upper bound of error bar (Q3 to median)
        error_y_minus='error_y_minus',  # Lower bound of error bar (median to Q1)
        labels={'median_translation_time': 'Median Translation Time (s)'},
        title='Translation Time by Model, Translator, and Sentence Type',
        template='simple_white',
        category_orders=CATEGORY_ORDERS
    )

    savepath = pathlib.Path('./plots/translation_time.png')
    savepath.parent.mkdir(exist_ok=True, parents=True)
    fig.write_image(str(savepath))


def plot_semantic_similarity():
    df = load_data(compute_semantic_similarity=True)
    for model in df['model'].unique():
        df_model = df[df['model'] == model]
        for yval in ['semantic_similarity', 'semantic_similarity_comparator']:
            # Semantic Similarity Analysis
            df_similarity = df_model[['translator', 'model', 'sentence_type', yval]]
            similarity_data = df_similarity.groupby(['translator', 'model', 'sentence_type']).agg(
                median_similarity=(yval, 'median'),
                q1_similarity=(yval, lambda x: x.quantile(0.25)),
                q3_similarity=(yval, lambda x: x.quantile(0.75))
            ).reset_index()

            # Calculate the error bars based on IQR for semantic similarity
            similarity_data['error_y_plus'] = similarity_data['q3_similarity'] - similarity_data['median_similarity']
            similarity_data['error_y_minus'] = similarity_data['median_similarity'] - similarity_data['q1_similarity']

            # Step 5: Plot semantic similarity with IQR-based error bars using Plotly
            fig_similarity = px.bar(
                similarity_data,
                x='sentence_type',
                y='median_similarity',
                color='translator',
                barmode='group',
                error_y='error_y_plus',  # Upper bound of error bar (Q3 to median)
                error_y_minus='error_y_minus',  # Lower bound of error bar (median to Q1)
                labels={
                    'median_similarity': 'Median Semantic Similarity',
                    'sentence_type': 'Sentence Type',
                    'translator': 'Translator',
                    'model': 'Model',
                },
                title=f'Semantic Similarity by Translator and Sentence Type ({model})',
                template='simple_white',
                category_orders=CATEGORY_ORDERS
            )

            # Display the semantic similarity plot
            savepath_similarity = pathlib.Path(f'./plots/{yval}-{model}.png')
            savepath_similarity.parent.mkdir(exist_ok=True, parents=True)
            fig_similarity.write_image(str(savepath_similarity))

def plot_cost():
    df = load_data(compute_semantic_similarity=False)

    model_prices = { # price per million tokens
        "gpt-4o": {
            "prompt": 2.50,
            "completion": 10.00,
        },
        "gpt-4o-mini": {
            "prompt": 0.150,
            "completion": 0.600
        }	
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

    # Cost Analysis
    df_cost = df[['translator', 'model', 'sentence_type', 'total_cost']]
    grouped_data = df_cost.groupby(['translator', 'model', 'sentence_type']).agg(
        median_cost=('total_cost', 'median'),
        q1_cost=('total_cost', lambda x: x.quantile(0.25)),
        q3_cost=('total_cost', lambda x: x.quantile(0.75))
    ).reset_index()
    grouped_data['error_y_plus'] = grouped_data['q3_cost'] - grouped_data['median_cost']
    grouped_data['error_y_minus'] = grouped_data['median_cost'] - grouped_data['q1_cost']

    fig = px.bar(
        grouped_data,
        x='sentence_type',
        y='median_cost',
        color='translator',
        barmode='group',
        facet_col='model',
        error_y='error_y_plus',  # Upper bound of error bar (Q3 to median)
        error_y_minus='error_y_minus',  # Lower bound of error bar (median to Q1)
        labels={'median_cost': 'Median Cost ($)'},
        title='Cost by Model, Translator, and Sentence Type',
        template='simple_white',
        category_orders=CATEGORY_ORDERS
    )

    savepath = pathlib.Path('./plots/cost.png')
    savepath.parent.mkdir(exist_ok=True, parents=True)
    fig.write_image(str(savepath))

def main():
    plot_translation_time()
    plot_semantic_similarity()
    plot_cost()
    


if __name__ == '__main__':
    main()
