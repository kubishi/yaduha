import json
import pandas as pd
import plotly.express as px
import pathlib
from yaduha.segment import semantic_similarity_sentence_transformers as semantic_similarity

def main():
    file_path = pathlib.Path('./results/evaluation_results_processed.json')
    data = json.loads(file_path.read_text())

    for result in data['results']:
        # back_translation = result['translation']['back_translation']
        # if not back_translation:
        #     # get translation from user
        #     back_translation = input(f"Please provide a back translation for the following sentence:\n{result['translation']['target']}\n")
        #     back_translation = back_translation.strip()
        #     result['translation']['back_translation'] = back_translation

        # if not back_translation: # translation cannot be provided because target sentence is grammatically incorrect
        #     result['semantic_similarity'] = 0
        # else:
        #     result['semantic_similarity'] = semantic_similarity(
        #         result['translation']['source'],
        #         back_translation,
        #         model="all-MiniLM-L6-v2"
        #     )
        # print(f"Model={result['model']} Translator={result['translator']} Sentence type={result['sentence_type']}")
        # print(f"Source: {result['translation']['source']}")
        # print(f"Target: {result['translation']['target']}")
        # print(f"Back translation: {back_translation}")
        # print(f"Semantic similarity: {result['semantic_similarity']}")
        # print()

        # # save the updated data
        # file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

        if 'semantic_similarity' not in result:
            result['semantic_similarity'] = 0


    # Step 2: Normalize the JSON data into a DataFrame
    df = pd.json_normalize(data['results'], sep='_')

    # Step 3: Select relevant columns for analysis
    df_analysis = df[['translator', 'model', 'sentence_type', 'translation_translation_time']]

    # Step 4: Calculate the median and interquartile range (IQR) for error bars
    grouped_data = df_analysis.groupby(['translator', 'model', 'sentence_type']).agg(
        median_translation_time=('translation_translation_time', 'median'),
        q1_translation_time=('translation_translation_time', lambda x: x.quantile(0.25)),
        q3_translation_time=('translation_translation_time', lambda x: x.quantile(0.75))
    ).reset_index()

    # Calculate the error bars based on IQR
    grouped_data['error_y_plus'] = grouped_data['q3_translation_time'] - grouped_data['median_translation_time']
    grouped_data['error_y_minus'] = grouped_data['median_translation_time'] - grouped_data['q1_translation_time']

    category_orders = {
        'sentence_type': [
            'subject-verb',
            'subject-verb-object',
            'two-clause',
            'two-verb',
            'complex',
            'nominalization',
        ],
        'translator': ['instructions', 'finetuned-simple', 'pipeline', 'agentic'],
        'models': ['gpt-4o-mini', 'gpt-4o'],
    }

    # Step 5: Plot median translation times with IQR-based error bars using Plotly
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
        category_orders=category_orders
    )

    # Display the plot
    savepath = pathlib.Path('./plots/translation_time.png')
    savepath.parent.mkdir(exist_ok=True, parents=True)
    fig.write_image(str(savepath))


    # Semantic Similarity Analysis
    # Step 3: Select relevant columns for analysis of semantic similarity
    df_similarity = df[['translator', 'model', 'sentence_type', 'semantic_similarity']]

    # Step 4: Calculate the median and interquartile range (IQR) for error bars of semantic similarity
    similarity_data = df_similarity.groupby(['translator', 'model', 'sentence_type']).agg(
        median_similarity=('semantic_similarity', 'median'),
        q1_similarity=('semantic_similarity', lambda x: x.quantile(0.25)),
        q3_similarity=('semantic_similarity', lambda x: x.quantile(0.75))
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
        facet_col='model',
        error_y='error_y_plus',  # Upper bound of error bar (Q3 to median)
        error_y_minus='error_y_minus',  # Lower bound of error bar (median to Q1)
        labels={
            'median_similarity': 'Median Semantic Similarity',
            'sentence_type': 'Sentence Type',
            'translator': 'Translator',
            'model': 'Model',
        },
        title='Semantic Similarity by Model, Translator, and Sentence Type',
        template='simple_white',
        category_orders=category_orders
    )

    # Display the semantic similarity plot
    savepath_similarity = pathlib.Path('./plots/semantic_similarity.png')
    savepath_similarity.parent.mkdir(exist_ok=True, parents=True)
    fig_similarity.write_image(str(savepath_similarity))


if __name__ == '__main__':
    main()
