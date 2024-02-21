from functools import partial
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from segment import semantic_similarity_spacy, semantic_similarity_bert, semantic_similarity_sentence_transformers
import plotly.express as px
import time
import numpy as np
import random

np.random.seed(0)
random.seed(0)

thisdir = pathlib.Path(__file__).parent.absolute()


def main():
    path = thisdir / '.data' / 'sentences-translated.csv'
    savedir = thisdir.joinpath('.output')
    savedir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(path, index_col=0)
    df = df.dropna()

    # replace "he/she/it" (not case-sensitive) with "he"
    df['backwards'] = df['backwards'].str.replace(r'\b([Hh]e/[Ss]he/[Ii]t)\b', 'he', regex=True)

    # semantic_similarity = semantic_similarity_spacy
    # semantic_similarity = semantic_similarity_bert
    semantic_similarity = partial(semantic_similarity_sentence_transformers, model='all-MiniLM-L6-v2')

    # compute similarity metrics
    df['sim_simple'] = df.apply(lambda row: semantic_similarity(row['sentence'], row['simple']), axis=1)
    df['sim_comparator'] = df.apply(lambda row: semantic_similarity(row['sentence'], row['comparator']), axis=1)
    df['sim_backwards'] = df.apply(lambda row: semantic_similarity(row['sentence'], row['backwards']), axis=1)

    n_samples = 200
    baseline_similarities = []
    for line1, line2 in zip(df.sample(n_samples, replace=True)['sentence'].values, df.sample(n_samples, replace=True)['sentence'].values):
        bsim = semantic_similarity(line1, line2)
        baseline_similarities.append(bsim)

    # plot baseline similarities as histogram with plotly
    fig = px.histogram(
        x=baseline_similarities,
        template='plotly_white',
        # make the bars gray
        color_discrete_sequence=['gray'],
    )
    # add x and y axis labels
    fig.update_xaxes(title_text='semantic similarity')
    fig.update_yaxes(title_text='count')
    
    fig.write_html(savedir / 'baseline_similarities.html')
    fig.write_image(savedir / 'baseline_similarities.pdf')
    time.sleep(1)
    fig.write_image(savedir / 'baseline_similarities.pdf') # do twice because of bug
        
    sim_mean = np.mean(baseline_similarities)
    sim_std = np.std(baseline_similarities)

    print(f"Mean: {sim_mean}")
    print(f"Std: {sim_std}")

    sim_metrics = ['sim_simple', 'sim_comparator', 'sim_backwards']
    df['translation_quality'] = df.apply(lambda row: row[sim_metrics].mean(), axis=1)
    # sort by translation_quality
    df = df.sort_values(by='translation_quality', ascending=False, ignore_index=True)

    threshold = sim_mean + 3*sim_std
    df['quality'] = 'Bad'
    df.loc[(df['sim_simple'] > threshold) & (df['sim_comparator'] > threshold) & (df['sim_backwards'] > threshold), 'quality'] = 'Good'
    df.loc[(df['sim_simple'] > threshold) & (df['sim_comparator'] < threshold) & (df['sim_backwards'] > threshold), 'quality'] = 'Good given vocab'

    # compute frequencies for each type and quality
    freqs = df.groupby(['type', 'quality']).size().unstack(fill_value=0)
    freqs = freqs.div(freqs.sum(axis=1), axis=0)

    # write summary to table
    freqs.to_csv(savedir / 'translation_quality_summary.csv')
    freqs.to_html(savedir / 'translation_quality_summary.html')

    # rename types
    df['type'] = df['type'].replace({
        'sv': 'subject-verb',
        'svo': 'subject-verb-object',
        '2v': 'two-verb',
        '2s': 'complex',
        '2c': 'two-clause',
    })

    df = df.melt(
        id_vars=['sentence', 'type', 'simple', 'comparator', 'target', 'backwards'],
        value_vars=sim_metrics,
        var_name='similarity_metric',
        value_name='similarity'
    )

    fig = px.scatter(
        df,
        x='sentence',
        y='similarity',
        color='similarity_metric',
        facet_col='type',
        facet_col_wrap=2,
        template='plotly_white',
        custom_data=['sentence', 'simple', 'comparator', 'target', 'backwards'],
    )
    fig.update_xaxes(matches=None)
    fig.for_each_xaxis(lambda yaxis: yaxis.update(showticklabels=True))
    fig.update_traces(marker=dict(size=10, opacity=0.5), mode="markers", hovertemplate=None)
    fig.update_layout(
        autosize=False,
        width=1650,
        height=2500,
        hovermode="x unified",
    )
    def set_hoverdata(trace) -> str:
        if trace.legendgroup == 'sim_simple':            
            trace.update(hovertemplate="<br>" + "<br>".join([
                "simple: %{customdata[1]}",
                "similarity: %{y}",
            ]))
        elif trace.legendgroup == 'sim_comparator':
            trace.update(hovertemplate="<br>" + "<br>".join([
                "comparator: %{customdata[2]}",
                "similarity: %{y}",
            ]))
        elif trace.legendgroup == 'sim_backwards':
            trace.update(hovertemplate="<br>" + "<br>".join([
                "target: %{customdata[3]}",
                "backwards: %{customdata[4]}",
                "similarity: %{y}",
            ]))
    # add data to hover data
    fig.for_each_trace(set_hoverdata)
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(range=[-0.1, 1.1])

    fig.add_hrect(
        y0=sim_mean - sim_std, y1=sim_mean + sim_std,
        line_width=0, fillcolor='gray',
        opacity=0.5, layer='below',
        row='all', col='all'
    )
    fig.add_hrect(
        y0=sim_mean - 2*sim_std, y1=sim_mean + 2*sim_std,
        line_width=0, fillcolor='gray',
        opacity=0.3, layer='below',
        row='all', col='all'
    )
    fig.add_hrect(
        y0=sim_mean - 3*sim_std, y1=sim_mean + 3*sim_std,
        line_width=0, fillcolor='gray',
        opacity=0.1, layer='below',
        row='all', col='all'
    )
    fig.write_html(savedir / 'translation_quality.html')
    fig.write_image(savedir / 'translation_quality.pdf')

    # generate same plot for each type separately
    for sim_type in df['type'].unique():
        df_type = df[df['type'] == sim_type]
        fig = px.scatter(
            df_type,
            x='sentence',
            y='similarity',
            color='similarity_metric',
            facet_col='type',
            facet_col_wrap=2,
            template='plotly_white',
            custom_data=['sentence', 'simple', 'comparator', 'target', 'backwards'],
        )
        fig.update_xaxes(matches=None)
        fig.for_each_xaxis(lambda yaxis: yaxis.update(showticklabels=True))
        fig.update_traces(marker=dict(size=10, opacity=0.5), mode="markers", hovertemplate=None)
        fig.update_layout(
            # autosize=False,
            # width=1650,
            # height=2500,
            hovermode="x unified",
        )
        def set_hoverdata(trace) -> str:
            if trace.legendgroup == 'sim_simple':            
                trace.update(hovertemplate="<br>" + "<br>".join([
                    "simple: %{customdata[1]}",
                    "similarity: %{y}",
                ]))
            elif trace.legendgroup == 'sim_comparator':
                trace.update(hovertemplate="<br>" + "<br>".join([
                    "comparator: %{customdata[2]}",
                    "similarity: %{y}",
                ]))
            elif trace.legendgroup == 'sim_backwards':
                trace.update(hovertemplate="<br>" + "<br>".join([
                    "target: %{customdata[3]}",
                    "backwards: %{customdata[4]}",
                    "similarity: %{y}",
                ]))

        # add data to hover data
        fig.for_each_trace(set_hoverdata)
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(range=[0, 1.05])

        fig.add_hrect(
            y0=sim_mean - sim_std, y1=sim_mean + sim_std,
            line_width=0, fillcolor='gray',
            opacity=0.5, layer='below',
            row='all', col='all'
        )
        fig.add_hrect(
            y0=sim_mean - 2*sim_std, y1=sim_mean + 2*sim_std,
            line_width=0, fillcolor='gray',
            opacity=0.3, layer='below',
            row='all', col='all'
        )
        fig.add_hrect(
            y0=sim_mean - 3*sim_std, y1=sim_mean + 3*sim_std,
            line_width=0, fillcolor='gray',
            opacity=0.1, layer='below',
            row='all', col='all'
        )

        fig.write_html(savedir / f'translation_quality_{sim_type}.html')
        # make figure narrower
        fig.update_layout(
            autosize=False,
            width=1000,
            height=500,
        )
        fig.write_image(savedir / f'translation_quality_{sim_type}.pdf')


if __name__ == '__main__':
    main()

