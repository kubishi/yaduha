import json
from typing import Dict
import pandas
import pathlib
import dotenv

from yaduha.base import Translation, Translator
from yaduha.forward import AgenticTranslator, PipelineTranslator

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.resolve()
savepath = thisdir / 'results/evaluation_results.csv'

def run(overwrite: bool = False):
    if savepath.exists() and not overwrite:
        return
    translators: Dict[str, Translator] = {
        'pipeline': PipelineTranslator(model='gpt-4o-mini'),
        'agentic': AgenticTranslator(model='gpt-4o-mini')
    }

    df = pandas.read_csv(thisdir / 'data/evaluation_sentences.csv')
    # get first 3 sentences of each type
    sentences = df.groupby('type').head(3)['sentence'].tolist()

    rows = []
    print('Translating sentences...')
    for i, (translator_name, translator) in enumerate(translators.items(), start=1):
        print(f'[TRANSLATOR] {i}/{len(translators)}')
        for j, sentence in enumerate(sentences, start=1):
            print(f'  [SENTENCE] {j}/{len(sentences)}', end='\r')
            translation = translator.translate(sentence)
            rows.append({
                'translator': translator_name,
                'sentence': sentence,
                'sentence_type': df[df['sentence'] == sentence]['type'].values[0],
                'model_calls': translation.metadata['model_calls'],
                'back_model_calls': translation.metadata['back_model_calls'],
            })
            df_results = pandas.DataFrame(rows)
            df_results.to_csv(savepath, index=False)
        print(" "*100, end='\r')
    print('Done.')


def analyze():
    df = pandas.read_csv(savepath)
    print(df)

    # get stats on model call by translator and sentence type
    stats = df.groupby(['translator', 'sentence_type']).agg({
        'model_calls': ['mean', 'std'],
        'back_model_calls': ['mean', 'std'],
    })

    print(stats)
    

def main():
    run()
    analyze()

if __name__ == '__main__':
    main()