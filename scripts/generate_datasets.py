import logging
import pathlib
import pandas as pd
import random

from yaduha.sentence_builder import NOUNS, format_sentence, get_random_sentence, get_random_simple_sentence, sentence_to_str
from yaduha.translate_ovp2eng import translate as translate_ovp2eng

thisdir = pathlib.Path(__file__).parent.absolute()
datadir = thisdir / "data"

uncommon_nouns = pd.read_csv(datadir / "uncommon_nouns.csv")
uncommon_verbs = pd.read_csv(datadir / "uncommon_verbs.csv")

def random_good_translations(savepath: pathlib.Path,
                             n: int = 100,
                             overwrite: bool = False,
                             replace_subject_noun: bool = False,
                             replace_object_noun: bool = False,
                             replace_verb: bool = False):
    _get_random_sentence = get_random_sentence
    if replace_object_noun or replace_subject_noun or replace_verb:
        _get_random_sentence = get_random_simple_sentence

    df = pd.DataFrame(columns=["ovp", "eng"])
    if savepath.exists():
        if not overwrite:
            df = pd.read_csv(savepath)
        else:
            logging.info(f"random_good_translations: Overwriting {savepath}")

    if len(df) >= n:
        logging.info(f"random_good_translations: Already have at least {n} sentences.")
        return df
    
    rows = []
    for i in range(n - len(df)):
        choices = _get_random_sentence()
        word_choices = {k: v['value'] for k, v in choices.items()}

        if replace_subject_noun and word_choices["subject_noun"] in NOUNS:
            word_choices["subject_noun"] = f'[{random.choice(uncommon_nouns["noun"])}]'
        if replace_object_noun and word_choices["object_noun"] in NOUNS:
            word_choices["object_noun"] = f'[{random.choice(uncommon_nouns["noun"])}]'
        if replace_verb:
            word_choices["verb"] = f'[{random.choice(uncommon_verbs["verb"])}]'

        eng = translate_ovp2eng(**word_choices)
        ovp = sentence_to_str(format_sentence(**word_choices))


        if eng is not None:
            rows.append({"ovp": ovp, "eng": eng})
            logging.info(f"random_good_translations: {len(df)} / {n}")
        else:
            logging.info(f"random_good_translations: Failed to translate {ovp}")


        _df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        _df.to_csv(savepath, index=False)

def main():
    overwrite = False
    random_good_translations(
        datadir / "random_good_translations.csv",
        overwrite=overwrite
    )
    random_good_translations(
        datadir / "random_no_subject_noun.csv",
        n=25,
        overwrite=overwrite,
        replace_subject_noun=True
    )
    random_good_translations(
        datadir / "random_no_object_noun.csv",
        n=25,
        overwrite=overwrite,
        replace_object_noun=True
    )
    random_good_translations(
        datadir / "random_no_verb.csv",
        n=25,
        overwrite=overwrite,
        replace_verb=True
    )
    random_good_translations(
        datadir / "random_no_nouns.csv",
        n=25,
        overwrite=overwrite,
        replace_subject_noun=True,
        replace_object_noun=True
    )
    random_good_translations(
        datadir / "random_no_vocab.csv",
        n=25,
        overwrite=overwrite,
        replace_subject_noun=True,
        replace_object_noun=True,
        replace_verb=True
    )
    print(uncommon_nouns)


if __name__ == "__main__":
    main()
