from typing import List
import argparse
import os
import pathlib
import json
import traceback
import pandas as pd
from yaduha.segment import split_sentence, make_sentence
from yaduha.forward.pipeline import translate_simple, order_sentence, comparator_sentence, semantic_similarity, PipelineTranslator
from yaduha.back_translate import translate as translate_ovp_to_english

thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    parser = argparse.ArgumentParser(description="Translate English to Paiute")
    subparsers = parser.add_subparsers(dest='command', required=True)

    translate_parser = subparsers.add_parser('translate', help="Translates sentences from English to Paiute")
    translate_parser.add_argument('sentence', help="The English sentence to translate (if not provided, will enter interactive mode)", nargs='?')
    translate_parser.add_argument('--model', help="The model to use for translation", default='gpt-4o-mini')
    translate_parser.set_defaults(func="translate")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    elif args.command == 'translate':
        translator = PipelineTranslator(model=args.model)
        translation = translator.translate(args.sentence)
        print(translation)

if __name__ == '__main__':
    main()
