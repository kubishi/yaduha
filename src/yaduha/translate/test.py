from yaduha.translate.full_translator import translate_simple_sentences, translate_sentence
from yaduha.translate.pipeline import split_sentence


def main():
    sentence = "Where is my dog?"

    sentence_translation = translate_sentence(sentence, model="gpt-4o-mini")
    print(sentence_translation["translation_prompt_tokens"])
    print(sentence_translation["translation"])


if __name__ == "__main__":
    main()