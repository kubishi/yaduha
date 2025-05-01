from yaduha.forward.rag_translator import RAGTranslator


def main():
    translator = RAGTranslator(model='gpt-4o')
    sentences = [
        "The dog fell.",
        # "That horse heard the runner.",
        "The frog is standing by the door.",
        # "That guy who is standing there is looking at me.",
        "My dog ate my toast while I was running."
    ]
    for sentence in sentences:
        translation = translator.translate(sentence)
        print(translation)
        # print(translation.metadata)


if __name__ == '__main__':
    main()