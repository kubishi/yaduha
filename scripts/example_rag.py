from yaduha.forward.rag_translator import RAGTranslator


def main():
    translator = RAGTranslator(model='gpt-4o-mini')
    sentences = [
        "The dog fell.",
        "That horse heard the runner.",
        "The frog is standing by the door."
    ]
    for sentence in sentences:
        translation = translator.translate(sentence)
        print(translation)

if __name__ == '__main__':
    main()