from yaduha.agentic import AgenticTranslator


def main():
    translator = AgenticTranslator(model="gpt-4o-mini")
    print(translator.translate("The dog is running."))

if __name__ == "__main__":
    main()