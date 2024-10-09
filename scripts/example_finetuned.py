from yaduha.forward import FinetunedTranslator

def main():
    model="ft:gpt-4o-mini-2024-07-18:kubishi:brackets-plus-prompt-merged:AFvrmkic"
    translator = FinetunedTranslator(model=model)
    translation = translator.translate("The dog is eating a bone.")
    print(translation)

if __name__ == "__main__":
    main()