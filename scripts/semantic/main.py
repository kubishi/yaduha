from yaduha.translate.semantic import SemanticTranslator
import json
import pathlib

thisdir = pathlib.Path(__file__).parent.absolute()

def save_messages(messages):
    path = thisdir / "messages.json"
    path.write_text(json.dumps(messages, indent=2, ensure_ascii=False))

def main():
    translator = SemanticTranslator(model="gpt-4o-mini")
    translator.translate("The dog is running.", messages_callback=save_messages)

if __name__ == '__main__':
    main()