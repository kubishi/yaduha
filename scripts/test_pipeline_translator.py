from yaduha.language.zephyrian import SubjectVerbObjectSentence, SubjectVerbSentence
from yaduha.agent.openai import OpenAIAgent
from yaduha.translator.pipeline import PipelineTranslator

import os
from dotenv import load_dotenv

load_dotenv()

def main():
    agent = OpenAIAgent(
        model="gpt-4o",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    translator = PipelineTranslator(
        agent=agent,
        SentenceType=(SubjectVerbObjectSentence, SubjectVerbSentence),
    )

    translation = translator.translate(
        "The dog is sitting at the lakeside, drinking some water."
    )
    print(f"Translation: {translation}")

if __name__ == "__main__":
    main()