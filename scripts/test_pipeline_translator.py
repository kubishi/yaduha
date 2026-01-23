from yaduha.translator.pipeline import PipelineTranslator
from yaduha.agent.openai import OpenAIAgent
from yaduha.agent.claude import ClaudeAgent
from yaduha.agent.ollama import OllamaAgent
from yaduha.language.ovp import SubjectVerbSentence, SubjectVerbObjectSentence

import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    agent = OllamaAgent(
        model="llama3.2:3b",
    )

    agents = {
        "openai": OpenAIAgent,
        "ollama": OllamaAgent,
    }
    models = {
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
        ],
        "ollama": [
            "llama3.1:8b",
            "llama3.2:3b",
            "qwen2.5:7b"
        ]
    }

    back_translation_agent = OpenAIAgent(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"]
    )

    source = "The dog is sitting at the lakeside, drinking some water."
    print(f"Source text: {source}\n")
    rows = []
    for agent_name, agent_class in agents.items():
        for model_name in models[agent_name]:
            # print(f"Testing {agent_name} model {model_name}...")
            agent = agent_class(
                model=model_name,
                api_key=os.environ.get(f"{agent_name.upper()}_API_KEY")
            )

            translator = PipelineTranslator(
                agent=agent,
                back_translation_agent=back_translation_agent,
                SentenceType=(SubjectVerbObjectSentence, SubjectVerbSentence)
            )

            translation = translator(source)
            back_translation = translation.back_translation.source if translation.back_translation else "N/A"
            # print(f"MODEL[{agent_name}:{model_name}]: {translation.target} -> {back_translation}")
            rows.append({
                "agent": agent_name,
                "model": model_name,
                "translation": translation.target,
                "back_translation": back_translation
            })

    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()

