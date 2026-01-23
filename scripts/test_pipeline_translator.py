"""Test script for the PipelineTranslator.

Tests all models and prints results in a pandas table.
"""

from yaduha.translator.pipeline import PipelineTranslator
from yaduha.agent.openai import OpenAIAgent
from yaduha.agent.anthropic import AnthropicAgent
from yaduha.agent.ollama import OllamaAgent
from yaduha.language.ovp import SubjectVerbSentence, SubjectVerbObjectSentence

import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

# Models to test
MODELS = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "anthropic": [
        "claude-sonnet-4-5",
        "claude-sonnet-4-20250514",
        "claude-3-haiku-20240307",
    ],
    "ollama": [
        "deepseek-r1:70b",
        "mixtral:8x22b",
        "llama3.1:8b",
    ],
}


def create_agent(agent_type: str, model: str):
    """Create an agent of the specified type."""
    if agent_type == "openai":
        return OpenAIAgent(
            model=model,  # type: ignore[arg-type]
            api_key=os.environ["OPENAI_API_KEY"],
        )
    elif agent_type == "anthropic":
        return AnthropicAgent(
            model=model,  # type: ignore[arg-type]
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    elif agent_type == "ollama":
        return OllamaAgent(model=model)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main():
    # ============================================================
    # CONFIGURATION
    # ============================================================

    source = "The dog is sitting at the lakeside, drinking some water."

    # Back-translation always uses gpt-4o
    back_agent = OpenAIAgent(
        model="gpt-4o",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # ============================================================

    print(f"Source: {source}")
    print(f"Back-translation: openai/gpt-4o")
    print("=" * 80)

    results = []

    for agent_type, models in MODELS.items():
        for model in models:
            print(f"Testing {agent_type}/{model}...")

            try:
                agent = create_agent(agent_type, model)

                translator = PipelineTranslator(
                    agent=agent,
                    back_translation_agent=back_agent,
                    SentenceType=(SubjectVerbObjectSentence, SubjectVerbSentence),
                )

                translation = translator(source)
                back_translation = (
                    translation.back_translation.source
                    if translation.back_translation
                    else "N/A"
                )

                results.append({
                    "agent": agent_type,
                    "model": model,
                    "translation": translation.target,
                    "back_translation": back_translation,
                })
                print(f"  OK")

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "agent": agent_type,
                    "model": model,
                    "translation": f"ERROR",
                    "back_translation": "N/A",
                })

    # Print results as a table
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80 + "\n")

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
