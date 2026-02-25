"""Test script for the PipelineTranslator.

Tests all models and prints results in a pandas table.
"""

from yaduha.logger import JsonLogger, Logger, WandbLogger
from yaduha.translator.pipeline import PipelineTranslator
from yaduha.agent.openai import OpenAIAgent
from yaduha.agent.anthropic import AnthropicAgent
from yaduha.agent.ollama import OllamaAgent
from yaduha.language.ovp import SubjectVerbSentence, SubjectVerbObjectSentence

import pandas as pd
from dotenv import load_dotenv
import weave
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


def create_agent(agent_type: str, model: str, logger: Logger):
    """Create an agent of the specified type."""
    if agent_type == "openai":
        return OpenAIAgent(
            model=model,  # type: ignore[arg-type]
            api_key=os.environ["OPENAI_API_KEY"],
            logger=logger.get_sublogger(functionality="translation")
        )
    elif agent_type == "anthropic":
        return AnthropicAgent(
            model=model,  # type: ignore[arg-type]
            api_key=os.environ["ANTHROPIC_API_KEY"],
            logger=logger.get_sublogger(functionality="translation")
        )
    elif agent_type == "ollama":
        return OllamaAgent(
            model=model, 
            logger=logger.get_sublogger(functionality="translation")
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main():
    # ============================================================
    # CONFIGURATION
    # ============================================================

    source = "The cat is drinking some water, and walking around"

    # logger = WandbLogger(name="Pipline Test - ALL - 2", project_name="Pipeline Test", tags=["Ollama", "logger"], notes="Testing logger functionality")

    logger = JsonLogger(filename="test-1.3")


    # Back-translation always uses gpt-4o
    back_agent = OpenAIAgent(
        model="gpt-4o",
        api_key=os.environ["OPENAI_API_KEY"],
        logger=logger.get_sublogger(functionality="back-agent")
    )
    
    # back_agent = OllamaAgent(model="llama3.1:8b", logger=logger)

    # ============================================================

    print(f"Source: {source}")
    print(f"Back-translation: openai/gpt-4o")
    print("=" * 80)

    results = []

    for agent_type, models in MODELS.items():
        for model in models:
            print(f"Testing {agent_type}/{model}...")

            try:
                agent = create_agent(agent_type, model, logger)

                translator = PipelineTranslator(
                    agent=agent,
                    back_translation_agent=back_agent,
                    SentenceType=(SubjectVerbObjectSentence, SubjectVerbSentence),
                    logger=logger.get_sublogger(functionality="pipeline")
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
