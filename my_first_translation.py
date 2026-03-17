from yaduha.translator.pipeline import PipelineTranslator
from yaduha.translator.agentic import AgenticTranslator
from yaduha.agent.openai import OpenAIAgent
from yaduha.logger import JsonLogger
from yaduha_ovp import SubjectVerbSentence, SubjectVerbObjectSentence
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = JsonLogger(file_path="results/test_agentic-1.1.jsonl")

# Create an AI agent
agent = OpenAIAgent(
    model="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.0,  # For deterministic outputs
    logger=logger
)

# Create a translator
pipeline = PipelineTranslator(
    agent=agent,
    SentenceType=(SubjectVerbObjectSentence, SubjectVerbSentence),
    logger=logger
)

translator = AgenticTranslator(
    agent=agent,
    logger=logger,
    tools=[pipeline]
)

# Translate!
result = translator("The dog runs away.")

print(f"English: {result.source}")
print(f"Paiute: {result.target}")
# print(f"Back-translation: {result.back_translation.source}")
print(f"Time: {result.translation_time:.2f}s")
print(f"Tokens: {result.prompt_tokens + result.completion_tokens}")