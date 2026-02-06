import requests
from yaduha.logger import JsonLogger, WandbLogger, PrintLogger
from yaduha.translator.pipeline import PipelineTranslator
from yaduha.translator.agentic import AgenticTranslator
from yaduha.agent.openai import OpenAIAgent
from yaduha.agent.anthropic import AnthropicAgent
from yaduha.agent.ollama import OllamaAgent
from yaduha.language.ovp import SubjectVerbSentence, SubjectVerbObjectSentence
from yaduha.language.ovp.prompts import get_prompt
from yaduha.tool import Tool

from typing import ClassVar, List, Dict, Tuple
from dotenv import load_dotenv
import os


load_dotenv()


def format_word(response_words: List) -> List:
    """
    Format the word response from the API into a string.
    """
    filtered = [
        {
            "paiute_translation": item["lexical_unit"],
            "english": item["senses"][0]["gloss"],
            "definition": item["senses"][0].get( "definition")
        } for item in response_words
    ]

    return filtered

class SearchEnglishTool(Tool):
    name: ClassVar[str] = "search_english"
    description: ClassVar[str] = "Search in English for Paiute words/translations."
    KUBISHI_API_URL: ClassVar[str] = "https://dictionary.kubishi.com/api"

    def _run(self, query: str, limit: int = 3) -> List[Dict]:
        print(f"Searching for English word: {query}")
        response = requests.get(f"{SearchEnglishTool.KUBISHI_API_URL}/search/english", params={"query": query, "limit": limit})
        response.raise_for_status()
        res_json: List[Dict] = response.json()

        results = [
            {
                "paiute_translation": item.get("lexical_unit"),
                "english": (item.get("senses") or [{}])[0].get("gloss"),
                "definition": (item.get("senses") or [{}])[0].get("definition")
            } for item in res_json
        ]

        self.log({
            "tool/search_english/query": query,
            "tool/search_english/results": results
        })

        return results
        
class SearchPaiuteTool(Tool):
    name: ClassVar[str] = "search_paiute"
    description: ClassVar[str] = "Search in Paiute for words/translations."
    KUBISHI_API_URL: ClassVar[str] = "https://dictionary.kubishi.com/api"

    def _run(self, query: str, limit: int = 3) -> List[Dict]:
        print(f"Searching for Paiute word: {query}")
        response = requests.get(f"{SearchPaiuteTool.KUBISHI_API_URL}/search/paiute", params={"query": query, "limit": limit})
        response.raise_for_status()
        res_json: List[Dict] = response.json()

        
        return format_word(res_json)

class SearchSentencesTool(Tool):
    name: ClassVar[str] = "search_sentences"
    description: ClassVar[str] = "Search in English (semantic search) for example sentences/translations."
    KUBISHI_API_URL: ClassVar[str] = "https://dictionary.kubishi.com/api"
    
    def _run(self, query: str, limit: int = 3) -> List[Dict]:
        print(f"Searching for English sentence: {query}")
        response = requests.get(f"{SearchPaiuteTool.KUBISHI_API_URL}/search/sentence", params={"query": query, "limit": limit})
        response.raise_for_status()
        res_json = response.json()
        infos = []
        for sentence in res_json:
            infos.append({
                "sentence": sentence["sentence"],
                "translation": sentence["translation"]
            })
        
        self.log({
            "tool/search_sentences/query": query,
            "tool/search_sentences/results": infos
        })
        return infos

def main():
    # logger = WandbLogger(
    #     project="kubishi",
    #     name="test-agentic-translator",
    # )
    logger = JsonLogger(filename="test-5")

    agent = OpenAIAgent(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        logger=logger
    )
    # agent = AnthropicAgent(
    #     model="claude-sonnet-4-5",
    #     api_key=os.environ["ANTHROPIC_API_KEY"]
    # )
    # agent = OllamaAgent(
    #     model="llama3.1:8b",
    #     logger=logger
    # )
    
    translator = AgenticTranslator(
        agent=agent,
        system_prompt=get_prompt(
            include_vocab=True,
            has_tools=True,
            include_examples=[SubjectVerbObjectSentence, SubjectVerbSentence]
        ),
        tools=[
            SearchEnglishTool(),
            SearchPaiuteTool(),
            SearchSentencesTool(),
            PipelineTranslator(
                agent=agent,
                SentenceType=(SubjectVerbObjectSentence, SubjectVerbSentence)
            )
        ],
        logger=logger
    )

    translation = translator("I am going to the store.")
    print(translation)

    # logger.stop()

if __name__ == "__main__":
    main()

