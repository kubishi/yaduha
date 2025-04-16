from typing import Dict, List
import openai
import os
import dotenv
import json
import pathlib

dotenv.load_dotenv()

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
KUBISHI_API_URL = "https://dictionary.kubishi.com/api"

thisdir = pathlib.Path(__file__).parent.absolute()

#TOOLS -------------------------------------
tools = [
    #search English ----------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "search_english",
            "description": "Search for Paiute words in English (semantic search).",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": [
                    "query"
                ],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search term, either a word or a sentence."
                    },
                },
                "additionalProperties": False
            }
        }
    },
    #Search Paiute ------------------------------
    {
        "type": "function",
        "function": {
            "name": "search_paiute",
            "description": "Search for English words in Paiute (semantic search).",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": [
                    "query"
                ],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search term, either a word or a sentence.",
                    },
                },
                "additionalProperties": False,
            },
        }
    },

    #Search Sentences ----------------------------
    {
        "type": "function",
        "function": {
            "name": "search_sentences",
            "description": "Search for sentences in English (semantic search).",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": [
                    "query"
                ],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search term, either a word or a sentence."
                    },
                },
                "additionalProperties": False
            }
        }
    },
    #search Grammar -----------------------------------------------
    {
        "type": "function",
        "function": {
            
            "name": "search_grammar",
            "description": "Given a sentence in Paiute, please analyze the grammar and sentence structure using the text provided, then rewrite the sentence to be accurate to the text.",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": [
                    "query"
                ],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search term, either a word or a sentence."
                    },
                },
                "additionalProperties": False
            }
        }   
    }
]

system_prompt = (
    "You are a friendly and knowledgeable chatbot translator specialized in Owen's Valley Paiute, a low-resource language. "
    "The user is a beginner eager to learn Paiute. You have access to several tools:\n\n"
    "- **Translation:** Translate words and phrases between English and Paiute.\n" 
    "- **Semantic Search:** Given an English sentence, retrieve similar Paiute translations to provide context and improve accuracy.\n"
    "- **Grammar Analysis:** Use your search_grammar function to analyze and correct grammar and punctuation in Paiute sentences.\n\n"
    "Always aim to provide clear, context-aware translations and helpful explanations. "
    "If the user asks about Paiute or related language topics, respond efficiently and accurately. " 
    "If the user asks to translate something make sure to state that you are making a guess unless you are 100% it's a perfect transation "
    "(e.g., because it comes from an exact match example sentence)"
)

system_prompt_b = (
    "You are a user who is eager to learn about a new language known as Owen's Valley Paiute. "
    "You have access to a chatbot that is able to answer any questions you may have about the language."
    "this can include translating english to paiute, translating paiute to english, and any grammar questions about the language."
    "Make sure your questions always revolve around the language and not any other topic."
    "Make sure you always ask questions, DON'T respond with thank you or you're welcome."
)

system_prompt_translator = (
    "You are a language translator for the language known as Owen's Valley Paiute. "
    "You will answer a direct translation for a sentence provided by the user."
    "You are similar to a google translator or another translation service. Always respond strictly in Paiute without any extra text or explanations."
)

system_prompt_translator_b = (
    "You are a user who wants to know how to say sentences in Paiute. "
    "You have access to a translator that is able to translate any basic english sentence"
    "Make sure your english sentence prompt is as basic as 3rd grade sentences"
    "Make sure you always respond in english sentences. NOTHING else"
    "Don't repeat sentences you already translated"
)
messages_b: List[Dict] = json.loads((thisdir / "example_messages_b.json").read_text())
messages_b.insert(0, {"role": "system", "content": system_prompt_b})

messages: List[Dict] = json.loads((thisdir / "example_messages.json").read_text())
messages.insert(0, {"role": "system", "content": system_prompt})

messages_translator: List[Dict] = json.loads((thisdir / "example_messages_translator.json").read_text())
messages_translator.insert(0, {"role": "system", "content": system_prompt_translator})

messages_translator_b: List[Dict] = json.loads((thisdir / "example_messages_translator_b.json").read_text())
messages_translator_b.insert(0, {"role": "system", "content": system_prompt_translator_b})
