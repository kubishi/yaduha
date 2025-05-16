import logging
import os
import json
import argparse

from yaduha.chatbot.tools.tools import tools, messages, messages_b, messages_translator, messages_translator_b
from yaduha.chatbot.tools.functions import search_english, search_grammar, search_paiute, search_sentences
from yaduha.common import get_openai_client

functions = {
    "search_english": search_english,
    "search_grammar": search_grammar,
    "search_paiute": search_paiute,
    "search_sentences": search_sentences
}

client = get_openai_client(api_key=os.getenv("OPENAI_API_KEY"))
welcome_art = r"""
                                    Welcome To
     ____   _    ___ _   _ _____ _____    ____ _   _    _  _____ ____   ___ _____ 
    |  _ \ / \  |_ _| | | |_   _| ____|  / ___| | | |  / \|_   _| __ ) / _ \_   _|
    | |_) / _ \  | || | | | | | |  _|   | |   | |_| | / _ \ | | |  _ \| | | || |  
    |  __/ ___ \ | || |_| | | | | |___  | |___|  _  |/ ___ \| | | |_) | |_| || |  
    |_| /_/   \_\___|\___/  |_| |_____|  \____|_| |_/_/   \_\_| |____/ \___/ |_|  
"""

def add_chatlog(message, chatbot):
    if chatbot:
        with open("messages/chatlog.txt", "a") as f:
            f.write(message)
    else:
        with open("messages/chatlog_translate.txt", "a") as f:
            f.write(message)

def get_parser():
    parser = argparse.ArgumentParser(description="Assistant for Owens Valley Paiute")
    parser.add_argument("--log-level", default="WARNING", help="Logging level")
    subparsers = parser.add_subparsers(dest="command")

    bot_parser = subparsers.add_parser("bot", help="Start the assistant chatbot")
    bot_b_parder = subparsers.add_parser("bot_b", help="Start the assistant chatbot_b")
    translator_parser = subparsers.add_parser("translate", help="Start the translator")
    translator_b_parser = subparsers.add_parser("translate_b", help="Start the translator")

    return parser

def run_chatbot(user: bool, model: str = "gpt-4o-mini"):
    print(welcome_art)

    if user:
        query_question = input("Assistant: Hello! What questions about Owens Valley Paiute can I help you with? \nYou: ")
    else:
        query_question = query_chatbot_b("Ask a question")

    add_chatlog("User question: " + query_question + "\n\n", chatbot=True)

    query = {
        "role": "user",
        "content": query_question
    }
    messages.append(query)

    while True:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=0.0,
        )

        messages.append(json.loads(completion.choices[0].message.model_dump_json()))

        if completion.choices[0].message.tool_calls:
            print("Searching for relevant words, sentences, and grammatical information...")
            for tool_call in completion.choices[0].message.tool_calls:
                kwargs = json.loads(tool_call.function.arguments)
                logging.info(f"Function: {tool_call.function.name}")
                logging.info(f"Arguments: {kwargs}")
                function = functions.get(tool_call.function.name)
                if not function:
                    logging.error(f"Function {tool_call.function.name} not found.")
                    continue
                res = function(**kwargs)
                logging.info(f"Result: {res}")
                messages.append({
                    "role": "tool",
                    "content": json.dumps(res, ensure_ascii=False),
                    "tool_call_id": tool_call.id
                })

                add_chatlog(
                    "Function: " + str(tool_call.function.name) + "\n"
                    "Arguments: " + str(kwargs) + "\n"
                    "Result: " + str(res) + "\n\n",
                    chatbot=True
                )

        else:
            assistant_response = completion.choices[0].message.content
            print(f"Assistant: {assistant_response}")
            add_chatlog("Assistant response: " + assistant_response + "\n\n", chatbot=True)

            if user:
                user_response = input("You: ")
            else:
                user_response = query_chatbot_b("You: ")

            messages.append({"role": "user", "content": user_response})
            add_chatlog("User response: " + user_response + "\n\n", chatbot=True)

def query_chatbot_b(message, model: str = "gpt-4o-mini"):
    query = {
        "role": "user",
        "content": message
    }
    messages_b.append(query)

    completion = client.chat.completions.create(
        model=model,
        messages=messages_b,
        temperature=0.7,
    )

    print("Query: " + completion.choices[0].message.content)
    return completion.choices[0].message.content

def query_translator_b(message, model: str = "gpt-4o-mini"):
    query = {
        "role": "user",
        "content": message
    }
    messages_translator_b.append(query)

    completion = client.chat.completions.create(
        model=model,
        messages=messages_translator_b,
        temperature=1.0,
    )

    print("Query: " + completion.choices[0].message.content)

    return completion.choices[0].message.content

def run_translator(user, model: str = "gpt-4o-mini"):
    
    if user:
        query_question = input("Sentence: ")
    else:
        query_question = query_translator_b("Give me a prompt")

    add_chatlog("Query: " + query_question + "\n\n", chatbot=False)

    query = {
        "role": "user",
        "content": query_question
    }

    messages_translator.append(query)

    while True:
        completion = client.chat.completions.create(
            model=model,
            messages=messages_translator,
            tools=tools,
            temperature=0.0,
        )

        messages_translator.append(json.loads(completion.choices[0].message.model_dump_json()))

        if completion.choices[0].message.tool_calls:
            print("Searching for relevant words, sentences, and grammatical information...")
            for tool_call in completion.choices[0].message.tool_calls:
                kwargs = json.loads(tool_call.function.arguments)
                logging.info(f"Function: {tool_call.function.name}")
                logging.info(f"Arguments: {kwargs}")
                function = functions.get(tool_call.function.name)
                if not function:
                    logging.error(f"Function {tool_call.function.name} not found.")
                    continue
                res = function(**kwargs)
                logging.info(f"Result: {res}")
                messages_translator.append({
                    "role": "tool",
                    "content": json.dumps(res, ensure_ascii=False),
                    "tool_call_id": tool_call.id
                })

                add_chatlog(
                    "Function: " + str(tool_call.function.name) + "\n"
                    "Arguments: " + str(kwargs) + "\n"
                    "Result: " + str(res) + "\n\n", 
                    chatbot=False
                )

        else:
            assistant_response = completion.choices[0].message.content
            print(f"Paiute: {assistant_response}")
            add_chatlog("Paiute: " + assistant_response + "\n\n", chatbot=False)
            
            if user:
                user_response = input("English: ")
            else:
                user_response = query_translator_b("English Sentence: ")

            messages_translator.append({"role": "user", "content": user_response})
            add_chatlog("User response: " + user_response + "\n\n", chatbot=False)

def translate(message: str, model: str = "gpt-4o-mini"):
    messages = [
        *messages_translator,
        {
            "role": "user",
            "content": message
        }
    ]

    translation = ""
    while True:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            tools=[
                tool for tool in tools if tool['function']['name'] == "search_english"	
            ],
        )

        messages.append(json.loads(completion.choices[0].message.model_dump_json()))

        if completion.choices[0].message.content:
            logging.info("Response: " + completion.choices[0].message.content)

        if not completion.choices[0].message.tool_calls:
            translation = completion.choices[0].message.content
            break
        
        for tool_call in completion.choices[0].message.tool_calls:
            kwargs = json.loads(tool_call.function.arguments)
            logging.info(f"Function: {tool_call.function.name}")
            # logging.info(f"Arguments: {kwargs}")
            function = functions.get(tool_call.function.name)
            if not function:
                logging.error(f"Function {tool_call.function.name} not found.")
                continue
            res = function(**kwargs)
            # logging.info(f"Result: {res}")
            messages.append({
                "role": "tool",
                "content": json.dumps(res, ensure_ascii=False),
                "tool_call_id": tool_call.id
            })

    response = {
        "translation_prompt_tokens": completion.usage.prompt_tokens,
        "translation_completion_tokens": completion.usage.completion_tokens,
        "translation_total_tokens": completion.usage.total_tokens,
        "translation": translation,
        "messages": messages_translator
    }
    return response

def main():
    # set log level
    parser = get_parser()
    args = parser.parse_args()
    log_level = args.log_level
    logging.basicConfig(level=log_level)


    if not hasattr(args, "command"):
        parser.print_help()
        return
    
    if args.command == "bot":
        run_chatbot(user=True)
    if args.command == "bot_b":
        run_chatbot(user=False)
    if args.command == "translate":
        run_translator(user=True)
    if args.command == "translate_b":
        run_translator(user=False)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()