import json
import pathlib
import random
import string
import argparse

from .functions import search_english, search_grammar, search_paiute, search_sentences



thisdir = pathlib.Path(__file__).parent.absolute()

characters = string.ascii_letters + string.digits
def get_random_tool_call_id():
    """Generate a random tool call id of the form call_aSENunZCF31ob7zV89clvL4n"""
    return "call_" + ''.join(random.choices(characters, k=24))

functions = {
    "search_english": search_english,
    "search_grammar": search_grammar,
    "search_paiute": search_paiute,
    "search_sentences": search_sentences
}

def main():
    parser = argparse.ArgumentParser(description="Generate examples for the chatbot")
    parser.add_argument("--input", default="examples.json", help="Path to the examples file")

    args = parser.parse_args()

    path_input = pathlib.Path(args.input).resolve()
    if not path_input.exists():
        raise FileNotFoundError(f"Input file {path_input} does not exist.")
    
    path_hydrated = path_input.parent / f"{path_input.stem}_hydrated.json"
    path_output = path_input.parent / f"{path_input.stem}_messages.json"

    examples = json.loads(path_input.read_text())
    prev_responses = {}
    if path_hydrated.exists():
        prev_responses = {
            example["query"]: example["response"]
            for example in json.loads(path_hydrated.read_text())
        }
    for example in examples:
        print(f"Query: {example['query']}")
        for tool_call in example.get("tool_calls", []):
            function = functions.get(tool_call["function"])
            tool_call["result"] = function(**tool_call["arguments"])
            tool_call["id"] = get_random_tool_call_id()
            print(f"Function: {tool_call['function']}")
            print(f"Arguments: {tool_call['arguments']}")
            print(f"Result: {tool_call['result']}")
            print()

        if example["query"] in prev_responses:
            example["response"] = prev_responses[example["query"]]
        else:
            example["response"] = input("Enter the desired response: ")
            
        print(f"Response: {example['response']}")
        print()

    path_hydrated.write_text(json.dumps(examples, indent=2, ensure_ascii=False))

    # format as messages
    messages = []
    for example in examples:
        messages.append({
            "role": "user",
            "content": example["query"]
        })
        tool_calls = [
            {
                "id": tool_call["id"],
                "function": {
                    "name": tool_call["function"],
                    "arguments": json.dumps(tool_call["arguments"], ensure_ascii=False)
                },
                "type": "function"
            }
            for tool_call in example.get("tool_calls", [])
        ]
        messages.append({
            "role": "assistant",
            "tool_calls": tool_calls
        })
        # add responses
        for tool_call in example.get("tool_calls", []):
            messages.append({
                "role": "tool",
                "content": json.dumps(tool_call["result"], ensure_ascii=False),
                "tool_call_id": tool_call["id"]
            })
        
        messages.append({
            "role": "assistant",
            "content": example["response"]
        })

    path_output.write_text(json.dumps(messages, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()