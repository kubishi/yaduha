import openai
import json
import os

def get_openai_client():
    try:
        return openai.Client(api_key=os.environ['OPENAI_API_KEY'])
    except KeyError:
        raise ValueError("OPENAI_API_KEY environment variable not set")

def parse_sentence(sentence):
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "".join([
                    "You are an assistant that splits user input sentences into a set of simple SVO or SV sentences. ",
                    "You must return your response strictly as a JSON object. ",
                    "Do not include any extra text outside the JSON response. ",
                    "The JSON must follow this structure: ",
                    "{'subject': str, 'verb': str, 'verb_tense': str, 'object': str or None}. ",
                    "Example: {'subject': 'I', 'verb': 'sit', 'verb_tense': 'present_continuous', 'object': None}"
                ])
            },
            {"role": "user", "content": sentence}
        ]
    )

    return response.choices[0].message.content

# Example usage:
parsed_output = parse_sentence("I am sitting in a chair.")
print(parsed_output)