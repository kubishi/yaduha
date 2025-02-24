import openai
import json
import os
import argparse

def get_openai_client():
    try:
        return openai.Client(api_key=os.environ['OPENAI_API_KEY'])
    except KeyError:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
sentence_schema = {
    "type": "object",
    "properties": {
        "subject": {
            "type": "string",
            "description": "The subject of the sentence. Must be a single word and singular (not plural)."
        },
        "subject_possessive_determiner": {
            "type": "string",
            "description": "The possessive determiner (if any) modifying the subject. Example: 'my', 'your', 'his', 'her', 'its', 'our', 'their'.",
            "enum": ["my", "your", "his", "her", "its", "our", "their"]
        },
        "verb": {
            "type": "string",
            "description": "The present-tense verb of the sentence. Must be a single word (infinitive without 'to')."
        },
        "verb_tense": {
            "type": "string",
            "description": "The tense of the verb. Must be one of: past, present, future, past_continuous, present_continuous.",
            "enum": ["past", "present", "future", "past_continuous", "present_continuous"]
        },
        "object": {
            "type": "string",
            "description": "The object of the sentence (optional). Must be a single word and singular (not plural)."
        },
        "object_possessive_determiner": {
            "type": "string",
            "description": "The possessive determiner (if any) modifying the object. Example: 'my', 'your', 'his', 'her', 'its', 'our', 'their'.",
            "enum": ["my", "your", "his", "her", "its", "our", "their"]
        }
    },
    "required": ["subject", "verb", "verb_tense"]
}

def parse_sentence(sentence, model):
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {'role': 'system', 'content': "".join([
                'You are an assistant that splits user input sentences into a set of simple SVO or SV sentences. ',
                'The response **must be a JSON object** and should follow this structure: ',
                '{"subject": str, "verb": str, "verb_tense": str, "object": str or None}. ',
                'No adjectives, adverbs, prepositions, or conjunctions should be added to the simple sentences. ',
                'Subjects and objects can be nominalized verbs (e.g., "run" -> "the runner", "the one who ran"). ',
                'Nominalizer_tense must be "present" or "future" if applicable, otherwise None. ',
                'Subjects and objects may be modified by possessive determiners (e.g., "my house", "their dog"). ',
                'Possessive determiners should be tagged separately as "subject_possessive_determiner" and "object_possessive_determiner".'
            ])},
            {'role': 'user', 'content': 'I am sitting in a chair.'},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "arguments": json.dumps({
                        'sentences': [
                            {'subject': 'I', 'verb': 'sit', 'verb_tense': 'present_continuous', 'object': None},
                        ]
                    }),
                    "name": "set_sentences"
                },
            },
            {'role': 'user', 'content': 'The one who ran is sitting.'},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "arguments": json.dumps({
                        'sentences': [
                            {'subject': 'run', 'subject_nominalizer': 'past', 'verb': 'sit', 'verb_tense': 'present_continuous', 'object': None},
                        ]
                    }),
                    "name": "set_sentences"
                },
            },
            {'role': 'user', 'content': 'The dogs were chasing their tails.'},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "arguments": json.dumps({
                        'sentences': [
                            {'subject': 'dog', 'verb': 'chase', 'verb_tense': 'past_continuous', 'object': 'tail', 'object_possessive_determiner': 'their'},
                        ]
                    }),
                    "name": "set_sentences"
                },
            },
            {'role': 'user', 'content': 'The drinker is eating.'},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "arguments": json.dumps({
                        'sentences': [
                            {'subject': 'drink', 'subject_nominalizer': 'present', 'verb': 'eat', 'verb_tense': 'present', 'object': None},
                        ]
                    }),
                    "name": "set_sentences"
                },
            },
            {'role': 'user', 'content': 'The book sits on the table.'},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "arguments": json.dumps({
                        'sentences': [
                            {'subject': 'book', 'verb': 'sit', 'verb_tense': 'present', 'object': None},
                        ]
                    }),
                    "name": "set_sentences"
                },
            },
            {'role': 'user', 'content': 'The boy talked about the girl.'},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "arguments": json.dumps({
                        'sentences': [
                            {'subject': 'boy', 'verb': 'talk', 'verb_tense': 'past', 'object': None},
                        ]
                    }),
                    "name": "set_sentences"
                },
            },
            {'role': 'user', 'content': 'I saw two men walking their dogs yesterday at Starbucks while drinking a cup of coffee'},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "arguments": json.dumps({
                        'sentences': [
                            {'subject': 'I', 'verb': 'see', 'verb_tense': 'past', 'object': 'man'},
                            {'subject': 'man', 'verb': 'walk', 'verb_tense': 'past_continuous', 'object': 'dog'},
                            {'subject': 'man', 'verb': 'drink', 'verb_tense': 'past_continuous', 'object': 'coffee'}
                        ]
                    }),
                    "name": "set_sentences"
                },
            },
            {'role': 'user', 'content': 'That runner down the street will eat the one that fell.'},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "arguments": json.dumps({
                        'sentences': [
                            {'subject': 'run', 'subject_nominalizer': 'present', 'verb': 'eat', 'verb_tense': 'future', 'object': 'fall', 'object_nominalizer': 'past'},
                        ]
                    }),
                    "name": "set_sentences"
                },
            },
            {'role': 'user', 'content': 'I saw my mom.'},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "arguments": json.dumps({
                        'sentences': [
                            {'subject': 'I', 'verb': 'see', 'verb_tense': 'past', 'object': 'mom', 'object_possessive_determiner': 'my'}
                        ]
                    }),
                    "name": "set_sentences"
                },
            },
            {'role': 'user', 'content': 'My dog chased her tail.'},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "arguments": json.dumps({
                        'sentences': [
                            {'subject': 'dog', 'subject_possessive_determiner': 'my', 'verb': 'chased', 'verb_tense': 'past', 'object': 'tail', 'object_possessive_determiner': 'her'}
                        ]
                    }),
                    "name": "set_sentences"
                },
            },
            {'role': 'user', 'content': 'Their family will go to the park next week.'},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "arguments": json.dumps({
                        'sentences': [
                            {'subject': 'family', 'subject_possessive_determiner': 'their', 'verb': 'go', 'verb_tense': 'future', 'object': 'park', 'object_determiner': 'the', 'time': 'next week'}
                        ]
                    }),
                    "name": "set_sentences"
                },
            },
            {"role": "user", "content": sentence}
        ]
    )

    return response.choices[0].message.content

if __name__ == '__main__':
    # Allow user to pass in example sentence to be segmented, or use default.
    parser = argparse.ArgumentParser()
    parser.add_argument("user_input", nargs="?", default="The dog fell yesterday", help="The input string to parse")
    args = parser.parse_args()
    sentence = args.user_input
    simple_sentence = parse_sentence(sentence, model=os.environ['OPENAI_MODEL'])    
    print(simple_sentence)