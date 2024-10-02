import json
import openai
import dotenv
import os

dotenv.load_dotenv()

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

# subject_noun: Optional[str] = None,
# subject_noun_nominalizer: Optional[str] = None,
# subject_suffix: Optional[str] = None,
# verb: Optional[str] = None,
# verb_tense: Optional[str] = None,
# object_pronoun: Optional[str] = None,
# object_noun: Optional[str] = None,
# object_noun_nominalizer: Optional[str] = None,
# object_suffix: Optional[str] = None) -> Dict[str, Any]:
def main():
    messages = [
        {
            "role": "user",
            "content": "The cat is eating an apple."
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "subject_noun": "cat",
                "subject_noun_nominalizer": None,

            })
        }
    ]

if __name__ == "__main__":
    main()