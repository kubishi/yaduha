import os
from typing import Dict
import pandas as pd

from main import get_random_sentence, sentence_to_str
import openai
import pathlib 
import dotenv

dotenv.load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
thisdir = pathlib.Path(__file__).parent.absolute()

def translate(sentence_details: Dict) -> str:
    sentence = sentence_to_str(sentence_details)
    
    df = pd.read_csv(thisdir.joinpath('sentences.csv'))
    messages = [{'role': 'system', 'content': 'You are a Paiute to English Translator'}]
    # iterate through the rows of the dataframe
    for index, row in df.iterrows():
        user_message = f"sentence={row['sentence']}\ndetails={row['details']}"
        bot_message = row['translation']
        messages.append({'role': 'user', 'content': user_message})
        messages.append({'role': 'assistant', 'content': bot_message})

    messages.append({'role': 'user', 'content': f"sentence={sentence}\ndetails={sentence_details}"})

    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=messages
    )
    translation = response['choices'][0]['message']['content']
    return translation

def main():
    sentence_details = get_random_sentence()
    print(f"Sentence: {sentence_to_str(sentence_details)}")
    translation = translate(sentence_details)
    print(f"Translation: {translation}")


if __name__ == '__main__':
    main()