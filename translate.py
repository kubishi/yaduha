import os
import pandas as pd

from main import get_random_sentence, sentence_to_str
import openai
import pathlib 
import dotenv

dotenv.load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    df = pd.read_csv(thisdir.joinpath('sentences.csv'))

    messages = [{'role': 'system', 'content': 'You are a Paiute to English Translator'}]
    # iterate through the rows of the dataframe
    for index, row in df.iterrows():
        user_message = f"sentence={row['sentence']}\ndetails={row['details']}"
        bot_message = row['translation']
        messages.append({'role': 'user', 'content': user_message})
        messages.append({'role': 'assistant', 'content': bot_message})

    details = get_random_sentence()
    sentence = sentence_to_str(details)
    messages.append({'role': 'user', 'content': f"sentence={sentence}\ndetails={details}"})

    print(f"Sentence: {sentence}")
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages
    )
    translation = response['choices'][0]['message']['content']
    print(f"Translation: {translation}")


if __name__ == '__main__':
    main()