import json
import time
import openai
from translate_eng2ovp import translate_english_to_ovp
import dotenv
import flask

dotenv.load_dotenv()

bp = flask.Blueprint('bot', __name__)

ASSISTANT_ID = "asst_D3k3AWyA22HCgtCZqYwjQWjO"

def main():
    client = openai.OpenAI()

    message = "How do you say 'I want to go to the big store and buy juicy apples' in Paiute?"
    run = client.beta.threads.create_and_run(
        assistant_id=ASSISTANT_ID,
        instructions=message
    )

    client.beta.threads.runs.list()
    while True:
        run = client.beta.threads.runs.retrieve(run_id=run.id, thread_id=run.thread_id)
        print(run.status)
        time.sleep(0.1)
        if run.status == 'requires_action':
            func = run.required_action.submit_tool_outputs.tool_calls[0].function
            if func.name == "translate":
                args = json.loads(func.arguments)
                translation = translate_english_to_ovp(args['sentence'])
                print(translation)
                translation['explanation'] = (
                    'The translator can only translate simple sentences, so it first breaks the input sentence into simple sentences ("simple") ' +
                    'and then translates them into Owens Valley Paiute ("target"). ' +
                    'You can tell the user that the translation may not be perfect, but it\'s (hopefully) a good starting point.'
                )
                if translation['sim_simple'] <= 0.8:
                    translation['warning'] = 'The input sentence is complex, so alot of meaning may have been lost in breaking it down into simple sentences.'
                elif translation['sim_backwards'] <= 0.8:
                    translation['warning'] = 'The translation doesn\'t seem to be very accurate.'
                elif translation['sim_comparator'] <= 0.8:
                    translation['warning'] = (
                        'Alot of the words are missing from the available vocabulary, so the translation leaves the english words as placeholders ' +
                        'and does it\'s best given the structures of sentences it knows how to build.'
                    )
                
                run = client.beta.threads.runs.submit_tool_outputs(
                    run_id=run.id,
                    thread_id=run.thread_id,
                    tool_outputs=[
                        {
                            "tool_call_id": run.required_action.submit_tool_outputs.tool_calls[0].id,
                            "output": json.dumps(translation)
                        }
                    ]
                )
        elif run.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=run.thread_id, order="asc")
            for message in messages.data:
                print(f"{message.role}: {message.content[0].text.value}")
            return

if __name__ == '__main__':
    main()

