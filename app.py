from typing import Dict
from flask import Flask, render_template, request, jsonify
from flask_talisman import Talisman
import os

app = Flask(__name__)

if os.getenv('FLASK_ENV') == 'production':
    Talisman(app, content_security_policy=None)

from main import get_all_choices, format_sentence, get_random_sentence, get_random_sentence_big
from translate import translate

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/choices', methods=['POST'])
def get_choices():
    data: Dict = request.get_json()
    choices = get_all_choices(
        subject_noun=data.get('subject_noun') or None,
        subject_suffix=data.get('subject_suffix') or None,
        verb=data.get('verb') or None,
        verb_tense=data.get('verb_tense') or None,
        object_pronoun=data.get('object_pronoun') or None,
        object_noun=data.get('object_noun') or None,
        object_suffix=data.get('object_suffix') or None,
    )
    
    sentence = []
    try:
        sentence = format_sentence(
            subject_noun=data.get('subject_noun') or None,
            subject_suffix=data.get('subject_suffix') or None,
            verb=data.get('verb') or None,
            verb_tense=data.get('verb_tense') or None,
            object_pronoun=data.get('object_pronoun') or None,
            object_noun=data.get('object_noun') or None,
            object_suffix=data.get('object_suffix') or None,
        )
    except Exception as e:
        print(e) 

    print(choices)

    # here you can call your functions and build the sentence
    return jsonify(choices=choices, sentence=sentence)

@app.route('/api/sentence', methods=['POST'])
def build_sentence():
    data: Dict = request.get_json()
    try:
        sentence = format_sentence(
            subject_noun=data.get('subject_noun') or None,
            subject_suffix=data.get('subject_suffix') or None,
            verb=data.get('verb') or None,
            verb_tense=data.get('verb_tense') or None,
            object_pronoun=data.get('object_pronoun') or None,
            object_noun=data.get('object_noun') or None,
            object_suffix=data.get('object_suffix') or None,
        )
        return jsonify(sentence=sentence)
    except Exception as e:
        return jsonify(sentence=[], error=str(e))
    

@app.route('/api/translate', methods=['POST'])
def get_translation():
    data: Dict = request.get_json()
    try:
        translation = translate(
            subject_noun=data.get('subject_noun') or None,
            subject_suffix=data.get('subject_suffix') or None,
            verb=data.get('verb') or None,
            verb_tense=data.get('verb_tense') or None,
            object_pronoun=data.get('object_pronoun') or None,
            object_noun=data.get('object_noun') or None,
            object_suffix=data.get('object_suffix') or None
        )
        return jsonify(translation=translation)
    except Exception as e:
        return jsonify(sentence=[], error=str(e))
    

# route to get random sentence
@app.route('/api/random', methods=['POST'])
def get_random():
    data: Dict = request.get_json() 
    try:
        choices = get_all_choices(
            subject_noun=data.get('subject_noun') or None,
            subject_suffix=data.get('subject_suffix') or None,
            verb=data.get('verb') or None,
            verb_tense=data.get('verb_tense') or None,
            object_pronoun=data.get('object_pronoun') or None,
            object_noun=data.get('object_noun') or None,
            object_suffix=data.get('object_suffix') or None,
        )

        choices = get_random_sentence(choices)
        sentence = format_sentence(**{k: v['value'] for k, v in choices.items()})
        return jsonify(choices=choices, sentence=sentence)
    except Exception as e:
        return jsonify(sentence=[], error=str(e))

@app.route('/api/random-example', methods=['GET'])
def get_random_example():
    try:
        choices = get_random_sentence_big()
        sentence = format_sentence(**{k: v['value'] for k, v in choices.items()})
        return jsonify(choices=choices, sentence=sentence)
    except Exception as e:
        return jsonify(sentence=[], error=str(e))

# health check
@app.route('/api/healthz', methods=['GET'])
def health_check():
    return jsonify(status='ok')

if __name__ == '__main__':
    app.run(debug=True)