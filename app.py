from typing import Dict
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

from main import get_all_choices, format_sentence
from translate import translate

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/choices', methods=['POST'])
def get_choices():
    data: Dict = request.get_json()
    # subject_suffix: Optional[str],
    #                 verb: Optional[str],
    #                 verb_tense: Optional[str],
    #                 object_pronoun: Optional[str],
    #                 object_noun: Optional[str],
    #                 object_suffix: Optional[str]) -> Dict[str, Any]:
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
    print(sentence)

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
        sentence = format_sentence(
            subject_noun=data.get('subject_noun') or None,
            subject_suffix=data.get('subject_suffix') or None,
            verb=data.get('verb') or None,
            verb_tense=data.get('verb_tense') or None,
            object_pronoun=data.get('object_pronoun') or None,
            object_noun=data.get('object_noun') or None,
            object_suffix=data.get('object_suffix') or None,
        )
        translation = translate(sentence)
        return jsonify(translation=translation)
    except Exception as e:
        return jsonify(sentence=[], error=str(e))

if __name__ == '__main__':
    app.run(debug=True)