"""API routes for the app"""
import logging
import os
from typing import Dict, List

from openai import OpenAI, APIError
from translate_eng2ovp import translate_ovp_to_english, translate_english_to_ovp
from sentence_builder import get_all_choices, format_sentence, get_random_sentence, get_random_sentence_big

from flask import g, jsonify, make_response, request, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from app_base import app
from app_oauth import get_app_metadata

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=os.environ.get('REDIS_URL'),
    storage_options={"socket_connect_timeout": 30},
    strategy="fixed-window",
    default_limits=["10/second"], # to prevent abuse
)
LIMITS = {
    "translate": "1/second;30/minute;200/month",
}

TRANSLATION_QUALITY_THRESHOLD = 0.8

# API Routes
@app.errorhandler(429)
def ratelimit_handler(exp):
    message = 'Add your OpenAI API key in your account settings for unlimited access'
    return make_response(jsonify({"rate_limit_message": f"{message}."}), 429)

@app.route('/api/builder/choices', methods=['POST'])
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

    # here you can call your functions and build the sentence
    return jsonify(choices=choices, sentence=sentence)

@app.route('/api/builder/sentence', methods=['POST'])
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
        return jsonify(sentence=[], error=str(e)), 400

@app.route('/api/builder/translate', methods=['POST'])
@limiter.limit(LIMITS['translate'], key_func = lambda: session.get('profile', {}).get('sub', 'anonymous'))
def get_translation():
    data: Dict = request.get_json()
    try:
        translation = translate_ovp_to_english(
            subject_noun=data.get('subject_noun') or None,
            subject_suffix=data.get('subject_suffix') or None,
            verb=data.get('verb') or None,
            verb_tense=data.get('verb_tense') or None,
            object_pronoun=data.get('object_pronoun') or None,
            object_noun=data.get('object_noun') or None,
            object_suffix=data.get('object_suffix') or None,
            model='gpt-3.5-turbo'
        )
        return jsonify(translation=translation)
    except Exception as e:
        return jsonify(sentence=[], error=str(e)), 400
    

# route to get random sentence
@app.route('/api/builder/random', methods=['POST'])
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
        return jsonify(sentence=[], error=str(e)), 400

@app.route('/api/builder/random-example', methods=['GET'])
def get_random_example():
    try:
        choices = get_random_sentence_big()
        sentence = format_sentence(**{k: v['value'] for k, v in choices.items()})
        return jsonify(choices=choices, sentence=sentence)
    except Exception as e:
        return jsonify(sentence=[], error=str(e)), 400
    
@app.route('/api/translator/translate', methods=['POST'])
@limiter.limit(LIMITS['translate'], key_func = lambda: session.get('profile', {}).get('sub', 'anonymous'))
def translate_sentence():
    data: Dict = request.get_json()
    try:
        response = translate_english_to_ovp(data.get('english'))
        logging.info(response)
        if response['sim_simple'] < TRANSLATION_QUALITY_THRESHOLD:
            response['warning'] = (
                'The input sentence is complex, so alot of meaning may have been lost in breaking ' +
                'it down into simple sentences.'
            )
        elif response['sim_backwards'] < TRANSLATION_QUALITY_THRESHOLD:
            response['warning'] = 'The translation doesn\'t seem to be very accurate.'
        elif response['sim_comparator'] < TRANSLATION_QUALITY_THRESHOLD:
            response['warning'] = (
                'The translator doesn\'t know some of the words in your input sentence. ' +
                'It left the english words as placeholders and gave you the best translation it could.'
            )
        elif all([response[k] >= TRANSLATION_QUALITY_THRESHOLD for k in ['sim_simple', 'sim_backwards', 'sim_comparator']]):
            response['message'] = 'The translation is probably pretty good!'
        return jsonify(
            english=response['backwards'],
            paiute=response['target'],
            message=response.get('message', ''),
            warning=response.get('warning', '')
        )
    except Exception as e:
        return jsonify(error=str(e)), 400

# health check
@app.route('/api/healthz', methods=['GET'])
def health_check():
    return jsonify(status='ok')

