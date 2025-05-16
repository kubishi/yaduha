import argparse
import json
import pathlib
import random
from typing import Dict
from yaduha.translate.pipeline_sentence_builder import (
    NOUNS, Verb, Object, Subject,
    get_all_choices, format_sentence, sentence_to_str
)
from yaduha.translate.pipeline_back_translate import translate
from dotenv import load_dotenv

load_dotenv()

lessons = {
    "lesson_1": {
        "title": "Lesson 1: Pronouns",
        "description": "Pronouns",
        "choices": [
            ("subject_noun", list(Subject.PRONOUNS.keys())),
            ("verb", ["tüka"]),
            ("verb_tense", ["ti"])
        ]
    },
    "lesson_2": {
        "title": "Lesson 2: Intransitive Verbs",
        "description": "Intransitive Verbs",
        "choices": [
            ("subject_noun", list(Subject.PRONOUNS.keys())),
            ("verb", list(Verb.INTRANSITIVE_VERBS.keys())),
            ("verb_tense", ["ti"])
        ]
    },
    "lesson_3": {
        "title": "Lesson 3: Verb Tenses",
        "description": "Verb Tenses",
        "choices": [
            ("subject_noun", list(Subject.PRONOUNS.keys())),
            ("verb", list(Verb.INTRANSITIVE_VERBS.keys())),
            ("verb_tense", list(Verb.TENSES.keys()))
        ]
    },
    "lesson_4": {
        "title": "Lesson 4: Transitive Verbs",
        "description": "Transitive Verbs",
        "choices": [
            ("subject_noun", list(Subject.PRONOUNS.keys())),
            ("object_pronoun", list(Object.PRONOUNS.keys())),
            ("verb", list(Verb.TRANSITIVE_VERBS.keys())),
            ("verb_tense", None)
        ]
    },
    "lesson_5": {
        "title": "Lesson 5: Subject Nouns",
        "description": "Transitive Verbs",
        "choices": [
            ("subject_noun", list(NOUNS.keys())),
            ("subject_suffix", list(Subject.SUFFIXES.keys())),
            ("verb", None),
            ("object_pronoun", list(Object.PRONOUNS.keys())),
            ("verb_tense", None)
        ]
    },
    "lesson_6": {
        "title": "Lesson 6: Object Nouns",
        "description": "Transitive Verbs",
        "choices": [
            ("subject_noun", [*NOUNS.keys(), *Subject.PRONOUNS.keys()]),
            ("subject_suffix", None),
            ("object_noun", [*NOUNS.keys()]),
            ("object_suffix", None),
            ("verb", None),
            ("object_pronoun", None),
            ("verb_tense", None)
        ]
    },
    "lesson_7": {
        "title": "Lesson 7: Simple Sentences",
        "description": "Simple Sentences with any subject and object",
        "choices": [
            ("subject_noun", [*NOUNS.keys(), *Subject.PRONOUNS.keys()]),
            ("subject_suffix", None),
            ("verb", None),
            ("object_pronoun", None),
            ("object_noun", [*NOUNS.keys()]),
            ("object_suffix", None),
            ("verb_tense", None)
        ]
    },
}

def get_random_sentence(lesson: str) -> Dict[str, str]:
    if lesson not in lessons:
        raise ValueError(f"Lesson {lesson} not found")
    
    lesson_data = lessons[lesson]
    choices = get_all_choices()
    words = {pos: None for pos, _ in lesson_data["choices"]}
    for pos, pos_choices in lesson_data["choices"]:
        if pos_choices is None:
            pos_choices = list(choices[pos]['choices'].keys())
        if choices[pos]["requirement"] != "disabled":
            words[pos] = random.choice(pos_choices)

        choices = get_all_choices(**words)

    return words
        
def create_lesson_instance(lesson_key: str,
                           savepath: pathlib.Path,
                           num_sentences: int = 10) -> None:
    paiute_sentences = set()
    sentences = []
    for i in range(num_sentences):
        words = get_random_sentence(lesson_key)
        sentence = format_sentence(**words)
        paiute = sentence_to_str(sentence).strip() + "."
        if paiute in paiute_sentences:
            continue
        paiute_sentences.add(paiute)
        english = translate(**words)
        sentences.append({
            "sentence": sentence,
            "paiute": paiute,
            "english": english
        })

    lesson = {
        "key": lesson_key,
        "title": lessons[lesson_key]["title"],
        "description": lessons[lesson_key]["description"],
        "sentences": sentences
    }

    savepath.parent.mkdir(parents=True, exist_ok=True)
    savepath.write_text(json.dumps(lesson, indent=2, ensure_ascii=False))

def quiz_from_lesson(lesson_key: str) -> None:
    lesson = json.loads(pathlib.Path(f"lessons/{lesson_key}.json").read_text())
    sentences = lesson["sentences"]
    all_english = [l["english"] for l in sentences]

    for i, l in enumerate(sentences):
        print(f"Question {i+1}: {l['paiute']}")
        answers = random.sample(all_english, 3)
        answers.append(l["english"])
        random.shuffle(answers)
        for j, a in enumerate(answers):
            print(f"{j+1}. {a}")
        print()

        res = input("Answer: ")
        while not res.isdigit() or int(res) not in range(1, 5):
            res = input("Answer: ")
        res = int(res)
        if answers[res-1] == l["english"]:
            print("Correct!")
        else:
            print("Incorrect. The correct answer is:")
            print(l["english"])
        print()

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create Paiute lessons")
    subparsers = parser.add_subparsers(dest="command")

    create_parser = subparsers.add_parser("create", help="Create a lesson")
    create_parser.add_argument("lesson_key", help="Lesson key")
    create_parser.add_argument("--num-sentences", type=int, default=10, help="Number of sentences to generate")
    create_parser.add_argument("--savepath", default=None, help="Path to save the lesson")

    quiz_parser = subparsers.add_parser("quiz", help="Quiz from a lesson")
    quiz_parser.add_argument("lesson_key", help="Lesson key")

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    if not getattr(args, "command"):
        parser.print_help()
        return
    if args.command == "create":
        num_sentences = args.num_sentences

        if args.lesson_key == "all":
            if args.savepath is not None:
                savepath = pathlib.Path(args.savepath)
            else:
                savepath = pathlib.Path(f"lessons")
            for lesson_key in lessons.keys():
                create_lesson_instance(lesson_key, savepath.joinpath(f"{lesson_key}.json"), num_sentences)
        else:
            if args.savepath is not None:
                savepath = pathlib.Path(args.savepath)
            else:
                savepath = pathlib.Path(f"lessons/{args.lesson_key}.json")
            create_lesson_instance(args.lesson_key, savepath, num_sentences)
    elif args.command == "quiz":
        quiz_from_lesson(args.lesson_key)

            

if __name__ == "__main__":
    main()

