from flask import Flask, jsonify, send_from_directory
import json
import pathlib
import random

app = Flask(__name__)

MAX_QUESTIONS = 10

# Load lesson data
def load_lesson(lesson_key):
    lesson_path = pathlib.Path(f"lessons/{lesson_key}.json")
    if lesson_path.exists():
        return json.loads(lesson_path.read_text())
    else:
        return None

@app.route('/quiz/<lesson_key>', methods=['GET'])
def get_quiz(lesson_key):
    lesson = load_lesson(lesson_key)
    if not lesson:
        return jsonify({"error": "Lesson not found"}), 404

    sentences = lesson["sentences"]
    quiz_data = []
    all_english = [item["english"] for item in sentences]
    random.shuffle(sentences)
    for item in sentences[:MAX_QUESTIONS]:
        question = {
            "paiute": item["paiute"],
            "sentence_parts": item["sentence"],
            "correct": item["english"],
            "choices": random.sample(all_english, 3) + [item["english"]],
        }
        random.shuffle(question["choices"])
        quiz_data.append(question)

    lesson = {
        "key": lesson_key,
        "title": lesson["title"],
        "description": lesson["description"],
        "questions": quiz_data
    }
    return jsonify(lesson)

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/lessons', methods=['GET'])
def get_lessons():
    lesson_path = pathlib.Path("lessons")
    if not lesson_path.exists():
        return jsonify([])  # Return empty list if no lessons are available

    lessons = [json.loads(p.read_text()) for p in lesson_path.glob("*.json")]
    lessons = [{k: lesson[k] for k in ["key", "title", "description"]} for lesson in lessons]
    lessons = sorted(lessons, key=lambda x: x["key"])
    return jsonify(lessons)

if __name__ == '__main__':
    app.run(debug=True)
