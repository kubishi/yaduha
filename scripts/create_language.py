#!/usr/bin/env python3
"""
Simple script to create a new language using the editor.
"""

import os
from pathlib import Path
import dotenv

dotenv.load_dotenv()

from yaduha.agent.anthropic import AnthropicAgent
from yaduha.editor import LanguageEditor
from yaduha.editor.vocabulary import WordCategory
from yaduha.editor.grammar import WordOrder

# Create agent and editor
agent = AnthropicAgent(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-sonnet-4-20250514"
)
editor = LanguageEditor(agent)

# Create a new language
lang_path = Path("./my-lang")
editor.create_language(
    path=lang_path,
    name="My Language",
    code="myl",
    description="A test constructed language",
    word_order=WordOrder.sov,
)
print(f"Created language at {lang_path}")

# Get vocabulary suggestions
print("\n--- Vocabulary Suggestions ---")
suggestion = editor.suggest_vocabulary("water", WordCategory.noun)
print(f"water -> {suggestion.target}")
print(f"  Rationale: {suggestion.rationale}")

# Generate vocab code
suggestions = editor.suggest_vocabulary_batch([
    ("sun", WordCategory.noun),
    ("moon", WordCategory.noun),
    ("run", WordCategory.intransitive_verb),
    ("see", WordCategory.transitive_verb),
])
print("\nGenerated vocabulary:")
for s in suggestions:
    print(f"  {s.english} -> {s.target}")

code = editor.apply_vocabulary(suggestions)
print("\n--- Generated Code ---")
print(code)

# Design a sentence type
print("\n--- Sentence Type Design ---")
template = editor.design_sentence_type(
    description="Simple statements with subject and intransitive verb",
    example_sentences=["The cat sleeps.", "The dog runs.", "The bird flies."],
    word_order=WordOrder.sov,
)
print(f"Name: {template.name}")
print(f"Description: {template.description}")
print(f"Components: {template.required_components}")

# Generate the sentence type code
print("\n--- Generated Sentence Type Code ---")
st_code = editor.generate_sentence_type_code(template)
print(st_code)

print("\n--- Session Summary ---")
print(editor.get_session_summary())
