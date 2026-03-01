# Yaduha

**A type-safe, AI-powered framework for structured language translation**

(readme documentation produced by Claude)

Yaduha is a Python framework for building translation systems that combine the power of Large Language Models (LLMs) with formal linguistic structures. It provides tools for creating grammatically-constrained translations with full type safety and verification.

## Features

- **🔧 Type-Safe Tool Framework**: Build LLM-callable tools with strict parameter validation
- **🤖 Agent Abstraction**: Unified interface for AI agents (OpenAI, with extensibility for others)
- **📝 Structured Sentences**: Define language grammars as Pydantic models for guaranteed correctness
- **🔄 Multiple Translation Strategies**: Choose between pipeline-based or free-form agentic translation
- **✅ Back-Translation Verification**: Automatically verify translation quality
- **📊 Token & Performance Tracking**: Built-in monitoring for costs and latency
- **🎯 Few-Shot Learning**: Automatic example generation for better LLM performance

## Quick Start

### Installation

```bash
pip install -e .
```

### Set up your OpenAI API key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

```python
from yaduha.translator.pipeline import PipelineTranslator
from yaduha.agent.openai import OpenAIAgent
from yaduha.language.ovp import SubjectVerbSentence, SubjectVerbObjectSentence

# Create a translator
translator = PipelineTranslator(
    agent=OpenAIAgent(
        model="gpt-4o-mini",
        api_key="your-api-key"
    ),
    SentenceType=(SubjectVerbObjectSentence, SubjectVerbSentence)
)

# Translate English to Owens Valley Paiute
result = translator("The dog is sleeping.")
print(f"Translation: {result.target}")
print(f"Back-translation: {result.back_translation.source}")
print(f"Tokens used: {result.prompt_tokens + result.completion_tokens}")
```

## Architecture Overview

```
┌─────────────────┐
│  User Input     │
│  (English)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Translator    │ ◄─── Pipeline or Agentic
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Tools       │ ◄─── EnglishToSentences, SentenceToEnglish
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Agent       │ ◄─── OpenAI (gpt-4o, gpt-4o-mini)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Structured    │
│   Sentences     │ ◄─── SubjectVerbSentence, SubjectVerbObjectSentence
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Target Lang    │
│  (e.g., OVP)    │
└─────────────────┘
```

## Core Concepts

### 1. Agents

Agents are AI models that can generate text and call tools. Yaduha provides an OpenAI implementation with full type safety:

```python
from yaduha.agent.openai import OpenAIAgent

agent = OpenAIAgent(
    model="gpt-4o-mini",
    api_key="...",
    temperature=0.0  # For deterministic outputs
)
```

### 2. Tools

Tools are callable functions that LLMs can use. They have strict type validation and automatic schema generation:

```python
from yaduha.tool import Tool
from typing import ClassVar, List, Dict

class SearchTool(Tool):
    name: ClassVar[str] = "search"
    description: ClassVar[str] = "Search for information"

    def _run(self, query: str, limit: int = 5) -> List[Dict]:
        # Tool implementation
        return [{"result": "..."}]
```

### 3. Structured Sentences

Define language grammars as Pydantic models:

```python
from yaduha.language import Sentence
from pydantic import BaseModel

class MyLanguageSentence(Sentence):
    subject: Noun
    verb: Verb

    def __str__(self) -> str:
        # Convert to target language representation
        return f"{self.subject.target}-{self.verb.target}"

    @classmethod
    def get_examples(cls):
        return [
            ("I sleep", MyLanguageSentence(subject=..., verb=...)),
            # More examples...
        ]
```

### 4. Translators

Two translation strategies are provided:

#### Pipeline Translator (Structured)
Guarantees grammatical correctness by constraining output to defined sentence structures:

```python
translator = PipelineTranslator(
    agent=agent,
    SentenceType=(SentenceType1, SentenceType2)  # Can mix multiple types
)
```

#### Agentic Translator (Free-form)
Uses LLM reasoning with optional tool assistance for flexible translation:

```python
translator = AgenticTranslator(
    agent=agent,
    system_prompt="You are a translation expert...",
    tools=[SearchTool(), DictionaryTool(), PipelineTranslator(...)]
)
```


## Correctness-First Translation with Structured Outputs

Yaduha implements **LLM-Assisted Rule-Based Machine Translation (LLM-RBMT)**, a novel paradigm designed specifically for **no-resource and extremely low-resource languages** where traditional neural MT approaches fail due to lack of parallel corpora. Rather than relying on unconstrained text generation, Yaduha leverages **Pydantic models as linguistic constraints** to guarantee grammatical correctness while harnessing the semantic understanding of large language models.

### The Core Innovation: Pydantic Models as Linguistic Grammars

In Yaduha, every grammatical structure (such as `SubjectVerbSentence` or `SubjectVerbObjectSentence`) is defined as a **Pydantic model** that explicitly encodes the syntactic and morphological rules of the target language. These models act as **type-safe grammars** where each field corresponds to a validated linguistic feature:

- **Part-of-speech categories** (Subject, Verb, Object)
- **Morphological features** (`Person`, `Plurality`, `Proximity`, `Inclusivity`)
- **Tense-aspect systems** (`TenseAspect`: past/present/future, simple/continuous/perfect)
- **Language-specific constraints** (e.g., fortis/lenis consonant mutation in OVP)

This structured representation enables what we call **rule-based sentence synthesis**: the LLM never needs to "know" the target language directly. Instead, it acts as a **syntactic and structural intermediary**, decomposing natural English into structured forms that our grammatical rules can then synthesize into the target language.

### How It Works: Structured Outputs via Constrained Decoding

Yaduha leverages OpenAI's [**Structured Outputs**](https://openai.com/index/introducing-structured-outputs-in-the-api/) feature (also called *constrained decoding*) to force the LLM to output responses conforming exactly to our Pydantic schemas. Here's the process:

1. **Schema Generation**: Pydantic models are automatically converted to JSON Schema definitions
2. **Constrained Generation**: The LLM generates outputs that are guaranteed to conform to the schema
3. **Automatic Validation**: Responses are validated at runtime, ensuring grammatical correctness
4. **Sentence Synthesis**: Valid structured data is rendered into the target language using linguistic rules

For example:

```python
from yaduha.translator.pipeline import PipelineTranslator
from yaduha.agent.openai import OpenAIAgent
from yaduha.language.ovp import SubjectVerbSentence, SubjectVerbObjectSentence

translator = PipelineTranslator(
    agent=OpenAIAgent(model="gpt-4o-mini"),
    SentenceType=(SubjectVerbObjectSentence, SubjectVerbSentence)
)

# Input: Complex English sentence
result = translator("The dog is sitting at the lakeside, drinking some water.")

# Output: Grammatically valid OVP sentence(s)
print(f"OVP: {result.target}")
print(f"Back-translation: {result.back_translation.source}")
```

Behind the scenes, the LLM performs **sentence segmentation**: breaking down the input into simple SV/SVO structures that match our defined sentence types. Each segment is then validated against the Pydantic schema, ensuring every generated sentence is well-formed according to OVP's grammatical rules.

### Why This Matters for Endangered Languages

This **correctness-first approach** is particularly crucial for endangered and no-resource languages because:

* **No parallel data required**: The system works with only a lexicon and grammatical rules --- no bilingual corpus needed
* **Guaranteed grammatical validity**: Every output is structurally correct by construction
* **Suitable for language learning**: Learners can trust the grammatical correctness of generated sentences
* **Extensible**: Adding new vocabulary or grammatical patterns is straightforward
* **Transparent**: The structured intermediate representation is human-readable and debuggable

### Learn More

For more information including the evaluation methodology and empirical results demonstrating this approach, please read our paper:

**📄 [LLM-Assisted Rule Based Machine Translation for Low/No-Resource Languages](https://arxiv.org/pdf/2405.08997)**


## Currently Supported Languages

### Owens Valley Paiute (OVP)

Yaduha includes a complete implementation for Owens Valley Paiute, a Uto-Aztecan language:

- **37 nouns** (coyote, dog, water, mountain, etc.)
- **35 verbs** (14 transitive, 21 intransitive)
- **Full pronoun system** (person, number, proximity, inclusivity)
- **Tense/aspect system** (6 tenses: past simple/continuous, present simple/continuous/perfect, future)
- **Complex morphology** (fortis/lenis consonant mutation, proximity-based suffixes)

**Sentence structures:**
- Subject-Verb: "I sleep" → "nüü üwi-dü"
- Subject-Verb-Object: "You read the mountains" → "üü toyabi-noka ui-nia-dü"

## Examples

### Example 1: Basic Translation

```python
from yaduha.translator.pipeline import PipelineTranslator
from yaduha.agent.openai import OpenAIAgent
from yaduha.language.ovp import SubjectVerbObjectSentence
import os

translator = PipelineTranslator(
    agent=OpenAIAgent(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"]
    ),
    SentenceType=SubjectVerbObjectSentence
)

result = translator("The cat drinks water")
print(f"English: {result.source}")
print(f"OVP: {result.target}")
print(f"Verification: {result.back_translation.source}")
```

### Example 2: Custom Tools

```python
from yaduha.translator.agentic import AgenticTranslator
from yaduha.tool import Tool
import requests

class DictionaryTool(Tool):
    name: ClassVar[str] = "dictionary_lookup"
    description: ClassVar[str] = "Look up word translations"

    def _run(self, word: str) -> List[Dict]:
        response = requests.get(f"https://api.example.com/lookup?word={word}")
        return response.json()

translator = AgenticTranslator(
    agent=agent,
    tools=[DictionaryTool()]
)

result = translator("How do you say 'hello' in Paiute?")
print(result.target)
print(f"Confidence: {result.metadata['confidence_level']}")
```

### Example 3: Token Tracking

```python
result = translator("A complex sentence to translate")

print(f"Translation time: {result.translation_time:.2f}s")
print(f"Forward tokens: {result.prompt_tokens + result.completion_tokens}")
print(f"Back-translation tokens: {result.back_translation.prompt_tokens + result.back_translation.completion_tokens}")
print(f"Total cost (approx): ${(result.prompt_tokens * 0.15 + result.completion_tokens * 0.60) / 1_000_000:.4f}")
```

## Documentation

- [Getting Started Guide](docs/getting-started.md)
- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api-reference.md)
- [Creating Custom Languages](docs/custom-languages.md)
- [Building Tools](docs/building-tools.md)
- [Examples & Tutorials](docs/examples.md)

## Project Structure

```
yaduha/
├── yaduha/                    # Main package
│   ├── agent/                 # AI agent abstraction
│   │   ├── __init__.py
│   │   └── openai.py
│   ├── tool/                  # Tool framework
│   │   ├── __init__.py
│   │   ├── english_to_sentences.py
│   │   └── sentence_to_english.py
│   ├── translator/            # Translation strategies
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   └── agentic.py
│   └── language/              # Language implementations
│       ├── __init__.py
│       └── ovp/               # Owens Valley Paiute
│           ├── __init__.py
│           ├── vocab.py
│           └── prompts.py
├── scripts/                   # Example scripts
├── docs/                      # Documentation
└── setup.py
```

## Development

Please find the Development Documentation at [docs/index.md](docs/index.md).

### Running Tests

```bash
# Test pipeline translator
python scripts/test_pipeline_translator.py

# Test agentic translator
python scripts/test_agentic_translator.py

# Test agent functionality
python scripts/test_agent.py
```

## Citation

If you use Yaduha in your research, please cite:

```bibtex
@software{yaduha2024,
  title={Yaduha: A Type-Safe Framework for Structured Language Translation},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-username]/yaduha}
}
```
