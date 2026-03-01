# Yaduha

A framework for translating into low-resource and endangered languages using LLMs with grammatical constraints. Implements **LLM-Assisted Rule-Based Machine Translation (LLM-RBMT)** -- the LLM never needs to "know" the target language. Instead, it decomposes English input into structured forms that linguistic rules synthesize into the target language.

Based on the paper: [LLM-Assisted Rule Based Machine Translation for Low/No-Resource Languages](https://arxiv.org/pdf/2405.08997)

## Install

```bash
pip install yaduha          # core (pydantic only)
pip install yaduha[full]    # + LLM backends, API server, evaluation metrics
```

Language packages are installed separately:

```bash
pip install yaduha-ovp      # Owens Valley Paiute
```

## Usage

```python
from yaduha.agent.openai import OpenAIAgent
from yaduha.translator.pipeline import PipelineTranslator

translator = PipelineTranslator.from_language(
    "ovp",
    agent=OpenAIAgent(model="gpt-4o-mini", api_key="..."),
)

result = translator("The dog is sleeping.")
print(result.target)                    # target language output
print(result.back_translation.source)   # back-translation for verification
```

## How it works

Sentence types are Pydantic models that encode a language's grammar. The LLM fills them via constrained decoding (structured output), guaranteeing every output is grammatically valid by construction. No parallel corpora required -- only a lexicon and grammatical rules.

```
English input
    -> LLM decomposes into structured Sentence models
    -> Sentence.__str__() renders target language text
    -> (optional) back-translate for verification
```

## Translation strategies

**Pipeline** -- grammar-guaranteed via structured output. The LLM maps English into one or more `Sentence` subclasses; the `__str__` method renders the target language. Output is always grammatically correct.

```python
from yaduha.translator.pipeline import PipelineTranslator
translator = PipelineTranslator(agent=agent, SentenceType=(SVO, SV))
```

**Agentic** -- free-form with tool assistance. The LLM reasons freely and can call tools (dictionary lookup, pipeline translator, etc.) to produce a translation.

```python
from yaduha.translator.agentic import AgenticTranslator
translator = AgenticTranslator(agent=agent, tools=[dictionary, pipeline])
```

## Creating a language package

Define sentence types as Pydantic models and register via entrypoint:

```python
from yaduha.language import Sentence

class SVSentence(Sentence):
    subject: Subject
    verb: Verb

    def __str__(self) -> str:
        return f"{self.subject.render()} {self.verb.render()}"

    @classmethod
    def get_examples(cls) -> list[tuple[str, "SVSentence"]]:
        return [("I sleep.", cls(subject=..., verb=...))]
```

```toml
# pyproject.toml
[project.entry-points."yaduha.languages"]
my_lang = "my_package:language"
```

See [yaduha-ovp](https://github.com/kubishi/yaduha-ovp) for a complete example.

## LLM backends

- OpenAI (`yaduha.agent.openai`)
- Anthropic (`yaduha.agent.anthropic`)
- Google Gemini (`yaduha.agent.gemini`)
- Ollama (`yaduha.agent.ollama`)

## Evaluation

Built-in evaluators for back-translation quality: chrF, BLEU, BERTScore, COMET, and OpenAI embedding similarity.

```python
from yaduha.evaluator.chrf import ChrfEvaluator
from yaduha.evaluator import batch_evaluate

results = batch_evaluate(translations, ChrfEvaluator())
```

## CLI

```bash
yaduha languages list              # list installed language packages
yaduha languages info ovp          # show language details
yaduha languages validate ovp      # validate a language implementation
yaduha serve                       # start FastAPI server + dashboard
```

## Development

```bash
pip install yaduha[dev]
pytest tests/ -q
ruff check yaduha/ tests/
pyright yaduha/
```

## Citation

```bibtex
@article{coleman2024llm,
  title={LLM-Assisted Rule Based Machine Translation for Low/No-Resource Languages},
  author={Coleman, Jared and Cuadros, Diego and Leeds, Nicholas and Krishnamachari, Bhaskar and Toal, Kira and Rosales, Ruben and Iskarous, Khalil},
  journal={arXiv preprint arXiv:2405.08997},
  year={2024}
}
```
