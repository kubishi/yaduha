# To-Dos & Roadmap for Yaduha

## 1. OpenAPI Tool Integration

* [ ] Implement a **universal OpenAPI tool loader** that converts any OpenAPI spec into a `Tool` instance automatically.

## 2. Multi-Agent Support

* [ ] Add **Llama Agent** (via Ollama or Groq SDK).
* [ ] Add **Anthropic Agent** (Claude 3.x SDK integration).
* [ ] Add **Gemini Agent** (Google Generative AI SDK).

## 3. Confidence & Evidence Framework

* [ ] Reconsider `AgenticTranslator` **confidence scoring** and **evidence mode**.
* [ ] Allow user control via init params:

  ```python
  AgenticTranslator(confidence=True, evidence=False)
  ```

## 4. Experiment Tracking & Cloud Storage
(Weights & Biases)
* [ ] Design a **unified experiment runner** (e.g. `ExperimentManager`).
* [ ] Log translations, prompts, metadata (latency, tokens, model, confidence, etc.).
* [ ] Support **cloud persistence** somehow.
