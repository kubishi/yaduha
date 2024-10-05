import pathlib
from yaduha.agentic import AgenticTranslator
import logging
import json

thisdir = pathlib.Path(__file__).parent.absolute()

# set logging level to INFO
# logging.basicConfig(level=logging.INFO)

def main():
    results = []
    for model in ["gpt-4o-mini", "gpt-4o"]:	
        print(f"=== Using model: {model} ===")
        target_sentence = "This coyote is eating rice."
        translation = AgenticTranslator(
            savepath=pathlib.Path(f"messages-openai-simple-{model}.json"),
            openai_model=model
        ).translate(target_sentence)
        results.append(translation.to_dict())
        print(translation, end="\n\n")

        target_sentence = "The boy and the girl are eating a quesadilla."
        translation = AgenticTranslator(
            savepath=pathlib.Path(f"messages-openai-complex-{model}.json"),
            openai_model=model
        ).translate(target_sentence)
        results.append(translation.to_dict())
        print(translation, end="\n\n")

        target_sentence = "The cook saw the ones who walked."
        translation = AgenticTranslator(
            savepath=pathlib.Path(f"messages-openai-nominalization-{model}.json"),
            openai_model=model
        ).translate(target_sentence)
        results.append(translation.to_dict())
        print(translation, end="\n\n")

    savepath = thisdir / "results.json"
    savepath.write_text(json.dumps(results, indent=4, ensure_ascii=False))

def run_cost():
    models = { # price per million tokens
        "gpt-4o": {
            "translation_prompt_tokens": 2.50,
            "translation_completion_tokens": 10.00,
            "back_translation_prompt_tokens": 2.50,
            "back_translation_completion_tokens": 10.0
        },
        "gpt-4o-mini": {
            "translation_prompt_tokens": 0.150,
            "translation_completion_tokens": 0.600,
            "back_translation_prompt_tokens": 0.150,
            "back_translation_completion_tokens": 0.600
        }	
    }

    resultspath = thisdir / "results.json"
    results = json.loads(resultspath.read_text())
    costs = {}
    for i, result in enumerate(results):
        model = "gpt-4o-mini" if i <= 2 else "gpt-4o"
        cost = sum([
            result[token] * models[model][token] / 1_000_000
            for token in [
                "translation_prompt_tokens",
                "translation_completion_tokens",
                "back_translation_prompt_tokens",
                "back_translation_completion_tokens"
            ]
        ])
        print(f"Cost for message {i + 1}: ${cost:.2f}")
        costs.setdefault(model, []).append(cost)


    print(f"Total cost for gpt-4o-mini: ${sum(costs['gpt-4o-mini']):.2f}")
    print(f"Average cost for gpt-4o-mini: ${sum(costs['gpt-4o-mini']) / 3:.2f}")

    print(f"Total cost for gpt-4o: ${sum(costs['gpt-4o']):.2f}")
    print(f"Average cost for gpt-4o: ${sum(costs['gpt-4o']) / 3:.2f}")


if __name__ == "__main__":
    # main()
    run_cost()