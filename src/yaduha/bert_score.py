from evaluate import load
from typing import List, Optional, Dict, Union
bertscore = None

def get_bertscore(sentence1: str,
                  sentence2: str,
                  lang: str = "en",
                  metric: Optional[str] = None) -> Union[float, Dict[str, float]]:
    global bertscore
    if bertscore is None:
        bertscore = load("bertscore")
    results = bertscore.compute(
        predictions=[sentence1],
        references=[sentence2],
        lang=lang
    )
    if metric is not None:
        return results[metric][0]
    return {metric: vals[0] for metric, vals in results.items()}

def test():
    predictions = ["hello there", "general kenobi"]
    references = ["hello there", "general kenobi"]
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    print(results)

if __name__ == "__main__":
    test()
