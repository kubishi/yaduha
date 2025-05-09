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
    return {
        "precision": results["precision"][0],
        "recall": results["recall"][0],
        "f1": results["f1"][0],
        "hashcode": results["hashcode"]
    }

def get_bertscore_batch(sentences1: List[str],
                        sentences2: List[str],
                        lang: str = "en",
                        metric: Optional[str] = None) -> Union[List[float], Dict[str, List[float]]]:
     global bertscore
     if bertscore is None:
          bertscore = load("bertscore")
     results = bertscore.compute(
          predictions=sentences1,
          references=sentences2,
          lang=lang
     )
     if metric is not None:
          return results[metric]
     return {
          "precision": results["precision"],
          "recall": results["recall"],
          "f1": results["f1"],
          "hashcode": results["hashcode"]
     }

def test():
    pred = "The cat sat on the mat."
    ref = "The cat is sitting on the mat."
    print(f"Prediction: {pred}")
    print(f"Reference: {ref}")
    score = get_bertscore(pred, ref)
    print(f"BERTScore: {score}")
    print()

if __name__ == "__main__":
    test()
