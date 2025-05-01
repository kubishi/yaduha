from evaluate import load

comet_metric = None

def get_comet_score(sentence1: str, sentence2: str) -> float:
    """
    Compute the COMET score for a single prediction, reference, and source.
    
    Args:
        prediction (str): The predicted translation.
        reference (str): The reference translation.
        source (str): The source sentence.
        
    Returns:
        float: The COMET score.
    """
    global comet_metric
    if comet_metric is None:
        comet_metric = load("comet")
    result = comet_metric.compute(
        predictions=[sentence1],
        references=[sentence2],
        sources=[sentence2]
    )
    return result['mean_score']

def test():
    global comet_metric
    if comet_metric is None:
        comet_metric = load("comet")
    # Example usage
    predictions = ["The fire could be stopped", "Schools and kindergartens were open"]
    references = ["They were able to control the fire.", "Schools and kindergartens opened"]
    sources = references
    
    comet_score = comet_metric.compute(predictions=predictions, references=references, sources=sources)
    print(comet_score)
    
if __name__ == "__main__":
    test()