import dspy
from typing import Literal
import time


def main():
    dictionary = {"verylow": 0, "low": 1,
                  "neutral": 2, "high": 3, "veryhigh": 4}
    model = OllamaModel("llama3.2", "test")


def makePredictions(model, dataset):
    dspy.configure(lm=model.model)
    movieReviewSentimentClassifier = dspy.Predict(ReviewClasifier)
    dataSet = loadDataSet("data/{dataset}.txt")
    predictions = []
    startTime = time.time()

    for review in dataSet:
        predition = f"{review} ({movieReviewSentimentClassifier(
            movieReview=review).get("sentiment")})"
        predictions.append(predition)
    endTime = time.time()

    with open(r"predictions/llama3.2(vanilla)", "w") as f:
        f.writelines("\n".join(predictions))
    print(f"Total time {endTime - startTime} for model {model.name}")
    return


def loadDataSet(path):
    dataset = []
    with open(path, "r") as f:
        for line in f.readlines():
            dataset.append(line[:-5])
    return dataset


class ReviewClasifier(dspy.Signature):
    """Classify the sentiment of the movieReview."""
    movieReview: str = dspy.InputField()
    sentiment: Literal["verylow", "low", "neutral",
                       "high", "veryhigh"] = dspy.OutputField()


class OllamaModel():
    def __init__(self, modelName):
        self.name = modelName
        self.model = dspy.LM(f"ollama_chat/{modelName}",
                             api_base="http://localhost:11434", api_key="")


if __name__ == "__main__":
    main()
