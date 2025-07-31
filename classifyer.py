import dspy
from typing import Literal
import time


def main():
    simpleModelName = "llama3.2"
    simpleLLM = dspy.LM(f"ollama_chat/{simpleModelName}",
                        api_base="http://localhost:11434", api_key="")
    dspy.configure(lm=simpleLLM)
    movieReviewSentimentClassifier = dspy.Predict(ReviewClasifier)
    dataSet = loadDataSet("data/test.txt")
    predictions = []
    startTime = time.time()
    count = 0
    for review in dataSet:
        count += 1
        predition = f"{count} {review} ({movieReviewSentimentClassifier(
            movieReview=review).get("sentiment")})"
        print(predition)
        predictions.append(predition)
    endTime = time.time()

    with open(r"predictions/llama3.2(vanilla)", "w") as f:
        f.writelines(predictions)
    print(f"Total time {endTime - startTime} for model {simpleModelName}")


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


if __name__ == "__main__":
    main()
