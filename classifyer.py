import dspy
from typing import Literal
import time
import re


def main():
    gemma3n = OllamaModel("gemma3n:e2b")
    phi3 = OllamaModel("phi3")
    llama3 = OllamaModel("llama3.2")
    models = [gemma3n, phi3, llama3]

    # makePredictions(model, "test")
    # print(eval("data/test.txt", f"predictions/{model.name}(vanilla)"))
    exmaples = createDspExmaples("data/test.txt", 200)
    for e in exmaples:
        print(e)


def makePredictions(model, dataset):
    dictionary = {"verylow": 0, "low": 1,
                  "neutral": 2, "high": 3, "veryhigh": 4}
    dspy.configure(lm=model.model)
    movieReviewSentimentClassifier = dspy.Predict(ReviewClasifier)
    dataSet = loadTraining(f"data/{dataset}.txt")
    predictions = []
    startTime = time.time()

    for review in dataSet:
        predition = f"{review} ({dictionary[movieReviewSentimentClassifier(
            movieReview=review).get("sentiment")]})"
        predictions.append(predition)
    endTime = time.time()

    with open(f"predictions/{model.name}(vanilla)", "w") as f:
        f.writelines("\n".join(predictions))
    print(f"Total time {endTime - startTime} for model {model.name}")
    return


def eval(resultPath, goldPath):
    def getLable(line): return int(re.findall(r"\((\d)\)", line)[-1])
    result = [getLable(dataPoint) for dataPoint in loadDataSet(resultPath)]
    gold = [getLable(dataPoint) for dataPoint in loadDataSet(goldPath)]
    correctPredicted = 0
    for prediction, goldValue in zip(result, gold):
        if prediction == goldValue:
            correctPredicted += 1
    return correctPredicted/(len(gold))


def loadTraining(path):
    dataset = loadDataSet(path)
    return [splitDataPoint(dataPoint)[0] for dataPoint in dataset]


def splitDataPoint(dataPoint):
    review = dataPoint[:-5]
    lable = int(re.findall(r"\d", dataPoint[-5:])[0])
    return review, lable


def loadDataSet(path):
    dataset = []
    with open(path, "r") as f:
        for line in f.readlines():
            dataset.append(line)
    return dataset


def createDspExmaples(pathFile, size):
    dictionary = {0: "verylow", 1: "low",
                  2: "neutral", 3: "high", 4: "veryhigh"}
    exampleList = []
    dataSet = loadDataSet(pathFile)

    if size < len(dataSet):
        dataSet = dataSet[:size]

    for entry in dataSet:
        dataLablePair = splitDataPoint(entry)
        exampleList.append(dspy.Example(
            movieReview=dataLablePair[0],
            sentiment=dictionary[dataLablePair[1]])
        )
    return exampleList


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
