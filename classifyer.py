import dspy
from typing import Literal
import time
import re


def main():
    gemma3n = OllamaModel("gemma3n:e2b")
    phi3 = OllamaModel("phi3")
    llama3 = OllamaModel("llama3.2")
    models = [gemma3n, llama3, phi3]
    # modelEvaluatior = ModelEvaluator(models, "vanilla", 200)
    # modelEvaluatior.makePredictions()

    # makePredictions(model, "test")
    for model in models:
        print(model.name, eval("data/test.txt",
              f"predictions/{model.name}(vanilla with 200)"))


class ModelEvaluator():

    evalMap = {"verylow": 0, "low": 1,
               "neutral": 2, "high": 3, "veryhigh": 4}

    dataset = "test"

    def __init__(self, models, modelState, size):
        self.models = models
        self.modelState = modelState
        self.examples = createDspExmaples(f"data/{self.dataset}.txt", size)
        self.size = size

    def makePredictions(self):
        for model in self.models:
            print(f"Started Model {model.name}")
            startTime = time.time()
            self.runModel(model)
            endTime = time.time()
            duration = (endTime - startTime)/60
            print(f"Model {model.name} in {duration} min finished")

    def getReviewSentiment(self, example, classifier):
        for i in range(10):
            try:
                review = example.get("movieReview")
                prediction = f"{
                    review} ({self.evalMap[classifier(movieReview=review).get("sentiment")]})"
                return prediction
            except:
                print(f"exception {i} in {review}")
                pass
        return "Exception"

    def runModel(self, model):
        dspy.configure(lm=model.model)
        movieReviewSentimentClassifier = dspy.Predict(ReviewClasifier)
        predictions = []

        for exmaple in self.examples:
            predictions.append(self.getReviewSentiment(
                exmaple, movieReviewSentimentClassifier))

        with open(f"predictions/{model.name}({self.modelState} with {self.size})", "w") as f:
            f.writelines("\n".join(predictions))
        return


def eval(goldPath, resultPath):
    result = loadDataSet(resultPath)
    gold = loadDataSet(goldPath)[:200]
    correctPredicted = 0
    for prediction, goldValue in zip(result, gold):
        try:
            if splitDataPoint(prediction)[1] == splitDataPoint(goldValue)[1]:
                correctPredicted += 1
        except:
            continue
    return correctPredicted/(len(gold))


def loadTraining(path):
    dataset = loadDataSet(path)
    return [splitDataPoint(dataPoint)[0] for dataPoint in dataset]


def splitDataPoint(dataPoint):
    dataPoint = dataPoint.strip()
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
