import dspy
from dspy.teleprompt import MIPROv2
from typing import Literal
import time
import re


def main():
    gemma3n = OllamaModel("gemma3n:e2b")
    phi3 = OllamaModel("phi3")
    llama3 = OllamaModel("llama3.2")
    models = [gemma3n, llama3, phi3]
   # tp = dspy.MIPROv2(metric=getMetric(), auto="light", num_threads=1)
   # opt = tp.compile(
   #     sentimentAnaly, trainset=trainset[:5], max_bootstrapped_demos=2, max_labeled_demos=2)
    for model in models:
        trained = trainingRun(model)
        trained.save(f"./models/testrun-{model.name}.json")


def trainingRun(model):
    print(f"training run model {model.name} started")
    dspy.configure(lm=model.model)
    sentimentAnaly = dspy.Predict(ReviewClasifier)
    trainset = createDspExamples("data/train.txt", 10)[:5]
    print(f"training run model {model.name} ended")
    return getMIPROv2Optimized(model.model, trainset, sentimentAnaly)


class ModelEvaluator():

    evalMap = {"verylow": 0, "low": 1,
               "neutral": 2, "high": 3, "veryhigh": 4}

    dataset = "test"

    def __init__(self, models, modelState, size):
        self.models = models
        self.modelState = modelState
        self.examples = createDspExamples(f"data/{self.dataset}.txt", size)
        self.size = size

    def makePredictions(self):
        count = 0
        for model in self.models:
            print(f"Started Model {model.name}")
            startTime = time.time()
            self.runModel(model)
            endTime = time.time()
            duration = (endTime - startTime)/60
            print(f"Model {model.name} in {duration} min finished")

    def getReviewSentiment(self, example, classifier):
        for i in range(3):
            try:
                review = example.get("movieReview")
                prediction = f"{
                    review} ({self.evalMap[classifier(movieReview=review).get("sentiment")]})"
                return prediction
            except:
                # print(f"exception {i} in {review}")
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


def getMIPROv2Optimized(model, trainset, toBeImp):
    teleprompter = MIPROv2(
        metric=getMetric, auto="light", prompt_model=model)
    return teleprompter.compile(
        toBeImp,
        trainset=trainset,
        requires_permission_to_run=False)


def getMetric(example, pred, trace=None):
    if example.sentiment == pred.sentiment:
        return 1.0
    else:
        return 0.0


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


def createDspExamples(pathFile, size):
    dictionary = {0: "verylow", 1: "low",
                  2: "neutral", 3: "high", 4: "veryhigh"}
    exampleList = []
    dataSet = loadDataSet(pathFile)

    if size < len(dataSet):
        dataSet = dataSet[:size]

    for entry in dataSet:
        dataLablePair = splitDataPoint(entry)
        example = dspy.Example(
            movieReview=dataLablePair[0],
            sentiment=dictionary[dataLablePair[1]]
            # Specify that movieReview is the input field
        ).with_inputs('movieReview')
        exampleList.append(example)
    return exampleList


class ReviewClasifier(dspy.Signature):
    """Classify the sentiment of the movieReview."""
    movieReview: str = dspy.InputField()
    sentiment: Literal["verylow", "low", "neutral",
                       "high", "veryhigh"] = dspy.OutputField()

    # def predict(self, movieReview: str):
    #    return self.predict(movieReview=movieReview)

    def forward(self, input):
        return self.predict(self, input=input)


class OllamaModel():
    def __init__(self, modelName):
        self.name = modelName
        self.model = dspy.LM(f"ollama_chat/{modelName}",
                             api_base="http://localhost:11434", api_key="")


if __name__ == "__main__":
    main()
