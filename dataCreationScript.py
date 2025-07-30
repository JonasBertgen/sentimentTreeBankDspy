import os
import re


def main():
    relPath = "../sentiment-treebank/fiveclass/"
    devFileName = "dev.txt"
    testFileName = "test.txt"
    trainFileName = "train.txt"

    dev = getFile(os.path.join(relPath, devFileName))
    test = getFile(os.path.join(relPath, testFileName))
    train = getFile(os.path.join(relPath, trainFileName))

    writeFile(devFileName, dev)
    writeFile(trainFileName, train)
    writeFile(testFileName, test)


def getFile(inputPath):
    file = []
    with open(inputPath, "r") as f:
        for line in f.readlines():
            file.append(line)
    return file


def writeFile(fileName, dataPointsList):
    with open("data/" + fileName, "w") as f:
        for dataPoint in dataPointsList:
            simpleSentence, score = simplifyDataPoint(dataPoint)
            f.write(simpleSentence + " (" + score + ")\n")
    pass


def simplifyDataPoint(original):
    regex = re.findall(r"d", original)
    score = re.findall("[0-9]", original)[0]  # score
    splitter = re.findall("[0-9]", original)[-1]  # score
    simpleSentence = original.split(str(splitter))
    # last number neutral since point
    simpleSentence = ''.join(simpleSentence[:-1])
    simpleSentence = re.sub("[0-9()]", "", simpleSentence)
    simpleSentence = re.sub("\s+", " ", simpleSentence)
    simpleSentence = simpleSentence.strip() + "."
    return simpleSentence, score


if __name__ == "__main__":
    main()
