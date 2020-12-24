from collections import Counter
import numpy as np



def ReadFile(filename):
    X = []
    y = []
    for line in open(filename, 'r'):
        line = line.rstrip('\n')
        row = line.split(',')
        X.append(row[0:-1])
        y.append(row[-1])
    X = np.array(X)
    y = np.array(y)
    return X, y


def ProbOfLabels(y):
    count = Counter(y[1:])
    total = count["True"] + count["False"]
    ProbDict = {"True": count['True'] / total, "False": count['False'] / total}
    return (ProbDict, count["True"], count['False'])


def gaussianProb(trainCol, input):
    std = trainCol.std(axis=0)
    mean = trainCol.mean(axis=0)

    probDist = (1 / np.sqrt(2 * 3.14 * std ** 2)) * np.exp(-0.5 * ((input - mean) ** 2) / (std ** 2))
    return probDist


def ProbOfFeature(features, ProbLabelStore, y):
    """
    :param features: Input dataset
    :param y:  Label
    :return: Maximum probability dictionaries for each feature => One for spam and one for not spam
    """
    featureProbDictYes = {}
    featureProbDictNo = {}

    # Mean and std when SPAM and NO SPAM
    MeanYes = []
    MeanNo = []
    StdYes = []
    StdNo = []
    for i in range(len(features[0])):

        column = features[:, i]
        name = column[0].strip()
        countTT = 0  # when feature = True and label = True
        countFT = 0  # when feature = False and label = True
        for j in range(len(column)):
            # column[j] = column[j].strip()
            y[j] = y[j].strip()
            if column[j] == "True" and y[j] == "True":
                countTT += 1

            if column[j] == "False" and y[j] == "True":
                countFT += 1

            if column[j].isdigit():  # Handle real values
                if y[j] == "True":
                    if name not in featureProbDictYes:
                        featureProbDictYes[name] = [column[j]]
                    else:
                        x = featureProbDictYes.get(name)
                        x.append(column[j])
                        featureProbDictYes[name] = x

        # { FeatureName : {"True":prob, "False":prob }  When SPAM = TRUE }
        if name not in featureProbDictYes:
           # print("SPAM", name, "TT",countTT,"FT",countFT)
            featureProbDictYes[name] = {"True": countTT / ProbLabelStore[1], "False": countFT / ProbLabelStore[1]}

        countTT = 0  # when feature = True and label = False
        countFT = 0  # when feature = False and label = False
        for j in range(len(column)):
            column[j] = column[j].strip()
            y[j] = y[j].strip()
            if column[j] == "True" and y[j] == "False":
                countTT += 1
            if column[j] == "False" and y[j] == "False":
                countFT += 1
            if column[j].isdigit():
                if y[j] == "False":
                    if name not in featureProbDictNo:
                        featureProbDictNo[name] = [column[j]]
                    else:
                        x = featureProbDictNo.get(name)
                        x.append(column[j])
                        featureProbDictNo[name] = x

                    # { FeatureName : {"True":prob, "False":prob }  When SPAM = FALSE }

        if name not in featureProbDictNo:
            #print("No SPAM", name, "TF", countTT, "FF", countFT)
            featureProbDictNo[name] = {"True": countTT / ProbLabelStore[2], "False": countFT / ProbLabelStore[2]}

    print("Label Probabilities", ProbLabelStore)
    print("Feature parameters SPAM", featureProbDictYes)
    print("Feature parameters NO SPAM", featureProbDictNo)

    return featureProbDictYes, featureProbDictNo


def CalcConditionalProbs(featureProbDictYes, featureProbDictNo, ProbLabelStore, TestSet, output, realValueFeatures):
    ProbSpam = 0
    ProbNoSpam = 0
    LikelyhoodSpam = 1
    for key, values in featureProbDictYes.items():  # SPAM DICT

        testValue = TestSet.get(key)  # Eg :  key = isHtml. testValue = TestSet.get(key) = True
        values = featureProbDictYes.get(key)  # Eg : values = { isHtml = { True: prob, False:prob } }
        if key == '# sentences' or key == '# words':
            getRelevantProb = gaussianProb(np.array(values).astype(int), int(testValue))
        else:
            getRelevantProb = values[testValue]  # Eg: getRelevantProb = values[testValue]

        if getRelevantProb > 0:
            LikelyhoodSpam *= getRelevantProb  # Multiplying probabilities

    LikelyhoodSpam *= ProbLabelStore[0]['True']  # Multiply by P(Y)

    LikelyhoodNoSpam = 1
    for key, values in featureProbDictNo.items():  # NOT SPAM DICT
        testValue = TestSet.get(key)
        values = featureProbDictNo.get(key)
        if key == '# sentences' or key == '# words':
            getRelevantProb = gaussianProb(np.array(values).astype(int), int(testValue))
        else:
            getRelevantProb = values[testValue]  # Eg: getRelevantProb = values[testValue]

        if getRelevantProb > 0:
            LikelyhoodNoSpam *= getRelevantProb

    LikelyhoodNoSpam *= ProbLabelStore[0]['False']  # Multiply by P(Y)

    ProbSpam = LikelyhoodSpam / (LikelyhoodSpam + LikelyhoodNoSpam)
    ProbNoSpam = LikelyhoodNoSpam / (LikelyhoodSpam + LikelyhoodNoSpam)

    if ProbSpam >= 0.5:
        output.append("True")

    elif ProbNoSpam > 0.5:
        output.append("False")

    else:
        output.append("False")


fileNameTrain = 'spam_detection_train.csv'
X, y = ReadFile(fileNameTrain)
ProbLabelStore = ProbOfLabels(y)
realValueFeatures = X[:, [-1, -2]]

# You can remove or add any index to choose subset
Subset_Indexes = [0, 1, 2, 3, 4, 5, 6, 7]
print("____________________________________________")

print("By default, this code will use all features. To specify a subset, please add/remove indexes on variable named Subset_Indexes")

print("____________________________________________")
# Allows you to select a subset
featureSubset = X[:, Subset_Indexes]
# These are maximum-likelihood parameters
featureProbDictYes, featureProbDictNo = ProbOfFeature(featureSubset, ProbLabelStore, y)

fileNameTest = 'spam_detection_test.csv'
XTest, yTest = ReadFile(fileNameTest)
realValueFeaturesTest = XTest[:, [-1, -2]]
# XTest = XTest[:, :len(XTest[0]) - 2]

outputAns = []
for i in range(1, len(XTest)):
    testRow = XTest[i]
    Dict = {"in html": testRow[0], 'has emoji': testRow[1], "sent to list": testRow[2],
            'from .com': testRow[3], 'has my name': testRow[4]
        , 'has sig': testRow[5], '# sentences': testRow[6], '# words': testRow[7]}
    CalcConditionalProbs(featureProbDictYes, featureProbDictNo, ProbLabelStore, Dict, outputAns, realValueFeatures)

count = 0
for i in range(len(outputAns)):
    if outputAns[i] == yTest[i].strip():
        count += 1

print("Accuracy = ", (count / len(yTest)) * 100)
