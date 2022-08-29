import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import tree


def load_data(filename):
    with open(filename) as f:
        parsed = json.load(f)
        for item in parsed:
            all_features = item.get("features")
            averageSentenceLength.append(all_features.get("averageSentenceLength"))
            averageWordLengthPerSentence.append(all_features.get("averageWordLengthPerSentence"))
            averageWordLength.append(all_features.get("averageWordLength"))
            averagePassives.append(all_features.get("averagePassives"))
            averageFormalPronouns.append(all_features.get("averageFormalPronouns"))
            averageInformalPronouns.append(all_features.get("averageInformalPronouns"))


def run_analysis():
    features = np.array([averageSentenceLength, averageWordLengthPerSentence, averageWordLength,
                         averagePassives, averageFormalPronouns, averageInformalPronouns]).T
    labels = np.array([1] * 50 + [2] * 100)
    md = 4
    deci_tree = tree.DecisionTreeClassifier(max_depth=md)
    deci_tree = deci_tree.fit(features, labels)

    plt.figure(figsize=(15, 10))
    tree.plot_tree(deci_tree)
    plt.title("Decision tree on 150 samples, 50 formal, and 100 informal/semi-formal. Maximum depth: 4 layers.")
    plt.show()

    test_predict = deci_tree.predict(features)
    print("accuracy of this model is:", accuracy_score(labels, test_predict))


averageSentenceLength = []
averageWordLengthPerSentence = []
averageWordLength = []
averagePassives = []
averageFormalPronouns = []
averageInformalPronouns = []

load_data("data/processed/formal.json")
load_data("data/processed/informal.json")

run_analysis()
