import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import os
import random

from utils import tree_to_code

FEATURE_NAMES = ["SentenceLength", "WordsPerSentence", "WordLength", "Passives", "FormalPronouns", "InformalPronouns"]
VARIABLE_NAMES = ["charsPerSentence", "wordsPerSentence", "averageWordLength", "averagePassives",
                  "averageFormalPronouns", "averageInformalPronouns"]

INFORMAL_DATA_FILE = "data/processed/informal.json"
FORMAL_DATA_FILE = "data/processed/formal.json"
INFORMAL_DATA_FILE_BIGGER = "data/processed/informal_bigger.json"
FORMAL_DATA_FILE_BIGGER = "data/processed/formal_bigger.json"

averageSentenceLength = []
averageWordLengthPerSentence = []
averageWordLength = []
averagePassives = []
averageFormalPronouns = []
averageInformalPronouns = []
labels = []


def load_data(filename: str, label: str, max_items=550):
    assert os.path.exists(filename), f"{filename} does not exist."

    with open(filename, 'rt') as f:
        parsed = json.load(f)

        if len(parsed) > max_items:
            parsed = random.sample(parsed, max_items)

        for i, item in enumerate(parsed):
            all_features = item.get("features")
            if not all_features.get("averageSentenceLength") is None:
                averageSentenceLength.append(all_features.get("averageSentenceLength"))
            else:
                averageSentenceLength.append(all_features.get("charsPerSentence"))

            if not all_features.get("averageWordLengthPerSentence") is None:
                averageWordLengthPerSentence.append(all_features.get("averageWordLengthPerSentence"))
            else:
                averageWordLengthPerSentence.append(all_features.get("wordsPerSentence"))

            averageWordLength.append(all_features.get("averageWordLength"))
            averagePassives.append(all_features.get("averagePassives"))
            averageFormalPronouns.append(all_features.get("averageFormalPronouns"))
            averageInformalPronouns.append(all_features.get("averageInformalPronouns"))
            if label == "formal":
                labels.append(1)
            elif label == "informal":
                labels.append(2)
            else:
                raise ValueError(f"'{label}' is not a valid label")

    print(f"Loaded {filename}")
    print(f"Contains {len(parsed)} items.")


def n_fold_cross_validation(X: np.ndarray, y: np.array, md: int, n: int = 10):
    clf = tree.DecisionTreeClassifier(max_depth=md)
    scores = cross_val_score(clf, X, y, cv=n, scoring='f1_macro')

    print(f"*\n* {n}-fold cross validation\n*")
    print(f"* scores: {scores}\n*")
    print(f"* mean(sd): {scores.mean():.3f} ({scores.std():.3f})")

    return scores.mean()


def run_analysis(md=4, verbose=False):
    X = np.array([averageSentenceLength, averageWordLengthPerSentence, averageWordLength,
                  averagePassives, averageFormalPronouns, averageInformalPronouns]).T
    y = np.array(labels)

    deci_tree = tree.DecisionTreeClassifier(max_depth=md, min_samples_leaf=50)
    deci_tree = deci_tree.fit(X, y)

    if verbose:
        plt.figure(figsize=(20, 10))
        tree.plot_tree(deci_tree, feature_names=FEATURE_NAMES, fontsize=7)
        plt.title(
            f"Decision tree on {len(y)} samples, {np.count_nonzero(y == 1)} formal, "
            f"and {np.count_nonzero(y == 2)} informal/semi-formal. Maximum depth: {md} layers.")
        plt.show()

    test_predict = deci_tree.predict(X)

    if verbose:
        report = classification_report(y, test_predict)
        print(report)

        tree_to_code(deci_tree, VARIABLE_NAMES)

    return n_fold_cross_validation(X, y, md, 10)


def run_single():
    random.seed(13)
    load_data(FORMAL_DATA_FILE, 'formal')
    load_data(INFORMAL_DATA_FILE, 'informal')
    load_data(FORMAL_DATA_FILE_BIGGER, 'formal')
    load_data(INFORMAL_DATA_FILE_BIGGER, 'informal')
    run_analysis(verbose=True)


def check_seeds():
    means = []
    for seed in range(100):
        random.seed(seed)

        load_data(FORMAL_DATA_FILE, 'formal')
        load_data(INFORMAL_DATA_FILE, 'informal')
        load_data(FORMAL_DATA_FILE_BIGGER, 'formal')
        load_data(INFORMAL_DATA_FILE_BIGGER, 'informal')

        means.append(run_analysis())
    plt.figure()
    plt.hist(means)
    plt.show()


if __name__ == "__main__":
    run_single()
