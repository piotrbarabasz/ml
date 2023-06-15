import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.base import clone

from random_classifier import RandomClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline

smote = SMOTE(random_state=42)
tomek_links = TomekLinks()

rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=1410)

# datasets from http://keel.es/
PATH_DATASET = [
    # Imbalance ratio between 1.5 and 9
    'glass.csv',
    'wisconsin.csv',
    'ecoli-0_vs_1.csv',
    # Imbalance ratio higher than 9 - Part I
    'vowel0.csv',
    'yeast-0-5-6-7-9_vs_4.csv',
    'yeast-2_vs_4.csv',
]

DATASETS = []

for csv_file in PATH_DATASET:
    with open('dataset/' + csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)

        data = []
        target = []
        for row in reader:
            data_row = [float(value) for value in row[:-1]]
            target_value = int(row[-1].strip())

            data.append(data_row)
            np_data = np.array(data)

            target.append(target_value)
            np_target = np.array(target)

    dataset_tuple = (np_data, np_target)
    DATASETS.append(dataset_tuple)

smote_tomek_knn3_ppl = Pipeline([
    ('smote', SMOTE(sampling_strategy='auto')),
    ('tomek', TomekLinks()),
    ('classification', KNeighborsClassifier(n_neighbors=3))
])

tomek_smote_knn3_ppl = Pipeline([
    ('tomek', TomekLinks()),
    ('smote', SMOTE(sampling_strategy='auto')),
    ('classification', KNeighborsClassifier(n_neighbors=3))
])

smote_knn3_ppl = Pipeline([
    ('smote', SMOTE(sampling_strategy='auto')),
    ('classification', KNeighborsClassifier(n_neighbors=3))
])

tomek_knn3_ppl = Pipeline([
    ('tomek', TomekLinks()),
    ('classification', KNeighborsClassifier(n_neighbors=3))
])

smote_tomek_knn15_ppl = Pipeline([
    ('smote', SMOTE(sampling_strategy='auto')),
    ('tomek', TomekLinks()),
    ('classification', KNeighborsClassifier(n_neighbors=15))
])

tomek_smote_knn15_ppl = Pipeline([
    ('tomek', TomekLinks()),
    ('smote', SMOTE(sampling_strategy='auto')),
    ('classification', KNeighborsClassifier(n_neighbors=15))
])

smote_knn15_ppl = Pipeline([
    ('smote', SMOTE(sampling_strategy='auto')),
    ('classification', KNeighborsClassifier(n_neighbors=15))
])

tomek_knn15_ppl = Pipeline([
    ('tomek', TomekLinks()),
    ('classification', KNeighborsClassifier(n_neighbors=15))
])

CLASSIFIERS = [
    KNeighborsClassifier(n_neighbors=3),
    KNeighborsClassifier(n_neighbors=15),
    # RandomClassifier(random_state=1000),
    smote_tomek_knn3_ppl,
    tomek_smote_knn3_ppl,
    smote_knn3_ppl,
    tomek_knn3_ppl,
    smote_tomek_knn15_ppl,
    tomek_smote_knn15_ppl,
    smote_knn15_ppl,
    tomek_knn15_ppl
]

# print(pipeline)
# print(CLASSIFIERS)
# exit()
# DATASETS = [
#     load_breast_cancer(return_X_y=True),
#     load_iris(return_X_y=True)
# ]

scores = np.zeros(shape=(len(DATASETS), len(CLASSIFIERS), rskf.get_n_splits()))


def printDatasetsXy(dataset_idx):
    print('dataset_idx', dataset_idx)
    print("Original dataset:")
    print(len(X))
    print(len(y))
    print("Smote dataset:")
    print(len(X_smote))
    print(len(y_smote))
    print("Tomek-links dataset:")
    print(len(X_tomek))
    print(len(y_tomek))


def showScatters(dataset_idx):
    plt.figure(figsize=(7.50, 3.50))
    plt.title("Imbalanced dataset " + str(dataset_idx + 1), fontsize="12")
    plt.scatter(X[:, 0], X[:, -1], marker="o", c=y, s=40, edgecolor="k")

    # plt.figure(figsize=(7.50, 3.50))
    # plt.title("Oversampled by SMOTE " + str(dataset_idx + 1), fontsize="12")
    # plt.scatter(X_smote[:, 0], X_smote[:, -1], marker="o", c=y_smote, s=40, edgecolor="k")
    #
    # plt.figure(figsize=(7.50, 3.50))
    # plt.title("Undersampled by Tomek-Links " + str(dataset_idx + 1), fontsize="12")
    # plt.scatter(X_tomek[:, 0], X_tomek[:, -1], marker="o", c=y_tomek, s=40, edgecolor="k")

    plt.show()


for dataset_idx, (X, y) in enumerate(DATASETS):
    for classifier_idx, clf_prot in enumerate(CLASSIFIERS):
        # X_smote, y_smote = smote.fit_resample(X, y)
        # X_tomek, y_tomek = tomek_links.fit_resample(X_smote, y_smote)
        for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
            clf = clone(clf_prot)
            clf.fit(X[train], y[train])
            y_tomek_pred = clf.predict(X[test])
            score = accuracy_score(y[test], y_tomek_pred)
            scores[dataset_idx, classifier_idx, fold_idx] = score
        # print(classifier_idx)
#         printDatasetsXy(dataset_idx)
#     showScatters(dataset_idx)
print(np.mean(scores, axis=-1), np.shape(scores))

np.save("scores", scores)
