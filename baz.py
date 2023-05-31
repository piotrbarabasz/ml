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

smote = SMOTE(random_state=42)
tomek_links = TomekLinks()

rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=100)

PATH_DATASET = [
    'glass.csv',
    'wisconsin.csv',
    'ecoli-0_vs_1.csv'
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

CLASSIFIERS = [
    KNeighborsClassifier(n_neighbors=3),
    KNeighborsClassifier(n_neighbors=15),
    RandomClassifier(random_state=1000)
]

# DATASETS = [
#     load_breast_cancer(return_X_y=True),
#     load_iris(return_X_y=True)
# ]

scores = np.zeros(shape=(len(DATASETS), len(CLASSIFIERS), rskf.get_n_splits()))

for dataset_idx, (X, y) in enumerate(DATASETS):
    for classifier_idx, clf_prot in enumerate(CLASSIFIERS):
        X_smote, y_smote = smote.fit_resample(X, y)
        X_tomek, y_tomek = tomek_links.fit_resample(X_smote, y_smote)
        for fold_idx, (train, test) in enumerate(rskf.split(X_tomek, y_tomek)):
            clf = clone(clf_prot)
            clf.fit(X_tomek[train], y_tomek[train])
            y_tomek_pred = clf.predict(X_tomek[test])
            score = accuracy_score(y_tomek[test], y_tomek_pred)
            scores[dataset_idx, classifier_idx, fold_idx] = score
# print('######')
# print(np.mean(scores, axis=-1))

# print("Original dataset:")
# print(len(X))
# print(len(y))
# print("Smote dataset:")
# print(len(X_smote))
# print(len(y_smote))
# print("Tomek-links dataset:")
# print(len(X_tomek))
# print(len(y_tomek))

#
plt.figure(figsize=(7.50, 3.50))
plt.title("Imbalanced dataset", fontsize="12")
plt.scatter(X[:, 0], X[:, -1], marker="o", c=y, s=40, edgecolor="k")

plt.figure(figsize=(7.50, 3.50))
plt.title("Oversampled by SMOTE", fontsize="12")
plt.scatter(X_smote[:, 0], X_smote[:, -1], marker="o", c=y_smote, s=40, edgecolor="k")

plt.figure(figsize=(7.50, 3.50))
plt.title("Undersampled by Tomek-Links", fontsize="12")
plt.scatter(X_tomek[:, 0], X_tomek[:, -1], marker="o", c=y_tomek, s=40, edgecolor="k")
# plt.show()

np.save("scores", scores)
