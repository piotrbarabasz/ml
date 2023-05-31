import numpy as np
from tabulate import tabulate

from scipy.stats import ttest_rel

scores = np.load("scores.npy")

table = tabulate(np.mean(scores, axis=-1),
                 tablefmt="grid",
                 headers=[
                     "KNN 3", "KNN 15", "RC",
                     "SM TL KNN3", "TL SM KNN3",
                     "SM KNN 3", "TL KNN3"
                 ],
                 showindex=[
                     "glass", "wisconsin", "ecoli-0_vs_1",
                     "vowel0", "yeast-0-5-6-7-9_vs_4", "yeast-2_vs_4"
                 ])

print(table)

for dataset_idx in range(np.shape(scores)[0]):
    for classifier_idx in range(np.shape(scores)[1] - 1):
        for next_classifier_idx in range(classifier_idx + 1, np.shape(scores)[1]):
            print(dataset_idx, classifier_idx, ' compare to ', dataset_idx, next_classifier_idx)
            result = ttest_rel(
                scores[dataset_idx, classifier_idx, :],
                scores[dataset_idx, next_classifier_idx, :]
            )
            print('result.statistic', result.statistic)
            print('result.pvalue ', result.pvalue)
