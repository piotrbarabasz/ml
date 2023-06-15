import numpy as np
from tabulate import tabulate

from scipy.stats import ttest_rel

scores = np.load("scores.npy")

table = tabulate(np.mean(scores, axis=-1),
                 tablefmt="grid",
                 headers=[
                     "KNN 3", "KNN 15",
                     "SM TL KNN 3", "TL SM KNN 3",
                     "SM KNN 3", "TL KNN 3",
                     "SM TL KNN 15", "TL SM KNN 15",
                     "SM KNN 15", "TL KNN 15",
                 ],
                 showindex=[
                     "glass", "wisconsin", "ecoli-0_vs_1",
                     "vowel0", "yeast-0-5-6-7-9_vs_4", "yeast-2_vs_4"
                 ])
print(table)

result = []
for dataset_idx in range(scores.shape[0]):
    dataset_result = []
    for classifier_idx in range(scores.shape[1]):
        classifier_result = []
        for compare_classifier_idx in range(scores.shape[1]):
            classifier_result.append(ttest_rel(
                scores[dataset_idx, classifier_idx, :],
                scores[dataset_idx, compare_classifier_idx, :]
            ))
        dataset_result.append(classifier_result)
    result.append(dataset_result)


result_array = np.array(result)

# 1 idx - dataset
# 2 idx - cls_base
# 3 idx - clas_compared
# 4 idx - ttest_result

# 0 statistic
# 1 pvalue

datasets = [
    "glass", "wisconsin", "ecoli-0_vs_1",
    "vowel0", "yeast-0-5-6-7-9_vs_4", "yeast-2_vs_4"
]
print(len(datasets))
for dataset_idx in range(len(datasets)):
    table_pvalue = tabulate((result_array[dataset_idx, :, :, 1] < 0.05).astype(int),
                     tablefmt="grid",
                     headers=[
                         "KNN 3", "KNN 15",
                         "SM TL KNN 3", "TL SM KNN 3",
                         "SM KNN 3", "TL KNN 3",
                         "SM TL KNN 15", "TL SM KNN 15",
                         "SM KNN 15", "TL KNN 15",
                     ],
                     showindex=[
                         "KNN 3", "KNN 15",
                         "SM TL KNN 3", "TL SM KNN 3",
                         "SM KNN 3", "TL KNN 3",
                         "SM TL KNN 15", "TL SM KNN 15",
                         "SM KNN 15", "TL KNN 15",
                     ])
    table_statistic = tabulate((result_array[dataset_idx, :, :, 0]),
                     tablefmt="grid",
                     headers=[
                         "KNN 3", "KNN 15",
                         "SM TL KNN 3", "TL SM KNN 3",
                         "SM KNN 3", "TL KNN 3",
                         "SM TL KNN 15", "TL SM KNN 15",
                         "SM KNN 15", "TL KNN 15",
                     ],
                     showindex=[
                         "KNN 3", "KNN 15",
                         "SM TL KNN 3", "TL SM KNN 3",
                         "SM KNN 3", "TL KNN 3",
                         "SM TL KNN 15", "TL SM KNN 15",
                         "SM KNN 15", "TL KNN 15",
                     ])
    print('Results of statistic for', datasets[dataset_idx], ' dataset')
    print(table_statistic)
    print('Results of p value for', datasets[dataset_idx], ' dataset')
    print(table_pvalue)

