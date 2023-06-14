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


print(tabulate(scores))
# index 1. datasets
# index 2. classificators
print(scores[0, 0, :])
print(scores[1, 0, :])
print(scores.shape)
result = []
for dataset_idx in range(scores.shape[0]):
    # dataset_idx = 0
    # print('dataset: ', dataset_idx)
    for classifier_idx in range(scores.shape[1]):
        for compare_classifier_idx in range(scores.shape[1]):
            # print(classifier_idx, ' : ', compare_classifier_idx)
            result.append(ttest_rel(
                scores[dataset_idx, classifier_idx, :],
                scores[dataset_idx, compare_classifier_idx, :]
            ))

for i in range(len(result)):
    print(result[i].statistic)
# print(result.statistic)
# print(result.pvalue)
# print(tabulate(result))
exit()

result = np.zeros((6, 10))
for x_index in range(6):
    for y_index in range(9):
        ttest_result = ttest_rel(scores[x_index, y_index, :], scores[x_index, y_index + 1, :])
        result[x_index, y_index] = ttest_result.pvalue

print(tabulate(result))
print(result.shape)  # Output: (6, 10)


exit()
print(table)
res_statistic = []
res_pvalue = []
for dataset_idx in range(np.shape(scores)[0]):
    for classifier_idx in range(np.shape(scores)[1] - 1):
        for next_classifier_idx in range(classifier_idx + 1, np.shape(scores)[1]):
            # print(dataset_idx, classifier_idx, ' compare to ', dataset_idx, next_classifier_idx)
            result = ttest_rel(
                scores[dataset_idx, classifier_idx, :],
                scores[dataset_idx, next_classifier_idx, :]
            )
            res_statistic.append(result.statistic < 0.05)
            res_pvalue.append(result.pvalue < 0.05)
            # ktores z tych

            # print(result)
            # print('result.statistic', result.statistic)
            # print('result.pvalue ', result.pvalue)
# print('result.statistic', result.statistic)
# print('result.pvalue ', result.pvalue)
print(res_pvalue)
print(res_statistic)