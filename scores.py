import numpy as np
from tabulate import tabulate

from scipy.stats import ttest_rel, wilcoxon

scores = np.load("scores.npy")
# print('scores', scores)
# print(np.mean(scores, axis=-1))
# exit()


table = tabulate(np.mean(scores, axis=-1),
                 tablefmt="grid",
                 headers=["KNN 3", "KNN 15", "RC"],
                 showindex=["glass", "wisconsin", "ecoli-0_vs_1"])

print(table)
# exit()
# print(scores[0, 0, :])
# print(scores[0, 1, :])
# print(scores[1, 0, :])
# print(scores[1, 1, :])
# exit()

result = ttest_rel(scores[0, 0, :], scores[0, 1, :])
print(result.statistic)
print(result.pvalue)
# exit()

# print(table)
