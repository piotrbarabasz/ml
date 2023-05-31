import csv
import numpy as np
# Path to the CSV file\
file = 'yeast-2_vs_4.csv'
csv_file = 'dataset/' + file

exit()
data = np.genfromtxt(csv_file, delimiter=',', dtype=str, skip_header=1)
class_column_index = -1  # Index of the last column

# Update the values in the 'Class' column
for row in data:
    class_value = row[class_column_index]
    if class_value == 'positive' or class_value == ' positive':
        row[class_column_index] = '1'
    elif class_value == 'negative' or class_value == ' negative':
        row[class_column_index] = '0'

np.savetxt(file, data, delimiter=',', fmt='%s')
