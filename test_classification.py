import json
from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt


dataset_name = 'parkinson'

file_name = dataset_name+".json"
with open(file_name ,'r') as fh:
    result = json.loads(fh.read())

pt = PrettyTable()
pt.field_names = ['dataset', "SVM", "SVM with PI"
    , "KT LUPI", "Robust KT LUPI"]


rmseSVM = result['svm']
rmseSVM_PI = result['svm_pi']
rmseKT_LUPI = result['svm_kt_lupi']
rmseRobustKT_LUPI = result['svm_robust_kt_lupi']

pt.add_row(['name', np.mean(rmseSVM), np.mean(rmseSVM_PI),
            np.mean(rmseKT_LUPI), np.mean(rmseRobustKT_LUPI)])


plt.xlim([0,5])
plt.ylim([0,0.1])
plt.boxplot(rmseSVM, positions=[1])
plt.boxplot(rmseSVM_PI, positions=[2])
plt.boxplot(rmseKT_LUPI, positions=[3])
plt.boxplot(rmseRobustKT_LUPI, positions=[4])
plt.xticks([0,1, 2, 3, 4],
           ['','SVM', 'SVM(all)', 'KT_LUPI', 'R_KT_LUPI'])

plt.show()

print(pt)