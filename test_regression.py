import json
from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt


dataset_name = 'synthetic'

file_name = dataset_name+".json"
with open(file_name ,'r') as fh:
    result = json.loads(fh.read())

pt = PrettyTable()
pt.field_names = ["Train Size", "SVM", "SVM with PI"
    , "KT LUPI", "Robust KT LUPI"]

for M, r in result.items():
    rmseSVM = list()
    rmseSVM_PI = list()
    rmseKT_LUPI = list()
    rmseRobustKT_LUPI = list()

    #rmseSVM.append(r['svm'])
    #rmseSVM_PI.append(r['svm_pi'])
    rmseKT_LUPI.append(r['svm_kt_lupi'])
    rmseRobustKT_LUPI.append(r['svm_robust_kt_lupi'])
    pt.add_row([M, np.mean(rmseSVM), np.mean(rmseSVM_PI),
                np.mean(rmseKT_LUPI), np.mean(rmseRobustKT_LUPI)])


    plt.xlim([0,3])
    #plt.ylim([0.5,2])
    #plt.boxplot(rmseSVM, positions=[1])
    #plt.boxplot(rmseSVM_PI, positions=[2])
    plt.boxplot(rmseKT_LUPI, positions=[1])
    plt.boxplot(rmseRobustKT_LUPI, positions=[2])
    plt.xticks([0,1, 2],
               ['','KT-LUPI', 'RKT-LUPI'])

    plt.show()

print(pt)