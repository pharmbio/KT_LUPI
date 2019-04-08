"""
 Author: Niharika gauraha
 Synergy Conformal Prediction Using Random Forest Classifier
"""

import csv
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pydataset import data
from sklearn.datasets import load_boston
import fileUtils.csrFormat as csr



def load_drug_discovery_data():
    X, y = csr.readCSRFile("DDdatasets/Cathepsin_D_regr.precomp.csr", delimiter=' ')
    y_label = np.array(y)
    y_label[y_label <= 10000] = -1
    y_label[y_label > 10000] = 1

    return X, y, y_label

def load_energy_data():
    data = []
    # Read the training data
    file = open('data/ENB2012_data.csv')
    reader = csv.reader(file)

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-2] for x in data]).astype(np.float)
    y = np.array([x[-2] for x in data]).astype(np.float)
    x_star = np.array([x[-1] for x in data]).astype(np.float)

    del data # free up the memory
    #X = preprocessing.scale(X)

    print(X.shape)
    return X, y, x_star




# comment: SCP works fine with wine data
def load_wine_data():
    data = []
    # Read the training data
    file = open('data/winequality-white.csv')
    reader = csv.reader(file)

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-2] for x in data]).astype(np.float)
    y = np.array([x[-2] for x in data]).astype(np.float)
    y_label = np.array([x[-1] for x in data]).astype(np.float)
    y_label = y_label.astype(int)

    # consider only dataset with digits 2 and 3
    y5 = y_label[y_label == 5]
    y6 = y_label[y_label == 6]
    X5 = X[y_label == 5]
    X6 = X[y_label == 6]

    y_label = np.concatenate((y5, y6))
    y_label[y_label == 5] = 1
    y_label[y_label == 6] = 0
    X = np.concatenate((X5, X6), axis=0)
    print(X.shape)

    del data # free up the memory
    X = preprocessing.scale(X)
    y = preprocessing.scale(y)
    print(X.shape)
    return X, y, y_label


# comments: SCP works, sometimes
def load_PD_data():
    data = []
    # Read the training data
    file = open('data/train_data.txt')
    reader = csv.reader(file)

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[1:-2] for x in data]).astype(np.float)
    y = np.array([x[-2] for x in data]).astype(np.float)
    y_label = np.array([x[-1] for x in data]).astype(np.float)
    y_label = y_label.astype(int)

    del data # free up the memory
    X = preprocessing.scale(X)
    print(X.shape)
    return X, y, y_label


def load_gridStability_data():
    data = []
    # Read the training data
    file = open('data/grid_stability.csv')
    reader = csv.reader(file)

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-2] for x in data]).astype(np.float)
    y = np.array([x[-2] for x in data]).astype(np.float)
    y_label = np.array([x[-1] for x in data])
    y_label[y_label == 'stable'] = 1
    y_label[y_label == 'unstable'] = 0
    y_label = y_label.astype(int)

    del data # free up the memory

    #X = preprocessing.scale(X)

    print(X.shape)
    return X, y, y_label


def load_CBM_data():
    data = []
    # Read the training data
    file = open('data/CBM_data.txt')
    reader = csv.reader(file, delimiter=',')
    #next(reader)
    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[1:-2] for x in data]).astype(np.float)
    y = np.array([x[-1] for x in data]).astype(np.float)
    del data # free up the memory
    #X = preprocessing.scale(X)
    #y = preprocessing.scale(y)
    print(X.shape)
    return X, y


def load_wpbc_data():
    data = []
    # Read the training data
    file = open('data/wpbc.data')
    reader = csv.reader(file, delimiter=',')

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[3:-3] for x in data]).astype(float)
    y = np.array([x[2] for x in data]).astype(float)
    y_label = np.array([x[1] for x in data])
    y_label[y_label == 'R'] = 1
    y_label[y_label == 'N'] = 0
    y_label = y_label.astype(int)
    del data # free up the memory
    #X = preprocessing.scale(X)
    #y = preprocessing.scale(y)
    print(X.shape)
    return X, y, y_label

if __name__ == '__main__':
    #X, y, y_label = load_wpbc_data()
    X, y, y_label = load_drug_discovery_data()
    print(y[1:10])
    print(y_label[1:10])