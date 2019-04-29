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
from sklearn.preprocessing import StandardScaler


def load_drug_discovery_data():
    X, y = csr.readCSRFile("DDdatasets/Cathepsin_D_regr.precomp.csr", delimiter=' ')
    y_label = np.array(y)
    y_label[y_label <= 10000] = -1
    y_label[y_label > 10000] = 1

    return X, y, y_label

# multiple output dataset
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

def load_bc_data():
    # Read the training data
    # 5 features are area_mean, radius_se, texture_worst, compactness_worst, smoothness_worst.
    bc = load_breast_cancer()
    #print(bc.feature_names)
    X, y = bc['data'], bc['target']
    #return X, y
    sel_fetaures = [0, 13, 3, 2, 6, 27, 20, 7, 22, 23]
    #sel_fetaures = [27, 20, 7, 22, 23]
    X_star = X[:, sel_fetaures]
    X = np.delete(X, np.s_[sel_fetaures], axis= 1)
    return X, y, X_star


def load_boston_data():
    # Read the training data
    boston = load_boston()
    X, y = boston['data'], boston['target']

    sel_fetaures = [10, 4, 2, 5, 12]
    X_star = X[:, sel_fetaures]
    X = np.delete(X, np.s_[sel_fetaures], axis=1)

    return X, y, X_star


def load_ionosphere_data():
    data = []
    # Read the training data
    file = open('data/ionosphere.data')
    reader = csv.reader(file)

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-1] for x in data]).astype(np.float)
    print(X.shape)
    y = np.array([x[-1] for x in data])
    #x_star = np.array([x[-1] for x in data]).astype(np.float)
    y[y=='g'] = 1
    y[y == 'b'] = 0
    y = y.astype(int)
    x_star = X[:,[4,5,20,21]]
    X = np.delete(X, np.s_[4,5,20,21], axis= 1)
    del data # free up the memory
    #X = preprocessing.scale(X)
    return X, y, x_star


def load_kc2_data():
    data = []
    # Read the training data
    file = open('data/kc2_csv.csv')
    reader = csv.reader(file)
    next(reader, None)  # skip the headers
    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-1] for x in data]).astype(np.float)
    print(X.shape)
    y = np.array([x[-1] for x in data])
    '''
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    '''
    y[y=='yes'] = 1
    y[y == 'no'] = 0
    y = y.astype(int)
    x_star = X[:,14:21]
    X = X[:,0:14]
    del data # free up the memory

    return X, y, x_star

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
    del data # free up the memory

    sel_fetaures = [3, 4, 7]
    X_star = X[:, sel_fetaures]
    X = np.delete(X, np.s_[sel_fetaures], axis=1)
    
    return X, y, X_star
    #return X, y


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
    #X = preprocessing.scale(X)
    #print(X.shape)
    #return X, y, y_label
    sel_fetaures = [4, 1, 3, 0, 16]
    #sel_fetaures = [11, 20, 19, 12, 7]
    #sel_fetaures = [0, 16]

    X_star = X[:, sel_fetaures]
    X = np.delete(X, np.s_[sel_fetaures], axis=1)

    return X, y, X_star

    #return X, y


# comments: SCP works, sometimes
def load_parkinsons_data():
    data = []
    # Read the training data
    file = open('data/parkinsons.data')
    reader = csv.reader(file)
    next(reader)  # skip the headers
    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[1:] for x in data]).astype(np.float)
    y_label = np.array([x[-7] for x in data]).astype(np.float)
    y_label = y_label.astype(int)

    #delete label from the features
    X = np.delete(X, -7, axis=1)

    del data # free up the memory

    sel_fetaures = [19, 4, 10, 13, 9, 12, 11, 8, 21, 18]
    #sel_fetaures = [21, 18]
    X_star = X[:, sel_fetaures]
    X = np.delete(X, np.s_[sel_fetaures], axis=1)
    return X, y_label, X_star



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


def load_concrete_data():
    data = []
    # Read the training data
    file = open('data/Concrete_Data.csv')
    reader = csv.reader(file)

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-1] for x in data]).astype(np.float)
    y = np.array([x[-1] for x in data]).astype(np.float)
    del data # free up the memory

    sel_fetaures = [6,7]
    X_star = X[:, sel_fetaures]
    X = np.delete(X, np.s_[sel_fetaures], axis=1)

    return X, y, X_star


if __name__ == '__main__':
    #X, y, y_label = load_wpbc_data()
    #X, y, y_label = load_drug_discovery_data()
    #X, y, x_star = load_ionosphere_data()
    #X, y, x_star = load_kc2_data()
    #X, y, x_star = load_boston_data()
    #X, y, x_star = load_bc_data()
    #X, y, x_star = load_concrete_data()
    #X, y = load_bc_data()
    X, y, x_star = load_parkinsons_data()
    #print(x_star[1:10])
    print(X.shape)
    print(x_star.shape)
    X_train, X_test, y_train_label, y_test_label, train_index, test_index = \
        train_test_split(X, y, range(len(X)), test_size=.2)
    print(X_test.shape)
    print(X_train.shape)

    #print(x_star.shape)
    #print(y[1:10])
