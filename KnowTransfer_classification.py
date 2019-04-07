import dataPreprocess as data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
import utils as util
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.model_selection import StratifiedKFold


def fit_SVM(X_train, y_train, X_test):
    grid_param = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [.1, 1, 10, 100, 1000]}]

    clf = GridSearchCV(SVC(), grid_param, cv=5)
    clf.fit(X_train, y_train)
    testPred = clf.predict(X_test)
    return testPred

def fit_LinSVM(X_train, y_train, X_test):
    grid_param = {'C': [1, 10, 100, 1000]}
    clf = GridSearchCV(LinearSVC(), grid_param, cv=5)
    clf.fit(X_train, y_train)
    testPred = clf.predict(X_test)
    return testPred


def KT_LUPI(X_train, y_train, y_train_label, X_test):
    gp_kernel = PairwiseKernel(metric= 'rbf')

    gpr = GaussianProcessRegressor(kernel=gp_kernel)
    gpr.fit(X_train, y_train)
    y_transform = gpr.predict(X_train)
    y_test_transform = gpr.predict(X_test)
    X = np.column_stack((X_train, y_transform))
    test_data = np.column_stack((X_test, y_test_transform))

    grid_param = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVC(), grid_param, cv=5)
    clf.fit(X, y_train_label)
    testPred = clf.predict(test_data)
    return testPred


def RobustKT_LUPI(X_train, X_star, y_train_label, X_test, n_splits=3):

    kf = StratifiedKFold(n_splits=n_splits)
    testPred = np.zeros((len(X_test),2))

    for index1, index2 in kf.split(X_train, y_train_label):
        X_part1, X_part2 = X_train[index1], X_train[index2]
        y_part1, y_part2  = y_train_label[index1], y_train_label[index2]
        X_star_1, X_star_2= X_star[index1], X_star[index2]

        # use part-2 for transfer
        gp_kernel = PairwiseKernel(metric= 'rbf')
        gpr = GaussianProcessRegressor(kernel=gp_kernel)
        gpr.fit(X_part2, X_star_2)
        X_star_train = gpr.predict(X_part1)
        X_star_test = gpr.predict(X_test)
        X = np.column_stack((X_part1, X_star_train))
        test_data = np.column_stack((X_test, X_star_test))

        grid_param = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [1, 10, 100, 1000]}]

        clf = GridSearchCV(SVC(probability=True), grid_param, cv=5)
        clf.fit(X, y_part1)
        testPred = testPred + clf.predict_proba(test_data)

    testPred = testPred/n_splits
    testPred = np.argmax(testPred, axis=1)
    return testPred


def LUPI(X_train, y_train, y_train_label, X_test):
    gp_kernel = PairwiseKernel(metric= 'rbf')
    gpr = GaussianProcessRegressor(kernel=gp_kernel)
    gpr.fit(X_train, y_train)
    #y_transform = gpr.predict(X_train)
    y_test_transform = gpr.predict(X_test)
    #X = np.column_stack((X_train, y_transform))
    X = np.column_stack((X_train, y_train))
    test_data = np.column_stack((X_test, y_test_transform))

    grid_param = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVC(), grid_param, cv=5)
    clf.fit(X, y_train_label)
    testPred = clf.predict(test_data)
    return testPred

# Using feature transformation for labels..
#X, y, y_label = data.load_gridStability_data()
#X, y, y_label = data.load_PD_data() # does not work
X, y, y_label = data.load_wine_data()
#X, y, y_label = data.load_wpbc_data() # not useful

X_train, X_test, y_train_label, y_test_label, train_index, test_index = \
    train_test_split(X, y_label, range(len(X)),test_size=.2, random_state=3, stratify=y_label)
y_train = y[train_index]
y_test = y[test_index]
print("train size", "test size")
print(len(X_train), len(X_test))

if 1:
    y_predicted = fit_SVM(X_train, y_train_label, X_test)
    print("SVM Error Rate:")
    errRate = util.compute_errorRate(y_test_label, y_predicted)
    print(errRate)


    X = np.column_stack((X_train, y_train))
    X_test_mod = np.column_stack((X_test, y_test))
    y_predicted = fit_SVM(X, y_train_label, X_test_mod)


    print("SVM with extra features Error Rate:")
    errRate = util.compute_errorRate(y_test_label, y_predicted)
    print(errRate)

if 1:
    y_predicted = KT_LUPI(X_train, y_train, y_train_label, X_test)

    print("Knowledge Transfer LUPI Error Rate:")
    errRate = util.compute_errorRate(y_test_label, y_predicted)
    print(errRate)

if 1:
    y_predicted = RobustKT_LUPI(X_train, y_train, y_train_label, X_test)

    print("Robust KT LUPI Error Rate:")
    errRate = util.compute_errorRate(y_test_label, y_predicted)
    print(errRate)

if 0:
    y_predicted = LUPI(X_train, y_train, y_train_label, X_test)
    correct = sum(y_test_label == y_predicted)
    print("LUPI without transformation while training Result:")
    print(correct/len(y_test_label))

