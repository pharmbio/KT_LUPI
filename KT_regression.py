## when we have more than one privileged feature

import dataPreprocess as data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import utils as util
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.model_selection import KFold
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler

grid_param  = {'C': np.logspace(0, 3, 5),
                "gamma": np.logspace(-8, 0, 20)}


def fit_SVR(X_train, y_train, testData):
    clf = GridSearchCV(SVR(kernel='rbf'), grid_param, cv=5)
    clf.fit(X_train, y_train)
    testPred = clf.predict(testData)
    return testPred
    print("fit SVR done")


# fit LUPI with feature transformation using kernel ridge,
def fit_KRR(X_train, x_star):
    param_grid = {"alpha": np.logspace(-10, 0, 11),
                  'kernel': ['rbf'], 'gamma': np.logspace(-10, 0, 11)}
    model= GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
    model.fit(X_train, x_star)
    #print(model.best_estimator_.get_params())
    return model

# fit LUPI with feature transformation using Gaussian process regression
def fit_GPR(X_train, x_star):
    gp_kernel = PairwiseKernel(metric= 'rbf')
    model = GaussianProcessRegressor(kernel=gp_kernel)
    model.fit(X_train, x_star)
    return model


def KT_LUPI(X_train, X_star, y_train_label, X_test, regMethod='KRR'):
    n_pi = X_star.shape[1]  # numbe of privileged features
    X_mod = None
    X_test_mod = None

    for indexPI in range(n_pi):
        x_s = X_star[:, indexPI]

        if regMethod == 'GPR':
            regModel = fit_GPR(X_train, x_s)
        else:
            regModel = fit_KRR(X_train, x_s)

        y_transform = regModel.predict(X_train)
        y_test_transform = regModel.predict(X_test)

        if X_mod is None:
            X_mod = np.column_stack((X_train, y_transform))
            X_test_mod = np.column_stack((X_test, y_test_transform))
        else:
            X_mod = np.column_stack((X_mod, y_transform))
            X_test_mod = np.column_stack((X_test_mod, y_test_transform))


    scaler = StandardScaler()
    scaler.fit(X_mod)
    X_mod = scaler.transform(X_mod)
    X_test_mod = scaler.transform(X_test_mod)

    testPred = fit_SVR(X_mod, y_train_label, X_test_mod)
    return testPred


def RobustKT_LUPI(X_train, X_star, y_train_label, X_test, regMethod='KRR', n_splits=3):
    n_pi = X_star.shape[1]  # numbe of privileged features
    kf = KFold(n_splits=n_splits)
    testPred = np.zeros(len(X_test))

    for index1, index2 in kf.split(X_train, y_train_label):
        X_mod = None
        X_test_mod = None
        X_part1, X_part2 = X_train[index1], X_train[index2]
        y_part1, y_part2 = y_train_label[index1], y_train_label[index2]
        X_star_1, X_star_2 = X_star[index1], X_star[index2]

        # use part-2 for learning the transfer function
        for indexPI in range(n_pi):
            x_s = X_star_2[:, indexPI]

            if regMethod == 'GPR':
                regModel = fit_GPR(X_part2, x_s)
            else:
                regModel = fit_KRR(X_part2, x_s)

            x_s_train = regModel.predict(X_part1)
            x_s_test = regModel.predict(X_test)

            if X_mod is None:
                X_mod = np.column_stack((X_part1, x_s_train))
                X_test_mod = np.column_stack((X_test, x_s_test))
            else:
                X_mod = np.column_stack((X_mod, x_s_train))
                X_test_mod = np.column_stack((X_test_mod, x_s_test))

        scaler = StandardScaler()
        scaler.fit(X_mod)
        X_mod = scaler.transform(X_mod)
        X_test_mod = scaler.transform(X_test_mod)

        clf = GridSearchCV(SVR(kernel='rbf'), grid_param, cv=5)
        clf.fit(X_mod, y_part1)

        testPred = testPred + clf.predict(X_test_mod)

    testPred = testPred / n_splits
    return testPred


# Using feature transformation for labels..
#X, y, X_star  = data.load_concrete_data()
#X, y, X_star  = data.load_boston_data()
X, y, X_star = data.load_wine_data()

iter = 1

rmseSVM = np.zeros(iter)
rmseSVM_PI = np.zeros(iter)
rmseKT_LUPI = np.zeros(iter)
rmseRobustKT_LUPI = np.zeros(iter)

pt = PrettyTable()

pt.field_names = ["Dataset", "SVM", "SVM with PI", "KT LUPI",
                  "Robust KT LUPI"]

for i in range(iter):
    X_train, X_test, y_train, y_test, train_index, test_index = \
        train_test_split(X, y, range(len(X)), test_size=.8)
    X_star_train = X_star[train_index]
    X_star_test = X_star[test_index]
    print("train size", "test size")
    print(X_train.shape, X_test.shape)

    # normalization of the training data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    '''
    scaler = StandardScaler()
    scaler.fit(X_star_train)
    X_star_train = scaler.transform(X_star_train)
    X_star_test = scaler.transform(X_star_test)
    '''
    if 1:
        y_predicted = fit_SVR(X_train, y_train, X_test)
        print("SVM Error Rate:")
        rmseSVM[i] = util.compute_rmse(y_test, y_predicted)
        print(rmseSVM[i])

        X_train_mod = np.column_stack((X_train, X_star_train))
        X_test_mod = np.column_stack((X_test, X_star_test))
        scaler = StandardScaler()
        scaler.fit(X_train_mod)
        X_train_mod = scaler.transform(X_train_mod)
        X_test_mod = scaler.transform(X_test_mod)
        # print(X_train_mod.shape)
        y_predicted = fit_SVR(X_train_mod, y_train, X_test_mod)
        print("SVM with extra features Error Rate:")
        rmseSVM_PI[i] = util.compute_rmse(y_test, y_predicted)
        print(rmseSVM_PI[i])

    if 1:
        y_predicted = KT_LUPI(X_train, X_star_train, y_train, X_test, regMethod='KRR')

        print("Knowledge Transfer LUPI Error Rate:")
        rmseKT_LUPI[i] = util.compute_rmse(y_test, y_predicted)
        print(rmseKT_LUPI[i])

    if 1:
        y_predicted = RobustKT_LUPI(X_train, X_star_train, y_train, X_test,
                                    regMethod='KRR', n_splits=5)

        print("Robust KT LUPI Error Rate:")
        rmseRobustKT_LUPI[i] = util.compute_rmse(y_test, y_predicted)
        print(rmseRobustKT_LUPI[i])


pt.add_row(["concrete", np.mean(rmseSVM), np.mean(rmseSVM_PI),
            np.mean(rmseKT_LUPI), np.mean(rmseRobustKT_LUPI)])

print(pt)