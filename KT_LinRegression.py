## when we have more than one privileged feature

import dataPreprocess as data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import GridSearchCV
import utils as util
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.model_selection import KFold
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from collections import Counter


grid_param  = {'C': np.logspace(-1, 3, 5),
                "gamma": np.logspace(-8, 0, 20)}


linear_grid_param  = {'C': np.logspace(0, 3, 5)}


def fit_LinearSVR(X_train, y_train, testData=None):
    clf = GridSearchCV(LinearSVR(), linear_grid_param, cv=5)
    clf.fit(X_train, y_train)

    if testData is None:
        return clf

    testPred = clf.predict(testData)
    return testPred
    print("fit Linear SVR done")


def fit_SVR(X_train, y_train, testData):
    clf = GridSearchCV(SVR(kernel='rbf'), grid_param, cv=5)
    clf.fit(X_train, y_train)
    testPred = clf.predict(testData)
    return testPred
    print("fit SVR done")


# fit LUPI with feature transformation using ridge regression (RR),
def fit_RR(X_train, x_star):
    param_grid = {"alpha": np.logspace(-10, 0, 11)}
    model= GridSearchCV(Ridge(), cv=5, param_grid=param_grid)
    model.fit(X_train, x_star)
    #print(model.best_estimator_.get_params())
    return model



def KT_LUPI(X_train, X_star, y_train_label, X_test, regMethod='Linear'):
    n_pi = X_star.shape[1]  # numbe of privileged features
    X_mod = None
    X_test_mod = None

    for indexPI in range(n_pi):
        x_s = X_star[:, indexPI]

        if regMethod == 'Linear':
            regModel = fit_RR(X_train, x_s)
        else:
            regModel = fit_LinearSVR(X_train, x_s)

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


def RobustKT_LUPI(X_train, X_star, y_train_label, X_test, regMethod='Linear', n_splits=5):
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

            if regMethod == 'Linear':
                regModel = fit_RR(X_part2, x_s)
            else:
                regModel = fit_LinearSVR(X_part2, x_s)

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

        clf = GridSearchCV(SVR(), grid_param, cv=5)
        clf.fit(X_mod, y_part1)

        testPred = testPred + clf.predict(X_test_mod)

    testPred = testPred / n_splits
    return testPred



def run_experiments(X_train, y_train, X_test, y_test,
                    X_star_train, X_star_test):
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
    results = OrderedDict()

    if 1:
        y_predicted = fit_SVR(X_train, y_train, X_test)
        results["svm"] = [util.compute_rmse(y_test, y_predicted)]

        X_train_mod = np.column_stack((X_train, X_star_train))
        X_test_mod = np.column_stack((X_test, X_star_test))
        scaler = StandardScaler()
        scaler.fit(X_train_mod)
        X_train_mod = scaler.transform(X_train_mod)
        X_test_mod = scaler.transform(X_test_mod)
        # print(X_train_mod.shape)
        y_predicted = fit_SVR(X_train_mod, y_train, X_test_mod)
        results["svm_pi"] = [util.compute_rmse(y_test, y_predicted)]

    if 1:
        y_predicted = KT_LUPI(X_train, X_star_train, y_train, X_test)

        results["svm_kt_lupi"] = [util.compute_rmse(y_test, y_predicted)]

    if 1:
        y_predicted = RobustKT_LUPI(X_train, X_star_train, y_train, X_test)
        results["svm_robust_kt_lupi"] = [util.compute_rmse(y_test, y_predicted)]

    return results


if __name__ == '__main__':
    # X, y, X_star  = data.load_concrete_data()
    # X, y, X_star  = data.load_boston_data()
    #X, y, X_star = data.load_wine_data()
    X, y, X_star = data.load_PD_data()

    iter = 20

    dataset_name = 'PD'

    '''
    rmseSVM = np.zeros(iter)
    rmseSVM_PI = np.zeros(iter)
    rmseKT_LUPI = np.zeros(iter)
    rmseRobustKT_LUPI = np.zeros(iter)
    '''

    rmseSVM = list()
    rmseSVM_PI = list()
    rmseKT_LUPI = list()
    rmseRobustKT_LUPI = list()

    pt = PrettyTable()

    pt.field_names = ["Train Size", "SVM", "SVM with PI", "KT LUPI",
                      "Robust KT LUPI"]

    train_size = [200, 300, 400, 500, 600]
    #train_size = [100, 200]

    dictResult = OrderedDict()
    #counterResult = Counter()
    result = OrderedDict()

    for i in range(iter):
        X_remain, X_test, y_remain, y_test, train_index, test_index = \
            train_test_split(X, y, range(len(X)), test_size=.2)
        X_star_remain = X_star[train_index]
        X_star_test = X_star[test_index]
        print("train size", "test size")


        for n in train_size:
            X_left, X_train, y_left, y_train, train_index, test_index = \
                train_test_split(X_remain, y_remain, range(len(X_remain)), test_size=n/len(X_remain))
            X_star_train = X_star[test_index]
            print("train size", "test size")
            print(X_train.shape, X_test.shape)

            dictResult[n] = run_experiments(X_train, y_train, X_test, y_test,
                    X_star_train, X_star_test)


        #results['{0}'.format(i)] = result
        for key, value in dictResult.items():
            if i==0:
                result[key] = value
            else:
                for key2, value2 in value.items():
                    result[key][key2].extend(value2) #concatenate lists

        #counterResult = counterResult + Counter(dictResult)
        #for key, value in result.items():
        #    result[key] = value / iter

    import json
    file_name = dataset_name+".json"
    with open(file_name, 'w') as fh:
        fh.write(json.dumps(result))

    '''
    file_name = dataset_name+".json"
    with open(file_name ,'r') as fh:
        result = json.loads(fh.read())

    for M, r in result.items():
        rmseSVM = list()
        rmseSVM_PI = list()
        rmseKT_LUPI = list()
        rmseRobustKT_LUPI = list()

        rmseSVM.append(r['svm'])
        rmseSVM_PI.append(r['svm_pi'])
        rmseKT_LUPI.append(r['svm_kt_lupi'])
        rmseRobustKT_LUPI.append(r['svm_robust_kt_lupi'])
        pt.add_row([M, np.mean(rmseSVM)/iter, np.mean(rmseSVM_PI)/iter,
                                np.mean(rmseKT_LUPI)/iter, np.mean(rmseRobustKT_LUPI)/iter])

                    #, np.mean(rmseRobustKT_LUPI)
#pt.add_row(["concrete", np.mean(rmseSVM), np.mean(rmseSVM_PI),
#            np.mean(rmseKT_LUPI), 'NA'])
    '''
#print(pt)