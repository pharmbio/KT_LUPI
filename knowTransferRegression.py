import dataPreprocess as data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
import utils as util
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import PairwiseKernel
from prettytable import PrettyTable


# fit SVR on training and predict on test
# return the prediction
def fit_SVR(X_train, y_train, testData):
    grid_param = [{'kernel': ['rbf'], 'gamma': [.1,1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVR(epsilon=0.05), grid_param, cv=5)
    clf.fit(X_train, y_train)
    testPred = clf.predict(testData)
    return testPred
    print("fit SVR done")


# fit LUPI with feature transformation using kernel ridge,
def fit_KRR(X_train, x_star):
    param_grid = {"alpha": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                  'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]}
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

# fit LUPI with feature transformation, we have only one privileged feature
def KT_LUPI(X_train, y_train, x_star, X_test, regMethod='KRR'):
    if regMethod == 'GPR':
        regModel = fit_GPR(X_train, x_star)
    else:
        regModel = fit_KRR(X_train, x_star)

    y_transform = regModel.predict(X_train)
    y_test_transform = regModel.predict(X_test)
    X_mod = np.column_stack((X_train, y_transform))
    X_test_mod = np.column_stack((X_test, y_test_transform))

    scaler = StandardScaler()
    scaler.fit(X_mod)
    X_mod = scaler.transform(X_mod)
    X_test_mod = scaler.transform(X_test_mod)

    grid_param = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVR(), grid_param, cv=5)
    clf.fit(X_mod, y_train)
    testPred = clf.predict(X_test_mod)
    return testPred


# fit LUPI with  robust knowledge transformation, we have only one privileged feature
def RobustKT_LUPI(X_train, X_star, y_train, X_test, regMethod = 'KRR', n_splits=5):
    kf = KFold(n_splits=n_splits)
    testPred = np.zeros(len(X_test))

    for index1, index2 in kf.split(X_train, y_train):
        X_part1, X_part2 = X_train[index1], X_train[index2]
        y_part1, y_part2  = y_train[index1], y_train[index2]
        X_star_1, X_star_2 = X_star[index1], X_star[index2]

        # use part-2 for transfer

        if regMethod == 'GPR':
            regModel = fit_GPR(X_part2, X_star_2)
        else:
            regModel = fit_KRR(X_part2, X_star_2)

        X_star_train = regModel.predict(X_part1)
        X_star_test = regModel.predict(X_test)
        X_mod = np.column_stack((X_part1, X_star_train))
        X_test_mod = np.column_stack((X_test, X_star_test))

        scaler = StandardScaler()
        scaler.fit(X_mod)
        X_mod = scaler.transform(X_mod)
        X_test_mod = scaler.transform(X_test_mod)

        grid_param = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [1, 10, 100, 1000]}]

        clf = GridSearchCV(SVR(), grid_param, cv=5)
        clf.fit(X_mod, y_part1)
        testPred = testPred + clf.predict(X_test_mod)

    testPred = testPred/n_splits

    return testPred


X, x_star,y = data.load_energy_data()
iter = 5
rmseSVM = np.zeros(iter)
rmseSVM_PI = np.zeros(iter)
rmseKT_LUPI = np.zeros(iter)
rmseRobustKT_LUPI = np.zeros(iter)

pt = PrettyTable()
pt.field_names = ["Dataset", "SVM", "SVM with PI", "KT LUPI",
"Robust KT LUPI"]

for i in range(iter):
    X_train, X_test, y_train, y_test, train_index, test_index = \
        train_test_split(X, y, range(len(X)),test_size=.2)
    x_star_train = x_star[train_index]
    x_star_test = x_star[test_index]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if 1:
        y_predicted = fit_SVR(X_train, y_train, X_test)
        rmseSVM[i] = util.compute_rmse(y_test, y_predicted)
        print("SVM Result:")
        print(rmseSVM[i])

        X_mod = np.column_stack((X_train, x_star_train))
        X_test_mod = np.column_stack((X_test, x_star_test))
        scaler = StandardScaler()
        scaler.fit(X_mod)
        X_mod = scaler.transform(X_mod)
        X_test_mod = scaler.transform(X_test_mod)

        y_predicted = fit_SVR(X_mod, y_train, X_test_mod)
        rmseSVM_PI[i] = util.compute_rmse(y_test, y_predicted)
        print("SVM with extra features Result:")
        print(rmseSVM_PI[i])

    if 1:
        y_predicted = KT_LUPI(X_train, y_train, x_star_train, X_test, regMethod='KRR')
        rmseKT_LUPI[i] = util.compute_rmse(y_test, y_predicted)
        print("Knowledge Transfer LUPI Result:")
        print(rmseKT_LUPI[i])

    if 1:
        y_predicted = RobustKT_LUPI(X_train, x_star_train, y_train, X_test, regMethod='KRR')
        rmseRobustKT_LUPI[i] = util.compute_rmse(y_test, y_predicted)
        print("Robust Knowledge Transfer LUPI Result:")
        print(rmseRobustKT_LUPI[i])

pt.add_row(["Energy", np.mean(rmseSVM) , np.mean(rmseSVM_PI),
            np.mean(rmseKT_LUPI), np.mean(rmseRobustKT_LUPI[i])])

print(pt)