import dataPreprocess as data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
import utils as util
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.model_selection import StratifiedKFold
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler


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

# fit LUPI with feature transformation using kernel ridge,
def fit_KRR(X_train, x_star):
    param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                  'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]}
    model= GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
    model.fit(X_train, x_star)
    return model

# fit LUPI with feature transformation using Gaussian process regression
def fit_GPR(X_train, x_star):
    gp_kernel = PairwiseKernel(metric= 'rbf')
    model = GaussianProcessRegressor(kernel=gp_kernel)
    model.fit(X_train, x_star)
    return model


def KT_LUPI(X_train, y_train, y_train_label, X_test, regMethod = 'KRR'):
    if regMethod == 'GPR':
        regModel = fit_GPR(X_train, y_train)
    else:
        regModel = fit_KRR(X_train, y_train)

    y_transform = regModel.predict(X_train)
    y_test_transform = regModel.predict(X_test)
    X = np.column_stack((X_train, y_transform))
    test_data = np.column_stack((X_test, y_test_transform))

    # Construct the decision rule
    grid_param = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVC(), grid_param, cv=5)
    clf.fit(X, y_train_label)
    testPred = clf.predict(test_data)
    return testPred


def RobustKT_LUPI(X_train, X_star, y_train_label, X_test, regMethod = 'KRR', n_splits=3):

    kf = StratifiedKFold(n_splits=n_splits)
    testPred = np.zeros((len(X_test),2))

    for index1, index2 in kf.split(X_train, y_train_label):
        X_part1, X_part2 = X_train[index1], X_train[index2]
        y_part1, y_part2  = y_train_label[index1], y_train_label[index2]
        X_star_1, X_star_2= X_star[index1], X_star[index2]

        # use part-2 for transfer

        if regMethod == 'GPR':
            regModel = fit_GPR(X_part2, X_star_2)
        else:
            regModel = fit_KRR(X_part2, X_star_2)

        X_star_train = regModel.predict(X_part1)
        X_star_test = regModel.predict(X_test)
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



# Using feature transformation for labels..
#X, y, y_label = data.load_gridStability_data()
#X, y, y_label = data.load_PD_data() # does not work
X, y, y_label = data.load_wine_data()
#X, y, y_label = data.load_wpbc_data() # not useful
#X, y, y_label = data.load_drug_discovery_data()


iter = 2

errRateSVM = np.zeros(iter)
errRateSVM_PI = np.zeros(iter)
errRateKT_LUPI = np.zeros(iter)
errRateRobustKT_LUPI = np.zeros(iter)

pt = PrettyTable()

pt.field_names = ["Dataset", "SVM", "SVM with PI", "KT LUPI",
"Robust KT LUPI"]

for i in range(iter):
    X_train, X_test, y_train_label, y_test_label, train_index, test_index = \
        train_test_split(X, y_label, range(len(X)), test_size=.25, stratify=y_label)
    y_train = y[train_index]
    y_test = y[test_index]

    # normalization of the training data
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)

    if 0:
        y_predicted = fit_SVM(X_train, y_train_label, X_test)
        print("SVM Error Rate:")
        errRateSVM[i] = util.compute_errorRate(y_test_label, y_predicted)

        # SVM with standard features and PI
        X_train_mod = np.column_stack((X_train, y_train))
        X_test_mod = np.column_stack((X_test, y_test))
        y_predicted = fit_SVM(X_train_mod, y_train_label, X_test_mod)
        print("SVM with extra features Error Rate:")
        errRateSVM_PI[i] = util.compute_errorRate(y_test_label, y_predicted)


    if 1:
        y_predicted = KT_LUPI(X_train, y_train, y_train_label, X_test, regMethod='KRR')

        print("Knowledge Transfer LUPI Error Rate:")
        errRateKT_LUPI[i] = util.compute_errorRate(y_test_label, y_predicted)
        print(errRateKT_LUPI[i])

    if 1:
        y_predicted = RobustKT_LUPI(X_train, y_train, y_train_label, X_test, regMethod='KRR')

        print("Robust KT LUPI Error Rate:")
        errRateRobustKT_LUPI[i] = util.compute_errorRate(y_test_label, y_predicted)
        print(errRateRobustKT_LUPI[i])



pt.add_row(["Wine", np.mean(errRateSVM) , np.mean(errRateSVM_PI),
            np.mean(errRateKT_LUPI), np.mean(errRateRobustKT_LUPI)])

print(pt)