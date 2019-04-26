## when we have more than one privileged feature

import dataPreprocess as data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.model_selection import GridSearchCV
import utils as util
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.model_selection import StratifiedKFold
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV


cv = 6

grid_param  = {'C': np.logspace(-5, 5, 21, base = 2),
                "gamma": np.logspace(-6, 6, 25, base = 2)}

def fit_SVM(X_train, y_train, X_test):
    #grid_param = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4],
    #               'C': [.1, 1, 10, 100]}]

    clf = GridSearchCV(SVC(kernel='rbf'), grid_param, cv=cv)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    testPred = clf.predict(X_test)
    return testPred

def fit_LinSVM(X_train, y_train, X_test):
    grid_param = {'C': [1, 10, 100, 1000]}
    clf = GridSearchCV(LinearSVC(), grid_param, cv=cv)
    clf.fit(X_train, y_train)
    testPred = clf.predict(X_test)
    return testPred

# fit LUPI with feature transformation using kernel ridge,
def fit_KRR(X_train, y_label):
    param_grid = {"alpha": np.logspace(-8, 0, 10),
                  'kernel': ['rbf'], 'gamma': np.logspace(-8, 0, 10, base = 2)}
    model= GridSearchCV(KernelRidge(), cv=cv, param_grid=param_grid)
    model.fit(X_train, y_label)
    return model

def fit_RR(X_train, y_label):
    #param_grid = {"alpha": np.logspace(-10, 0, 11, base = 2)}
    model = RidgeCV(alphas=np.logspace(-10, 0, 10, base = 2), cv=cv).fit(X_train, y_label)
    #model= GridSearchCV(Ridge(), cv=cv, param_grid=param_grid)
    #model.fit(X_train, y_label)
    #print(model.best_estimator_.get_params())
    return model

# fit LUPI with feature transformation using Gaussian process regression
def fit_GPR(X_train, x_star):
    gp_kernel = PairwiseKernel(metric= 'rbf')
    model = GaussianProcessRegressor(kernel=gp_kernel)
    model.fit(X_train, x_star)
    return model

# fit LUPI with feature transformation using kernel ridge,
def fit_SVR(X_train, x_star):
    model= GridSearchCV(SVR(kernel='rbf'), cv=cv, param_grid=grid_param)
    model.fit(X_train, x_star)
    return model

def KT_LUPI(X_train, X_star, y_train_label, X_test, regMethod = 'Linear'):
    n_pi = X_star.shape[1] #numbe of privileged features
    X_mod = None
    X_test_mod = None

    for indexPI in range(n_pi):
        x_s = X_star[:,indexPI]

        if regMethod == 'Linear':
            regModel = fit_RR(X_train, x_s)
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
    
    testPred = fit_SVM(X_mod, y_train_label, X_test_mod)
    return testPred


def RobustKT_LUPI(X_train, X_star, y_train_label, X_test, regMethod = 'Linear', n_splits=3):
    n_pi = X_star.shape[1]  # numbe of privileged features
    kf = StratifiedKFold(n_splits=n_splits)
    testPred = np.zeros((len(X_test),2))

    for index1, index2 in kf.split(X_train, y_train_label):
        X_mod = None
        X_test_mod = None
        X_part1, X_part2 = X_train[index1], X_train[index2]
        y_part1, y_part2  = y_train_label[index1], y_train_label[index2]
        X_star_1, X_star_2= X_star[index1], X_star[index2]

        # use part-2 for learning the transfer function
        for indexPI in range(n_pi):
            x_s = X_star_2[:, indexPI]

            if regMethod == 'Linear':
                regModel = fit_RR(X_part2, x_s)
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
        
        clf = GridSearchCV(SVC(probability=True), grid_param, cv=5)
        clf.fit(X_mod, y_part1)

        testPred = testPred + clf.predict_proba(X_test_mod)

    testPred = testPred/n_splits
    testPred = np.argmax(testPred, axis=1)
    return testPred



# Using feature transformation for labels..
#X, y, y_label = data.load_gridStability_data()
#X, y, y_label = data.load_PD_data() # does not work
#X, y, y_label = data.load_wine_data()
#X, y, y_label = data.load_wpbc_data() # not useful
#X, y, y_label = data.load_drug_discovery_data()

#X, y_label, X_star = data.load_ionosphere_data()
#X, y_label, X_star = data.load_kc2_data()
#X, y_label, X_star = data.load_bc_data()
X, y_label, X_star = data.load_parkinsons_data()

iter = 1

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
        y_predicted = fit_SVM(X_train, y_train_label, X_test)
        print("SVM Error Rate:")
        errRateSVM[i] = util.compute_errorRate(y_test_label, y_predicted)
        print(errRateSVM[i])

        X_train_mod = np.column_stack((X_train, X_star_train))
        X_test_mod = np.column_stack((X_test, X_star_test))

        scaler = StandardScaler()
        scaler.fit(X_train_mod)
        X_train_mod = scaler.transform(X_train_mod)
        X_test_mod = scaler.transform(X_test_mod)

        #print(X_train_mod.shape)
        y_predicted = fit_SVM(X_train_mod, y_train_label, X_test_mod)
        print("SVM with extra features Error Rate:")
        errRateSVM_PI[i] = util.compute_errorRate(y_test_label, y_predicted)
        print(errRateSVM_PI[i])

    if 1:
        y_predicted = KT_LUPI(X_train, X_star_train, y_train_label, X_test, regMethod='Linear')

        print("Knowledge Transfer LUPI Error Rate:")
        errRateKT_LUPI[i] = util.compute_errorRate(y_test_label, y_predicted)
        print(errRateKT_LUPI[i])

    if 1:
        y_predicted = RobustKT_LUPI(X_train, X_star_train, y_train_label, X_test,
                                    regMethod='Linear', n_splits=5)

        print("Robust KT LUPI Error Rate:")
        errRateRobustKT_LUPI[i] = util.compute_errorRate(y_test_label, y_predicted)
        print(errRateRobustKT_LUPI[i])



pt.add_row(["Parkinsons", np.mean(errRateSVM) , np.mean(errRateSVM_PI),
            np.mean(errRateKT_LUPI), np.mean(errRateRobustKT_LUPI)])

print(pt)