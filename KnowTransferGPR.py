import dataPreprocess as data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
import utils as util
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel



# fit SVR
def fit_SVR(X_train, y_train, testData):
    grid_param = [{'kernel': ['rbf'], 'gamma': [.1,1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVR(epsilon=0.05), grid_param, cv=5)
    clf.fit(X_train, y_train)
    testPred = clf.predict(testData)
    return testPred
    print("fit SVR done")


def fit_GPR(X_train, y_train, testData):
    gp_kernel = PairwiseKernel(metric= 'rbf')

    clf = GaussianProcessRegressor(kernel=gp_kernel)
    clf.fit(X_train, y_train)
    testPred = clf.predict(testData)
    return testPred
    print("done")


def fit_SVM(X_train, y_train, X_test):
    grid_param = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVC(), grid_param, cv=5)
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


'''
# SVR omitting the labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/2, random_state=3)
print(X_train.shape)
y_hat = fit_SVR(X_train, y_train, X_test)
rmse = util.compute_rmse(y_test, y_hat)
print(rmse)
#result: 0.018364943549612024
'''

'''
# SVR using labels as an extra feature
X = np.column_stack((X, y_label))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/2, random_state=3)
print(X_train.shape)
y_hat = fit_SVR(X_train, y_train, X_test)
rmse = util.compute_rmse(y_test, y_hat)
print(rmse)
# result: 0.017021743715948785
'''

'''
y_transform = fit_GPR(X_train, y_train, X_test)
rmse = util.compute_rmse(y_test, y_transform)
print(rmse)
# result 0.01345
'''

'''
X = np.column_stack((X, y_label))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/2, random_state=3)
print(X_train.shape)
y_hat = fit_SVR(X_train, y_train, X_test)
rmse = util.compute_rmse(y_test, y_hat)
print(rmse)
'''

# Using feature transformation for labels..

#X, y, y_label = data.load_gridStability_data()
X, y, y_label = data.load_PD_data() # does not work
#X, y, y_label = data.load_wine_data()

X_train, X_test, y_train_label, y_test_label, train_index, test_index = \
    train_test_split(X, y_label, range(len(X)),test_size=.2, random_state=3, stratify=y_label)
y_train = y[train_index]
y_test = y[test_index]

if 1:
    y_predicted = fit_SVM(X_train, y_train_label, X_test)
    correct = sum(y_test_label == y_predicted)
    #rmse = util.compute_rmse(y_test, y_transform)
    print("SVM Result:")
    print(correct)

    X = np.column_stack((X_train, y_train))
    X_test_mod = np.column_stack((X_test, y_test))
    y_predicted = fit_SVM(X, y_train_label, X_test_mod)
    correct = sum(y_test_label == y_predicted)
    #rmse = util.compute_rmse(y_test, y_transform)
    print("SVM with extra features Result:")
    print(correct)

if 1:
    y_predicted = KT_LUPI(X_train, y_train, y_train_label, X_test)
    correct = sum(y_test_label == y_predicted)
    print("Knowledge Transfer LUPI Result:")
    print(correct)


