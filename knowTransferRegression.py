import dataPreprocess as data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
import utils as util
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
import statsmodels.api as sm

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
    gp_kernel = RBF()

    clf = GaussianProcessRegressor(kernel=gp_kernel)
    clf.fit(X_train, y_train)
    testPred = clf.predict(testData)
    return testPred
    print("done")


# fit LUPI with feature transformation using GPR, when we have only one
# privileged feature
def KT_LUPI_GPR(X_train, y_train, x_star, X_test):
    #gp_kernel = PairwiseKernel(metric= 'rbf')
    #gp_kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp_kernel = RBF()
    gpr = GaussianProcessRegressor(kernel=gp_kernel)
    gpr.fit(X_train, x_star)
    y_transform = gpr.predict(X_train)
    y_test_transform = gpr.predict(X_test)
    X = np.column_stack((X_train, y_transform))
    test_data = np.column_stack((X_test, y_test_transform))

    grid_param = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVR(), grid_param, cv=5)
    clf.fit(X, y_train)
    testPred = clf.predict(test_data)
    return testPred


# fit LUPI with feature transformation using kernel ridge,
# when we have only one privileged feature
def KT_LUPI_KRR(X_train, y_train, x_star, X_test):
    param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
                  'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]}
    kr = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
    kr.fit(X_train, x_star)
    y_transform = kr.predict(X_train)
    y_test_transform = kr.predict(X_test)
    X = np.column_stack((X_train, y_transform))
    test_data = np.column_stack((X_test, y_test_transform))

    grid_param = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVR(), grid_param, cv=5)
    clf.fit(X, y_train)
    testPred = clf.predict(test_data)
    return testPred


def KT_LUPI_SVR(X_train, y_train, x_star, X_test):
    grid_param = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [1, 10, 100, 1000]}]

    svr = GridSearchCV(SVR(), grid_param, cv=5)
    svr.fit(X_train, x_star)
    y_transform = svr.predict(X_train)
    y_test_transform = svr.predict(X_test)
    X = np.column_stack((X_train, y_transform))
    test_data = np.column_stack((X_test, y_test_transform))

    grid_param = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVR(), grid_param, cv=5)
    clf.fit(X, y_train)
    testPred = clf.predict(test_data)
    return testPred

# fit LUPI with feature transformation using linear regression,
# when we have only one privileged feature
def KT_LUPI_LR(X_train, y_train, x_star, X_test):
    model = sm.OLS(x_star, X_train).fit()
    y_transform = model.predict(X_train)  # make the predictions by the model
    y_test_transform = model.predict(X_test)
    X = np.column_stack((X_train, y_transform))
    test_data = np.column_stack((X_test, y_test_transform))

    grid_param = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVR(), grid_param, cv=5)
    clf.fit(X, y_train)
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
X, y, x_star = data.load_energy_data()# does not work
#X, y, y_label = data.load_wine_data()

X_train, X_test, y_train, y_test, train_index, test_index = \
    train_test_split(X, y, range(len(X)),test_size=.2, random_state=3)
x_star_train = x_star[train_index]
x_star_test = x_star[test_index]

if 0:
    y_predicted = fit_SVR(X_train, y_train, X_test)
    rmse = util.compute_rmse(y_test, y_predicted)
    print("SVM Result:")
    print(rmse)

    X_mod = np.column_stack((X_train, x_star_train))
    X_test_mod = np.column_stack((X_test, x_star_test))
    y_predicted = fit_SVR(X_mod, y_train, X_test_mod)
    rmse = util.compute_rmse(y_test, y_predicted)
    print("SVM with extra features Result:")
    print(rmse)

if 0:
    y_predicted = KT_LUPI_GPR(X_train, y_train, x_star_train, X_test)
    rmse = util.compute_rmse(y_test, y_predicted)
    print("Knowledge Transfer LUPI Result:")
    print(rmse)


''' 
# specific to energy efficiency dataset
y_predicted = fit_SVR(x_star_train.reshape(-1, 1), y_train, x_star_test.reshape(-1, 1))
rmse = util.compute_rmse(y_test, y_predicted)
print("SVM with extra features Result:")
print(rmse)
'''
