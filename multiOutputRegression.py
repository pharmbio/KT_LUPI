import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
import utils as util


# fit SVR on training and predict for test data
def fit_SVR(X_train, y_train, testData):
    grid_param = [{'kernel': ['rbf'], 'gamma': [.1,1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVR(epsilon=0.05), grid_param, cv=5)
    clf.fit(X_train, y_train)
    testPred = clf.predict(testData)
    return testPred
    print("fit SVR done")

# fit LUPI with feature transformation using GPR, when we have only one
# privileged feature
def KT_LUPI_GPR(X_train, y_train, x_star, X_test):
    #gp_kernel = RBF()
    #gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) \
    #            + WhiteKernel(1e-1)
    gp_kernel = ExpSineSquared()
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


# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y += (0.5 - rng.rand(*y.shape))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=400, test_size=200, random_state=4)

if 0:
    y_predicted = fit_SVR(X_train, y_train[:,1], X_test)
    rmse = util.compute_rmse(y_test[:,1], y_predicted)
    print("SVM Result:")
    print(rmse)

if 0:

    X_mod = np.column_stack((X_train, y_train[:,1]))
    X_test_mod = np.column_stack((X_test, y_test[:,1]))
    y_predicted = fit_SVR(X_mod, y_train[:,0], X_test_mod)
    rmse = util.compute_rmse(y_test[:,0], y_predicted)
    print("SVM with extra features Result:")
    print(rmse)

if 1:
    y_predicted = KT_LUPI_GPR(X_train, y_train[:,0], y_train[:,1], X_test)
    rmse = util.compute_rmse(y_test[:,0], y_predicted)
    print("Knowledge Transfer LUPI Result:")
    print(rmse)


'''
max_depth = 30
regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                                          max_depth=max_depth,
                                                          random_state=0))
regr_multirf.fit(X_train, y_train)

regr_rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth,
                                random_state=2)
regr_rf.fit(X_train, y_train)

# Predict on new data
y_multirf = regr_multirf.predict(X_test)
y_rf = regr_rf.predict(X_test)

# Plot the results
plt.figure()
s = 50
a = 0.4
plt.scatter(y_test[:, 0], y_test[:, 1], edgecolor='k',
            c="navy", s=s, marker="s", alpha=a, label="Data")
plt.scatter(y_multirf[:, 0], y_multirf[:, 1], edgecolor='k',
            c="cornflowerblue", s=s, alpha=a,
            label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test))
plt.scatter(y_rf[:, 0], y_rf[:, 1], edgecolor='k',
            c="c", s=s, marker="^", alpha=a,
            label="RF score=%.2f" % regr_rf.score(X_test, y_test))
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Comparing random forests and the multi-output meta estimator")
plt.legend()
plt.show()
'''