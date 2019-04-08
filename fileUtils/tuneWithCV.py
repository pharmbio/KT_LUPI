import numpy as np
import fileUtils.gridSearchCV as gridsvmp
from sklearn import svm
import os
from sklearn.model_selection import GridSearchCV


# Parameter tuning with cross validation
def gridSearchWithCV(X_train, y_train, logFile):
    param_grid = {'C': [1, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}

    clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=5, n_jobs=5)
    clf = clf.fit(X_train, y_train)
    print(clf.best_estimator_)
    C, gamma = clf.best_params_['C'], clf.best_params_['gamma']

    # record the results in a file
    dirPath = "gridSearchResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + logFile, "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("param C = %f, gamma = %f \n" %(C, gamma))
    ofile.close()

    return C, gamma




# run SVM+ for sign descriptor files
def gridSearchSVMPlus(X_train, y_train, XStar_train, logFile,
                      kernelParam=0.0001, kernelParamStar=0.01):
    param_grid = {'C': [1, 10, 100, 1e3],
                  'gamma': [1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
    param_grid = {'C': [1], #[1, 10, 100, 1e3],
                  'gamma': [.01], #[1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                  'gamma_x': [kernelParam],
                  'gamma_xstar': [kernelParamStar]}

    C, gamma, gamma_x, gamma_x_star = gridsvmp.gridSearchSVMPlus(X_train, y_train, XStar_train,
                                                                 param_grid, n_splits=5)

    print(C, gamma, gamma_x, gamma_x_star)

    # record the results in a file
    dirPath = "gridSearchResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + logFile, "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("param C = %f, gamma = %f \n" % (C, gamma))
    ofile.close()

    return C, gamma