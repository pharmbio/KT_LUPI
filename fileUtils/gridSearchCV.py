import numpy as np
#import svmplus
from sklearn.model_selection import StratifiedKFold
import os
import multiprocessing
import svmPlusOpt as svmPlus

# global variable
gX_train = None
gy_train = None
gXStar_train = None
gFolds = None
gParam = []

def fitSVMPlus(k):
    fold = gFolds[k]
    cv_X_train = gX_train[fold[0]]
    cv_X_test = gX_train[fold[1]]
    cv_y_train = gy_train[fold[0]]
    cv_y_test = gy_train[fold[1]]
    # compute prediction accuracy using SVM+ on svmFile, and svmPlusFile2 as a priv-info
    svmp = svmPlus.svmPlusOpt(cv_X_train, cv_y_train, XStar=gXStar_train[fold[0]],
                         C=gParam[0], gamma=gParam[1],
                         kernel="rbf", kernelParam=gParam[2],
                         kernelStar="rbf", kernelStarParam=gParam[3])
    #libsvm.LibSVMPlus(gParam[0], gParam[1], gParam[2], gParam[3])
    #svmp.fit(cv_X_train, XStar=gXStar_train[fold[0]], y = cv_y_train)
    y_predict = svmPlus.predict(cv_X_test, svmp)
    correct = np.sum(y_predict == cv_y_test)
    accuracy = correct / len(y_predict)
    return accuracy


param_grid = {'C': [1, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
              'gamma_x': [1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
              'gamma_xstar': [1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}

# run SVM+ for sign descriptor files
def gridSearchSVMPlus(X_train, y_train, XStar_train,
                      param_grid, n_splits = 5, logfile = None):
    global gX_train
    gX_train = X_train
    global gy_train
    gy_train = y_train
    global gXStar_train
    gXStar_train = XStar_train
    global gParam

    if logfile is not None:
        ofile = open(logfile, "a")
        ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
        ofile.close()  # store file size and close, then open again

    predAcc = []
    index = []
    cv = StratifiedKFold(n_splits=n_splits)
    folds = [[train_index, test_index] for train_index, test_index in cv.split(X_train, y_train)]
    global gFolds
    gFolds = folds

    C = param_grid['C']
    gamma = param_grid['gamma']
    gamma_x = param_grid['gamma_x']
    gamma_xstar = param_grid['gamma_xstar']

    for i in range(len(C)):
        for j in range(len(gamma)):
            for k in range(len(gamma_x)):
                for l in range(len(gamma_xstar)):
                    gParam = [C[i], gamma[j], gamma_x[k], gamma_xstar[l]]
                    pool = multiprocessing.Pool(processes=n_splits)
                    accuracy = pool.map(fitSVMPlus, range(n_splits))
                    pool.close()
                    pool.join()

                    print(i, j)
                    accuracy = sum(accuracy) / n_splits
                    predAcc.append(accuracy)
                    index.append([i, j, k, l])
                    if logfile is not None:
                        ofile = open(logfile, "a")
                        ofile.write("param C = %f, gamma = %f, mean pred accuracy = %f \n" %
                                    (C[i], gamma[j], accuracy))
                        ofile.close()

    selectedIndex = np.argmax(predAcc)
    best_C = C[index[selectedIndex][0]]
    best_gamma = gamma[index[selectedIndex][1]]
    best_gamma_x = gamma_x[index[selectedIndex][2]]
    best_gamma_xstar = gamma_xstar[index[selectedIndex][3]]

    return best_C, best_gamma, best_gamma_x, best_gamma_xstar