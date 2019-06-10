import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LinearRegression, RidgeCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import utils as util
from sklearn.model_selection import StratifiedKFold
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from collections import Counter

# Data simulation functions ---------------------------------------------------------

def SimulateData(l=100, noise = 0.01):
    # Simulate training data
    x1 = np.random.uniform(-1, 1, size=l)
    x2 = np.random.uniform(-1, 1, size=l)

    y  = np.sign(x1 + x2)
    X_star = x1 + x2 + noise * np.random.normal(0, 1, size=l)

    X = np.column_stack((x1, x2))
    y[y==-1] = 0
    return X, y, X_star


grid_param  = {'C': [1/32,1/16,1/8,1/4,1/2,1, 32,16,8,4,2,1],
               "gamma": [1/64, 1/32,1/16,1/8,1/4,1/2,1, 64, 32,16,8,4,2,1]}


cv = 6

def fit_SVM(X_train, y_train, X_test):
    clf = GridSearchCV(SVC(kernel='rbf'), grid_param, cv=cv)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    testPred = clf.predict(X_test)
    return testPred


# fit LUPI with feature transformation using ridge regression (RR),
def fit_MLR(X_train, y_label):
    regression_model = LinearRegression()
    # Fit the data(train the model)
    regression_model.fit(X_train, y_label)
    return regression_model

def fit_RR(X_train, y_label):
    #param_grid = {"alpha": np.logspace(-10, 10, 20, base = 2)}
    #alphas=np.logspace(-10, 10, 20, base = 2
    model = RidgeCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1,
                                0.3, 0.6, 1], cv=3).fit(X_train, y_label)
    #model= GridSearchCV(Ridge(), cv=cv, param_grid=param_grid)
    model.fit(X_train, y_label)
    return model
    #print(model.best_estimator_.get_params())

def KT_LUPI(X_train, X_star, y_train_label, X_test):
    regModel = fit_MLR(X_train, X_star)

    y_transform = regModel.predict(X_train)
    y_test_transform = regModel.predict(X_test)

    X_train_mod = np.column_stack((X_train, y_transform))
    X_test_mod = np.column_stack((X_test, y_test_transform))

    '''
    scaler = StandardScaler()
    scaler.fit(X_mod)
    X_mod = scaler.transform(X_mod)
    X_test_mod = scaler.transform(X_test_mod)
    '''
    testPred = fit_SVM(X_train_mod, y_train_label, X_test_mod)
    #testPred = fit_SVM(y_transform.reshape(-1, 1), y_train_label, y_test_transform.reshape(-1, 1))

    return testPred


def RobustKT_LUPI(X_train, X_star, y_train_label, X_test, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits)
    testPred = np.zeros((len(X_test),2))

    for index1, index2 in kf.split(X_train, y_train_label):
        X_part1, X_part2 = X_train[index1], X_train[index2]
        y_part1, y_part2 = y_train_label[index1], y_train_label[index2]
        X_star_1, X_star_2 = X_star[index1], X_star[index2]

        # use part-2 for learning the transfer function

        x_s = X_star_2[:] #X_star_2[:, indexPI]

        regModel = fit_MLR(X_part2, x_s)

        x_s_train = regModel.predict(X_part1)
        x_s_test = regModel.predict(X_test)

        X_train_mod = np.column_stack((X_part1, x_s_train))
        X_test_mod = np.column_stack((X_test, x_s_test))

        '''
        scaler = StandardScaler()
        scaler.fit(X_train_mod)
        X_train_mod = scaler.transform(X_train_mod)
        X_test_mod = scaler.transform(X_test_mod)
        '''
        clf = GridSearchCV(SVC(probability=True), grid_param, cv=5)
        clf.fit(X_train_mod, y_part1)
        #clf.fit(x_s_train.reshape(-1, 1), y_part1)

        testPred = testPred + clf.predict_proba(X_test_mod)
        #testPred = testPred + clf.predict_proba(x_s_test.reshape(-1, 1))

    testPred = testPred / n_splits
    testPred = np.argmax(testPred, axis=1)
    return testPred





def run_experiments(X_train, y_train, X_test, y_test,
                    X_star_train, X_star_test):
    # normalization of the training data
    '''
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    '''
    results = OrderedDict()

    if 0:
        y_predicted = fit_SVM(X_train, y_train, X_test)
        results["svm"] = [util.compute_errorRate(y_test, y_predicted)]

    if 0:
        #X_train_mod = np.column_stack((X_train, X_star_train))
        #X_test_mod = np.column_stack((X_test, X_star_test))

        #scaler = StandardScaler()
        #scaler.fit(X_star_train.reshape(-1, 1))
        #X_train_mod = scaler.transform(X_star_train.reshape(-1, 1))
        #X_test_mod = scaler.transform(X_star_test.reshape(-1, 1))

        X_train_mod = X_star_train.reshape(-1, 1)
        X_test_mod = X_star_test.reshape(-1, 1)
        # print(X_train_mod.shape)

        y_predicted = fit_SVM(X_train_mod, y_train, X_test_mod)
        results["svm_pi"] = [util.compute_errorRate(y_test, y_predicted)]

    if 1:
        y_predicted = KT_LUPI(X_train, X_star_train, y_train, X_test)

        results["svm_kt_lupi"] = [util.compute_errorRate(y_test, y_predicted)]

    if 1:
        y_predicted = RobustKT_LUPI(X_train, X_star_train, y_train, X_test)
        results["svm_robust_kt_lupi"] = [util.compute_errorRate(y_test, y_predicted)]

    print(results)
    return results


if __name__ == '__main__':
    iter = 10

    dataset_name = 'synthetic'

    rmseSVM = list()
    rmseSVM_PI = list()
    rmseKT_LUPI = list()
    rmseRobustKT_LUPI = list()

    pt = PrettyTable()

    pt.field_names = ["Train Size", "SVM", "SVM with PI", "KT LUPI",
                      "Robust KT LUPI"]

    #train_size = [200, 300, 400, 500, 600]
    train_size = [25, 35, 45, 55]

    dictResult = OrderedDict()
    result = OrderedDict()

    for i in range(iter):
        print("iteration", i)


        for n in train_size:
            X_train, y_train, X_star_train = SimulateData(l=n)
            X_test, y_test, X_star_test = SimulateData(l=10000)

            dictResult[n] = run_experiments(X_train, y_train, X_test, y_test,
                    X_star_train, X_star_test)

        for key, value in dictResult.items():
            if i==0:
                result[key] = value
            else:
                for key2, value2 in value.items():
                    result[key][key2].extend(value2) #concatenate lists

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

print(np.mean(rmseSVM_PI))
#print(pt)