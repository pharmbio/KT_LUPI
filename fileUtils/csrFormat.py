import numpy as np
import csv as csv
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from numpy.random import RandomState

def formatStr(strInput):
    p = strInput.find(":")
    colNo = strInput[:p]
    data = strInput[p+1:]
    return float(colNo), float(data)


# to make sure there are no missing values
def readCSRFile(fileName, split = False, returnIndices = False, delimiter = '\t'):
    # Read the dataset from given file
    file = open(fileName)
    reader = csv.reader(file, delimiter = delimiter)
    nRow = 0 # number of rows
    colList = []
    rowList = []
    dataList = []
    labels = []
    for row in reader:
        # skip label
        labels = labels + row[:1]
        row = row[1:]
        for val in row:
            if val:
                colNo, data = formatStr(val)
                rowList.append(nRow)
                colList.append(colNo)
                dataList.append(data)
        nRow = nRow+1
    file.close()
    y = np.array([x for x in labels]).astype(float) #np.array(labels).astype(int).T
    columns = np.array(colList).astype(int)
    rows = np.array(rowList).astype(int)
    data = np.array(dataList)
    csMatrix = csr_matrix((np.array(data).astype(float), (np.array(rows).astype(int), np.array(columns).astype(int))))
    X = csMatrix.toarray()
    print(nRow)
    if split:
        if returnIndices:
            return train_test_split(X, y, range(nRow), test_size=0.2, stratify = y, random_state = 7)
        else:
            return train_test_split(X, y, test_size=0.2, stratify = y, random_state = 7)
    else:
        return X, y



if __name__ == "__main__":

    #X, y = readCSRFile("../DDdatasets/Cathepsin_D_flag.csr", delimiter = ' ')
    #print(X.shape)
    #print(X[1:10,1:10])
    X, y = readCSRFile("../DDdatasets/Cathepsin_D_regr.precomp.csr", delimiter=' ')
    y[y <= 10000] = -1
    y[y > 10000] = 1
    print(y[1:10])
    print(X.shape)
    #print(X[1:10, 1:10])
    #print(y[1:10])
    #X, y = readCSRFile("../DDdatasets/HERG_flag.precomp.csr", delimiter = ' ')
    #print(X.shape)
    #X, y = readCSRFile("../DDdatasets/HERG_regr.precomp.csr", delimiter=' ')
    #print(X.shape)
    #X, y = readCSRFile("../DDdatasets/Protein_tyrosine_phosphatase_1B_flag.precomp.csr", delimiter=' ')
    #print(X.shape)
    #X, y = readCSRFile("../DDdatasets/Protein_tyrosine_phosphatase_1B_regr.precomp.csr", delimiter=' ')
    #print(X.shape)