from sklearn.metrics import mutual_info_score
import dataPreprocess as data
import numpy as np
from sklearn.preprocessing import StandardScaler


def mutual_info_rank_features(feature_vecs, binary_labels):
    """
    Given a set of feature vectors and binary labels, return
    the list of indices of the features ranked by mutual information
    with the binary labels.
    Args:
        feature_vecs: list of feature vectors
        binary_labels: list of binary labels
    """

    # Convert Features to Boolean values
    bin_feature_vecs = []
    for feature_v in feature_vecs:

        nfv = []
        for elem in feature_v:
            if elem > 0:
                nfv.append(1)
            else:
                nfv.append(0)
        bin_feature_vecs.append(nfv)

    mutual_infos = []
    num_features = len(bin_feature_vecs[0])
    for i in range(num_features):
        row_i = [x[i] for x in bin_feature_vecs]
        mi = mutual_info_score(row_i, binary_labels)
        mutual_infos.append(mi)

    ranked_indices = [index for (mi, index) in sorted(zip(mutual_infos, [x for x in range(num_features)]))]
    return ranked_indices


if __name__ == "__main__":
    #X, y = data.load_wine_data()
    #X, y = data.load_PD_data()
    X, y = data.load_bc_data()

    print(X.shape)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    ranked_indices = mutual_info_rank_features(X,y)
    print(ranked_indices)
