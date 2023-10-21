import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

from main_ej1 import load_data

def standardize_matrix(matrix):
    resp = []

    for row in matrix:
        resp.append(np.array(stats.zscore(row)))

    return np.array(resp)

if __name__ == "__main__":
    data = load_data("./training_data/europe.csv")

    training_data = [row[1:] for row in data]

    pca = PCA(n_components=1)
    principalComponents = pca.fit_transform(standardize_matrix(training_data))

    print(pca.components_)
    print(principalComponents)