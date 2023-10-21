import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from main_ej1 import load_data

def standardize_matrix(matrix):
    resp = []

    matrix = np.transpose(matrix)

    for row in matrix:
        resp.append(np.array(stats.zscore(row)))

    return np.array(np.transpose(resp))

if __name__ == "__main__":
    data = load_data("./training_data/europe.csv")

    #Ambas formas de estandarizar son equivalentes
    training_data = StandardScaler().fit_transform([row[1:] for row in data])
    #training_data = standardize_matrix([row[1:] for row in data])

    pca = PCA(n_components=1)
    principalComponents = pca.fit_transform(training_data)

    # print(pca.components_)
    print(principalComponents)