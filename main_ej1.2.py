import numpy as np
from scipy import stats
from main_ej1 import load_data
from models.oja import LinearPerceptron
from sklearn.preprocessing import StandardScaler


def standardize_matrix(matrix):
    resp = []

    matrix = np.transpose(matrix)

    for row in matrix:
        resp.append(np.array(stats.zscore(row)))

    return np.array(np.transpose(resp))


if __name__ == "__main__":
    data = load_data("./training_data/europe.csv")

    perceptron = LinearPerceptron()

    #ambas formas de estandarizar son equivalentes
    # training_data = standardize_matrix([row[1:] for row in data])
    training_data = StandardScaler().fit_transform([row[1:] for row in data])

    weights = perceptron.train(training_data, 0.001, 10000)

    # print(f"Weights: {weights}")

    first_component = []
    for i in range(len(training_data)):
        first_component.append(perceptron.compute_activation(training_data[i]))

    print(first_component)
