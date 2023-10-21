import numpy as np
from scipy import stats
from main_ej1 import load_data
from models.oja import LinearPerceptron


def standardize_matrix(matrix):
    resp = []

    for row in matrix:
        resp.append(np.array(stats.zscore(row)))

    return np.array(resp)


if __name__ == "__main__":
    data = load_data("./training_data/europe.csv")

    perceptron = LinearPerceptron()

    training_data = standardize_matrix([row[1:] for row in data])

    perceptron.train(training_data, 0.0001, 1000)

    first_component = []
    for i in range(len(training_data)):
        first_component.append(perceptron.compute_activation(training_data[i]))

    print(first_component)
