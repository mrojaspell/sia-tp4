import copy
from random import randint
import numpy as np


def initialize_weights(size):
    resp = []
    for _ in range(size):
        resp.append(np.random.uniform(0, 1))
    return np.array(resp)


class LinearPerceptron:
    def __init__(self):
        self.weights = None

    def train(self, training_data, learning_constant, limit):

        self.weights = initialize_weights(len(training_data[0]))  # Tomo uno cualquiera como inicial

        for _ in range(limit):
            for row in training_data:
                activation = self.compute_activation(row)

                self.update_weights(learning_constant, activation, row)

        return self.weights

    def compute_activation(self, row):
        return np.dot(row, self.weights)

    def update_weights(self, learning_constant, activation, row):
        self.weights += learning_constant * activation * (row - activation * self.weights)
