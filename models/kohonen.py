import random

import numpy as np
from scipy import stats
import math


def calculate_euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector2 - vector1)


def standardize_input(data):
    resp = []
    training_data = []

    for idx in range(len(data)):
        resp.append(np.array([data[idx][0], None]))
        new_data = []
        for i in range(1, len(data[idx])):
            new_data.append(float(data[idx][i]))
        training_data.append(new_data)

    for idx, elem in enumerate(training_data):
        resp[idx][1] = np.array(stats.zscore(elem))

    return np.array(resp)


class Kohonen:
    class Neuron:
        def __init__(self, initial_weights):
            self.weights = initial_weights
            self.winners = []

        def add_winner(self, winner):
            self.winners.append(winner)

    # x es la cantidad de atributos por cada input
    def __init__(self, x, k, radius, initial_learning_rate, training_data, initialized_random=True):
        self.neurons = np.empty(dtype=Kohonen.Neuron, shape=(k, k))

        self.training_data = standardize_input(training_data)

        for i in range(k):
            for j in range(k):
                # Caso: inicializar con random
                if initialized_random:
                    random_array = np.random.uniform(-1, 1, size=x)  # TODO: check
                    self.neurons[i][j] = Kohonen.Neuron(random_array)

                # Caso: inicializar con datos al azar
                else:
                    self.neurons[i][j] = Kohonen.Neuron(
                        random.choice(self.training_data)[1])  # Ignoramos el nombre de los elementos

        self.radius = radius
        self.learning_rate = initial_learning_rate

    def train(self, limit, variable_radius=False):
        size = len(self.training_data)
        for i in range(limit):
            idx = random.randint(0, size - 1)
            city_name, data = self.training_data[idx]

            row, col = self.get_winner_neuron(data)

            self.update_weights(row, col, data)

            if variable_radius:
                if self.radius > 1:
                    self.radius -= 1
                if self.radius < 1:
                    self.radius = 1

    def test(self):

        resp = []

        for rows in range(len(self.neurons)):
            resp.append([])
            for cols in range(len(self.neurons)):
                resp[rows].append([])

        for city, data in self.training_data:
            row, col = self.get_winner_neuron(data)
            resp[row][col].append(city)
        return resp

    def update_weights(self, row, col, current_input):
        radius = min(self.radius,
                     (2 ** 1 / 2) * len(self.neurons))  # actoamos radio por diagonal de la matriz, sqrt(2) * K
        min_fil = math.floor(row - radius)
        max_fil = math.ceil(row + radius)
        min_col = math.floor(col - radius)
        max_col = math.ceil(col + radius)

        for current_row in range(min_fil, max_fil + 1):
            for current_col in range(min_col, max_col, + 1):
                distance = math.sqrt((current_row - row) ** 2 + (current_col - col) ** 2)
                # cumple radio y está dentro de la matriz
                if distance <= radius and 0 <= current_row < len(self.neurons) and 0 <= current_col < len(self.neurons[0]):
                    # se actualizan los pesos
                    neuron = self.neurons[current_row][current_col]
                    neuron.weights = neuron.weights + self.learning_rate * (current_input - neuron.weights)

    def get_winner_neuron(self, data):

        min_dif = float("inf")
        row = None
        col = None

        for i in range(len(self.neurons)):
            for j in range(len(self.neurons[i])):
                distance = calculate_euclidean_distance(self.neurons[i][j].weights, data)
                if distance < min_dif:
                    row = i
                    col = j
                    min_dif = distance

        return row, col
