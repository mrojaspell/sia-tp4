import copy
import random
import numpy as np
import math
from scipy.stats import stats


def standardize_input(data):
    resp = []
    training_data = []

    for idx in range(len(data)):
        resp.append(np.array([data[idx][0], None]))
        new_data = []
        for i in range(1, len(data[idx])):
            new_data.append(float(data[idx][i]))
        training_data.append(new_data)

    training_data = np.transpose(training_data)
    std_data = []
    for row in training_data:
        std_data.append(np.array(stats.zscore(row)))
    std_data = np.transpose(std_data)

    for idx, row in enumerate(std_data):
        resp[idx][1] = np.array(row)

    return np.array(resp)


def calculate_euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector2 - vector1)


class Kohonen:
    class Neuron:
        def __init__(self, initial_weights):
            self.weights = initial_weights
            self.winners = []

        def add_winner(self, winner):
            self.winners.append(winner)

    # x es la cantidad de atributos por cada input

    def __init__(self, x, k, radius, initial_learning_rate, variable_learning_rate, training_data,
                 initialized_random=True):
        self.neurons = np.empty(dtype=Kohonen.Neuron, shape=(k, k))
        self.variable_learning_rate = variable_learning_rate
        self.training_data = standardize_input(training_data)

        for i in range(k):
            for j in range(k):
                # Caso: inicializar con random
                if initialized_random:
                    random_array = np.random.uniform(-1, 1, size=x)  # TODO: check
                    self.neurons[i][j] = Kohonen.Neuron(random_array)

                # Caso: inicializar con datos al azar
                else:
                    self.neurons[i][j] = Kohonen.Neuron(copy.deepcopy(random.choice(self.training_data)[1]))

        self.radius = radius
        self.learning_rate = initial_learning_rate

    def get_learning_rate(self, iteration=None):
        if not self.variable_learning_rate or iteration is None or iteration == 0:
            return self.learning_rate
        return self.learning_rate / iteration

    def train(self, limit, variable_radius=False):
        for i in range(limit):
            for city_name, data in self.training_data:
                row, col = self.get_winner_neuron(data)

                self.update_weights(row, col, data, i)

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

    def update_weights(self, row, col, current_input, iteration):
        # acotamos radio por diagonal de la matriz, sqrt(2) * K
        radius = min(self.radius, len(self.neurons))

        min_fil = max(math.floor(row - radius), 0)
        max_fil = min(math.ceil(row + radius), len(self.neurons) - 1)
        min_col = max(math.floor(col - radius), 0)
        max_col = min(math.ceil(col + radius), len(self.neurons[0]) - 1)

        for current_row in range(min_fil, max_fil + 1):
            for current_col in range(min_col, max_col, + 1):
                distance = math.sqrt((current_row - row) ** 2 + (current_col - col) ** 2)
                # cumple radio y está dentro de la matriz
                if distance <= radius:
                    # se actualizan los pesos
                    neuron = self.neurons[current_row][current_col]
                    neuron.weights = neuron.weights + self.get_learning_rate(iteration) * (current_input - neuron.weights)

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
