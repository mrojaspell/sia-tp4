import math

import numpy as np


def vectorize_pattern(pattern):
    array = []
    for i in range(len(pattern)):
        for j in range(len(pattern[0])):
            array.append(pattern[i][j])
    return np.array(array)


def unvectorize_pattern(vector):
    # Asume una matriz cuadrada
    size = int(math.sqrt(len(vector)))
    array = []
    for i in range(size):
        array.append([])
        for j in range(size):
            array[i].append(vector[i*size + j])
    return array


# TODO: fijarse la performance
def sign(old_vector, new_vector):
    resp = []
    for idx, elem in enumerate(new_vector):
        if elem > 0:
            resp.append(1)
        elif elem < 0:
            resp.append(-1)
        else:
            resp.append(old_vector[idx])
    return np.array(resp)


class Hopfield:
    def __init__(self, patterns):
        self.patterns = patterns  # Patterns es lista de matrices
        self.matrix_dimension = len(patterns[0])
        self.weight_matrix = np.zeros((self.matrix_dimension ** 2, self.matrix_dimension ** 2))
        self.sign_function = np.vectorize(sign)

        size = len(self.patterns[0])

        #self.weight_matrix = self.initialize_weights()


        # Completamos pesos (triangulo inferior)
        for i in range(size * size):
            for j in range(i + 1, size * size):
                self.weight_matrix[i][j] = self.get_weight(i, j)

        # Reflejamos
        for i in range(size * size):
            for j in range(0, i):
                self.weight_matrix[i][j] = self.weight_matrix[j][i]

    def initialize_weights(self):
        matrix = []
        for pattern in self.patterns:
            matrix.append(vectorize_pattern(pattern))

        row_data_matrix = np.array(matrix)

        columnar_data_matrix = np.transpose(matrix)

        weights = (columnar_data_matrix @ row_data_matrix) / 25

        for i in range(len(weights)):
            for j in range(len(weights)):
                if i == j:
                    weights[i][j] = 0

        return weights

    def get_weight(self, elem1, elem2):
        row1 = elem1 // self.matrix_dimension
        col1 = elem1 % self.matrix_dimension
        row2 = elem2 // self.matrix_dimension
        col2 = elem2 % self.matrix_dimension
        result = 0
        for pattern in self.patterns:
            result += pattern[row1][col1] * pattern[row2][col2]

        return result / (self.matrix_dimension ** 2)

    def recognize_pattern(self, pattern):
        prev = vectorize_pattern(pattern)

        result = None  # por el scope de python
        i = 0
        while i <= 100000:
            result = sign(prev, self.weight_matrix @ prev)
            if np.array_equal(result, prev):
                return unvectorize_pattern(result)
            prev = result
            i += 1

        if result is not None:
            return unvectorize_pattern(result)

        raise ValueError("Result no puede ser nulo, algo raro esta pasando")
