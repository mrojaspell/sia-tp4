import numpy as np

def vectorize_pattern(pattern):
    array = []
    for i in range(len(pattern)):
        for j in range(len(pattern[0])):
            array.append(pattern[i][j])
    return np.array(array)

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return x


class Hopfield:
    def __init__(self, patterns):
        self.patterns = patterns                # Patterns es lista de matrices
        self.matrix_dimension = len(patterns[0])
        self.weight_matrix = np.zeros((self.matrix_dimension ** 2, self.matrix_dimension ** 2))
        self.sign_function = np.vectorize(sign)

        size = len(self.patterns[0])

        # Completamos pesos (triangulo inferior)
        for i in range(size * size):
            for j in range(i + 1, size * size):
                self.weight_matrix[i][j] = self.get_weight(i, j)

        # Reflejamos
        for i in range(size * size):
            for j in range(0, i):
                self.weight_matrix[i][j] = self.weight_matrix[j][i]


    def get_weight(self,elem1,elem2):
        row1 = elem1 // self.matrix_dimension
        col1 = elem1 % self.matrix_dimension
        row2 = elem2 // self.matrix_dimension
        col2 = elem2 % self.matrix_dimension
        result = 0
        for pattern in self.patterns:
            result += pattern[row1][col1] * pattern[row2][col2]

        return result / (self.matrix_dimension ** 2)



    def recognize_pattern(self, pattern):
        vector = vectorize_pattern(pattern)
        vector =
        i = 0
        prev = None
        while i <= 10000:

            prev = result

