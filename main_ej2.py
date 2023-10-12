import copy
import itertools
import json
import random
import numpy as np
from models.hopfield import Hopfield, vectorize_pattern


INPUT_SIZE = 4


# Genera un conjunto "mas othogonal" comenzando de un elemento aleatorio.
# Por lo tanto, no es el mas ortogonal, sino que uno bastante ortognal
# considerando el valor aleatorio inicial.
def generate_random_most_orthogonal(data_map, size):
    values = copy.deepcopy(data_map.values())
    random.shuffle(values)

    resp = [values.pop()]
    min_dot = None
    min_elem = None
    while len(resp) < size:
        for elem in values:
            dot_accum = 0
            for select in resp:
                dot_accum += abs(np.dot(select, elem))
            if min_dot is None or dot_accum < min_dot:
                min_dot = dot_accum
                min_elem = elem

        resp.append(min_elem)
        min_dot = None
    return resp


def generate_orthogonal_matrices(data_map, size):
    all_groups = itertools.combinations(data_map.keys(), r=size)

    avg_dot_product = []
    max_dot_product = []
    for combination in all_groups:
        group = np.array([v for k, v in data_map.items() if k in combination])
        orto_matrix = group.dot(group.T)
        np.fill_diagonal(orto_matrix, 0)
        row, _ = orto_matrix.shape
        avg_dot_product.append((np.abs(orto_matrix).sum()/(orto_matrix.size-row), combination))
        max_v = np.abs(orto_matrix).max()
        max_dot_product.append(((max_v, np.count_nonzero(np.abs(orto_matrix) == max_v) / 2), combination))

    return sorted(avg_dot_product, key=lambda x: x[0]), sorted(max_dot_product, key=lambda x: x[0])


def noisify_pattern(pattern, noise_prob):
    # No alteramos el original
    pattern = copy.deepcopy(pattern)

    # Asume que pattern esta vectorizado
    for i in range(len(pattern)):
        chance = random.uniform(0,1)
        if chance < noise_prob:
            pattern[i] = - pattern[i]

    return pattern


if __name__ == "__main__":

    with open("config_ej2.json") as file:
        config = json.load(file)

    with open(config["training_data"]) as f:
        letters: dict = json.load(f)

    data = {}
    for key, value in letters.items():
        data[key] = vectorize_pattern(value)

    orthogonal_avg, orthogonal_max = generate_orthogonal_matrices(data, config["selection_size"])

    selection_index = config["selection_index"]

    selected = [data[key] for key in orthogonal_avg[selection_index][1]]

    print(f"Using the following values: {orthogonal_avg[selection_index][1]}")

    hopfield = Hopfield(selected)

    for idx, selected_pattern in enumerate(selected):

        print(f"\nAttempting to recognize: {orthogonal_avg[selection_index][1][idx]}")

        result = hopfield.recognize_pattern(noisify_pattern(selected_pattern, config["noise_level"]), config["limit"])

        equal = True
        for i in range(len(result)):
            for j in range(len(result)):
                if result[i][j] != selected_pattern[5*i + j]:
                    equal = False

        if equal:
            print("Recognized correctly!")
        else:
            print("Did not recognize!")
