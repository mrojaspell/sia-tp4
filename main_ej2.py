import copy
import json
import random
import numpy as np
from models.hopfield import Hopfield, vectorize_pattern


INPUT_SIZE = 4


# Genera un conjunto "mas othogonal" comenzando de un elemento aleatorio.
# Por lo tanto, no es el mas ortogonal, sino que uno bastante ortognal
# considerando el valor aleatorio inicial.
def generate_random_most_orthogonal(input_values, size):

    random.shuffle(copy.deepcopy(input_values))

    resp = [input_values.pop()]
    min_dot = None
    min_elem = None
    while len(resp) < size:
        for elem in input_values:
            dot_accum = 0
            for select in resp:
                dot_accum += abs(np.dot(select, elem))
            if min_dot is None or dot_accum < min_dot:
                min_dot = dot_accum
                min_elem = elem

        resp.append(min_elem)
        min_dot = None
    return resp


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

    data = []
    for key, value in letters.items():
        data.append(vectorize_pattern(value))

    selected = generate_random_most_orthogonal(data, config["selection_size"])

    hopfield = Hopfield(selected)

    selected_pattern = selected[0]

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
