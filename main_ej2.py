import json
from models.hopfield import Hopfield

if __name__ == "__main__":
    with open("./training_data/letters_matrix.json") as f:
        letters: dict = json.load(f)

    data = []

    for key, value in letters.items():
        data.append(value)

    hopfield = Hopfield(data)

    result = hopfield.recognize_pattern(data[0])

    for i in range(len(result)):
        for j in range(len(result)):
            print(result[i][j], end=" ")
        print()
