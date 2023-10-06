import csv
import math

import numpy as np
from scipy import stats

from models.kohonen import Kohonen
import pprint

def load_data(path: str):
    data = []
    with open(path, "r") as f:
        reader = csv.reader(f)

        # Skip the header row
        next(reader)

        for row in reader:
            # Convert the row data to the appropriate data types
            country = row[0]
            area = float(row[1])
            gdp = float(row[2])
            inflation = float(row[3])
            life_expectancy = float(row[4])
            military = float(row[5])
            pop_growth = float(row[6])
            unemployment = float(row[7])

            # Create a list to represent a single row of data
            row_data = [country, area, gdp, inflation, life_expectancy, military, pop_growth, unemployment]

            # Append the row data to the list
            data.append(row_data)

    # Convert the list of lists to a NumPy array
    return np.array(data)

if __name__ == "__main__":
    data = load_data("./training_data/europe.csv")

    model = Kohonen(len(data[0]) - 1, 4, math.sqrt(2), 0.1, data, False)

    model.train(50000)

    resp = model.test()

    pprint.pprint(resp)
    for row in resp:
        for col in row:
            print(f"{len(col)} ", end='')
        print()



