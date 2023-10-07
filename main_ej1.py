import csv
import json
import math

import numpy as np
import plotly.graph_objects as go
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


def generate_heatmap(k, matrix):
    # Create a list to store the number of countries in each position
    num_countries_matrix = [[len(matrix[i][j]) for j in range(k)] for i in range(k)]

    # Create a list to store the labels for each cell
    labels = [[f'{num_countries_matrix[i][j]}: {", ".join(matrix[i][j])}' for j in range(k)] for i in range(k)]

    # Reverse the color scale
    colorscale = 'YlGnBu'

    # Create a heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=num_countries_matrix,
        x=[f'Column {i + 1}' for i in range(k)],
        y=[f'Row {i + 1}' for i in range(k)],
        text=labels,
        colorscale=colorscale,
        reversescale=True,
        colorbar=dict(title="Number of Countries")
    ))

    # Customize the layout
    fig.update_layout(
        title='Number of Countries in Each Position',
        xaxis=dict(side='top'),
    )

    # Show the heatmap
    fig.show()


if __name__ == "__main__":
    data = load_data("./training_data/europe.csv")

    with open("config_ej1.json") as file:
        config = json.load(file)

    # se le resta 1 al tamaño del input pues el nombre del pais no se usa
    model = Kohonen(len(data[0]) - 1, config["k"], config["radius"], config["initial_learning_rate"], data, config["initialize_random"])

    model.train(config["train_limit"])

    resp = model.test()

    pprint.pprint(resp)
    for row in resp:
        for col in row:
            print(f"{len(col)} ", end='')
        print()

    if config["generate_heatmap"]:
        generate_heatmap(config["k"], resp)




