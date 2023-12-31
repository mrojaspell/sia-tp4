import csv
import json
import numpy as np
import plotly.graph_objects as go
from models.kohonen import Kohonen


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
    return data

def generate_heatmap_for_variable(k:Kohonen,variable_index,variable_name,matrix):
    new_matrix = []
    for i in range(len(k.neurons)):
        subList = []
        for j in range(len(k.neurons)):
            subList.append(k.neurons[i][j].weights[variable_index])
        new_matrix.append(subList)

    matrix_results = np.array(new_matrix)

    # Create a list to store the labels for each cell
    labels = []

    for i in range(len(k.neurons)):
        row_labels = []
        for j in range(len(k.neurons)):
            countries = matrix[i][j]
            t = 0
            group = ''
            for name in countries:
                group += name
                if t < len(countries) - 1:
                    group += ', '  # Add a comma if it's not the last country
                if t % 3 == 2:
                    group += '<br>'
                t += 1
            row_labels.append(group)
        labels.append(row_labels)

    # Reverse the color scale
    colorscale = 'YlGnBu'


    # Creating a heatmap
    fig = go.Figure(data=go.Heatmap(z=matrix_results,
                                    x=[f'Column {i + 1}' for i in range(len(k.neurons))],
        y=[f'Row {i + 1}' for i in range(len(k.neurons))],
        text=labels,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale=colorscale,
        reversescale=True,
        colorbar=dict(title="Number of Countries")))



    # Customizing the layout
    fig.update_layout(
        title=f'Heatmap for variable {variable_name}'
    )

    # Showing the plot
    fig.show()




def generate_heatmap(k, matrix):
    # Create a list to store the number of countries in each position
    num_countries_matrix = [[len(matrix[i][j]) for j in range(k)] for i in range(k)]

    # Create a list to store the labels for each cell
    labels = []

    for i in range(k):
        row_labels = []
        for j in range(k):
            countries = matrix[i][j]
            t = 0
            group = ''
            for name in countries:
                group += name
                if t < len(countries) - 1:
                    group += ', '  # Add a comma if it's not the last country
                if t % 3 == 2:
                    group += '<br>'
                t += 1
            row_labels.append(group)
        labels.append(row_labels)

    # Reverse the color scale
    colorscale = 'YlGnBu'

    # Create a heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=num_countries_matrix,
        x=[f'Column {i + 1}' for i in range(k)],
        y=[f'Row {i + 1}' for i in range(k)],
        text=labels,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale=colorscale,
        reversescale=True,
        colorbar=dict(title="Number of Countries")
    ))

    # Customize the layout
    fig.update_layout(
        title='Kohonen Heatmap for countries',
        xaxis=dict(side='top'),
    )

    # Show the heatmap
    fig.show()


def generate_u_matrix(neuron_matrix, k, radius):
    # Initialize an empty U-matrix
    u_matrix = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            neuron = neuron_matrix[i, j]
            neighbors = []
            # Iterate through the neighbors of the current neuron
            for m in range(k):
                for n in range(k):
                    if (i, j) != (m, n) and ((i-m) ** 2 + (j-n) ** 2) <= radius ** 2:
                        neighbors.append(neuron_matrix[m, n])

            # Calculate the mean Euclidean distance between the current neuron and its neighbors
            neighbors = np.array(neighbors)
            average = 0
            for neighbor in neighbors:
                sum = (neuron.weights - neighbor.weights) ** 2
                sum = np.sum(sum)
                average += sum
            average /= len(neighbors)
            u_matrix[i][j] = average

    # Create a Plotly heatmap trace for the U-matrix
    heatmap_trace = go.Heatmap(z=u_matrix,
                               text=u_matrix,
                               texttemplate="%{text}",
                               textfont={"size": 10},
                               colorscale='Greys')

    layout = go.Layout(title='Kohonen U-Matrix')
    fig = go.Figure(data=[heatmap_trace], layout=layout)
    fig.show()


if __name__ == "__main__":
    data = load_data("./training_data/europe.csv")

    with open("config_ej1.json") as file:
        config = json.load(file)

    # se le resta 1 al tamaño del input pues el nombre del pais no se usa
    model = Kohonen(len(data[0]) - 1, config["k"], config["radius"], config["initial_learning_rate"], config["variable_learning_rate"], data, config["initialize_random"])

    model.train(config["train_limit"], config["variable_radius"])

    resp = model.test()

    if config["generate_heatmap"]:
        generate_heatmap(config["k"], resp)

    if config["generate_U_matrix"]:
        if not config["variable_radius"]:
            generate_u_matrix(model.neurons, config["k"], config["radius"])
        else:
            generate_u_matrix(model.neurons, config["k"], 1)


    if config["generate_heatmap_for_variable"]:
        variables = ["Area", "GDP", "Inflation", "Life.expect", "Military", "Pop.growth", "Unemployment"]

        variable_to_check = config["generate_heatmap_for_variable_name"]

        result = None

        if variable_to_check in variables:
            result_index = variables.index(variable_to_check)
        else:
            quit("heatmap variable not found")

        generate_heatmap_for_variable(model,result_index,variable_to_check,resp)






