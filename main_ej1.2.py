import json
import numpy as np
from scipy import stats
from main_ej1 import load_data
from models.oja import LinearPerceptron
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd


def standardize_matrix(matrix):
    resp = []

    matrix = np.transpose(matrix)

    for row in matrix:
        resp.append(np.array(stats.zscore(row)))

    return np.array(np.transpose(resp))


def oja_pc1_graph(ppcomponents_oja, ppcomponents_pca, countries):
    if ppcomponents_oja[0] * ppcomponents_pca[0][0] < 0:
        ppcomponents_pca = [-value for value in ppcomponents_pca]

    ppcomponents_pca_flat = [item[0] for item in ppcomponents_pca]  # Flatten the inner arrays

    # Create a DataFrame from the data
    bar_data = {'Country': countries, 'Oja': ppcomponents_oja, 'PCA': ppcomponents_pca_flat}
    df = pd.DataFrame(bar_data)

    # Melt the DataFrame to create separate bars for Oja and PCA
    df_melted = pd.melt(df, id_vars=['Country'], value_vars=['Oja', 'PCA'], var_name='Method',
                        value_name='Principal Component')

    # Create a grouped bar chart using Plotly Express
    fig = px.bar(df_melted, x='Country', y='Principal Component', color='Method',
                 barmode='group', labels={'Method': 'Method'}, title='Principal Components by Country')

    # Show the plot
    fig.show()


def weight_comparison_graph(oja_weights, pca_weights):
    attributes = ["area", "gdp", "inflation", "life_expectancy", "military", "pop_growth", "unemployment"]

    pca_weights = pca_weights[0]

    if oja_weights[0] * pca_weights[0] < 0:
        pca_weights = [-value for value in pca_weights]

    # Create a DataFrame from the data
    bar_data = {'Attribute': attributes, 'Oja': oja_weights, 'PCA': pca_weights}
    df = pd.DataFrame(bar_data)

    # Melt the DataFrame to create separate bars for Oja and PCA
    df_melted = pd.melt(df, id_vars=['Attribute'], value_vars=['Oja', 'PCA'], var_name='Method',
                        value_name='Weight')

    # Create a grouped bar chart using Plotly Express
    fig = px.bar(df_melted, x='Attribute', y='Weight', color='Method',
                 barmode='group', labels={'Method': 'Method'}, title='Principal Components by Country')

    # Show the plot
    fig.show()


if __name__ == "__main__":
    data = load_data("./training_data/europe.csv")

    print(data)

    with open("config_ej1.2.json") as file:
        config = json.load(file)

    perceptron = LinearPerceptron()

    # ambas formas de estandarizar son equivalentes
    # training_data = standardize_matrix([row[1:] for row in data])
    training_data = StandardScaler().fit_transform([row[1:] for row in data])

    weights = perceptron.train(training_data, 0.001, 10000)

    pca = PCA(n_components=1)
    principalComponents = pca.fit_transform(training_data)

    first_component = []
    for i in range(len(training_data)):
        first_component.append(perceptron.compute_activation(training_data[i]))

    oja_pc1_graph(first_component, principalComponents, [row[0] for row in data])

    weight_comparison_graph(weights, pca.components_)



