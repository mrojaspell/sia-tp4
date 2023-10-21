import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import pandas as pd

from main_ej1 import load_data

def standardize_matrix(matrix):
    resp = []

    matrix = np.transpose(matrix)

    for row in matrix:
        resp.append(np.array(stats.zscore(row)))

    return np.array(np.transpose(resp))

def oja_PC1_graph(ppcomponents, countries):
    values = [item for sublist in ppcomponents for item in sublist]

    df = pd.DataFrame({'Countries': countries, 'PC1 Value': values})
    fig = px.bar(df, x='Countries', y='PC1 Value', title='PC1 Value for each country')
    fig.show()

if __name__ == "__main__":
    data = load_data("./training_data/europe.csv")

    #Ambas formas de estandarizar son equivalentes
    training_data = StandardScaler().fit_transform([row[1:] for row in data])
    #training_data = standardize_matrix([row[1:] for row in data])

    pca = PCA(n_components=1)
    principalComponents = pca.fit_transform(training_data)

    # print(pca.components_)
    print(principalComponents)
    #extract the names of countries which are on the first element of the list coming from load_data function as a second argument for function oja_PC1_graph
    oja_PC1_graph(principalComponents,  [row[0] for row in data])
