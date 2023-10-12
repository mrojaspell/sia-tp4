import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go


data = pd.read_csv('../training_data/europe.csv')

features = data[['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']]

# Standardize data -> substract mean and divide by standard deviation
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# n_components is the components to keep, 2 for 2D visualization
pca = PCA()
pca_result = pca.fit_transform(scaled_features)

explained_variance = pca.explained_variance_ratio_
print(f"The variance of each component is {explained_variance}")
print(pca.components_)



# # Create a scatter plot of the scores on PC1 vs PC2
# plt.figure(figsize=(10, 8))
# plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
#
# # Add arrows to represent the variable loadings
# for i, var in enumerate(pca.components_.T):
#     plt.arrow(0, 0, var[0], var[1], color='r', alpha=0.5)
#     plt.text(var[0], var[1], features.columns[i], fontsize=12)
#
# # Label the axes
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
#
# # Show the biplot
# plt.grid()
# plt.show()


#
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'])

# Create a biplot using Plotly Express
biplot = px.scatter(
    pca_df, x='PC1', y='PC2',
    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
    title='PCA Biplot'
)
colors = px.colors.qualitative.Set1

# Add feature loadings to the biplot
loadings = pca.components_.T
loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'], index=features.columns)
for i, feature in enumerate(loadings_df.index):
    biplot.add_trace(go.Scatter(x=[0, loadings_df.loc[feature, 'PC1']], y=[0, loadings_df.loc[feature, 'PC2']],
                               mode='lines+text', line=dict(color=colors[i], width=1),
                               name=feature))
for i, country in enumerate(data['Country']):
    biplot.add_annotation(x=pca_df['PC1'][i], y=pca_df['PC2'][i], text=country, showarrow=False)
biplot.show()
