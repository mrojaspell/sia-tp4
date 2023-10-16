import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def biplot_graph(score, coeff, labels, label_names):
    xs = score[:, 0]
    ys = score[:, 1]

    # escalamos las variables
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    xs_scaled = xs * scalex
    ys_scaled = ys * scaley

    fig = px.scatter(
        x=xs_scaled, y=ys_scaled,
        text=labels,
        title="Biplot",
        labels={'x': 'PC1', 'y': 'PC2'}
    )

    for i in range(len(labels)):
        fig.add_annotation(x=xs_scaled[i], y=ys_scaled[i], text=labels[i], showarrow=False, font=dict(size=1))

    colors = px.colors.qualitative.Set2
    for i, label in enumerate(label_names):
        fig.add_shape(
            go.layout.Shape(
                type='line',
                x0=0,
                y0=0,
                x1=coeff[i, 0],
                y1=coeff[i, 1],
                line=dict(color=colors[i], width=2),
            )
        )
        fig.add_annotation(
            x=coeff[i, 0] * 1.15,
            y=coeff[i, 1] * 1.15,
            text=label,
            showarrow=False,
            font=dict(color=colors[i], size=10)
        )

    fig.update_xaxes(title_text="PC1")
    fig.update_yaxes(title_text="PC2")
    fig.update_layout(showlegend=False)

    return fig

def box_plot_graph(dataframe, title):
    fig = go.Figure()
    for col in dataframe.columns:
        fig.add_trace(go.Box(y=dataframe[col], name=col, boxmean=True, boxpoints=False))
    fig.update_xaxes(title='Variable')
    fig.update_layout(title=title)
    return fig

def main():
    df = pd.read_csv('../training_data/europe.csv')

    df_long = pd.melt(df, var_name='Column Name', value_name='Values')
    fig = px.box(df_long, x='Column Name', y='Values')
    fig.update_xaxes(tickangle=90, tickfont=dict(size=10))
    fig.update_layout(title_text='Before Standardizing')
    fig.show()

    features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)

    df_standardized = pd.DataFrame(x, columns=features)
    after_standardized = box_plot_graph(df_standardized, 'After Standardizing')
    after_standardized.show()

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    biplot_fig = biplot_graph(principalComponents, np.transpose(pca.components_[0:2, :]), df['Country'], features)
    biplot_fig.show()

    variance_explained = pca.explained_variance_ratio_
    fig = px.pie(
        values=variance_explained,
        names=["PC1", "PC2"],
        title="Variance of each component",
        labels={'labels': 'Component'}
    )
    fig.show()

    df_pc1 = pd.DataFrame({'Country': df['Country'], 'PC1': principalComponents[:, 0]})
    df_pc1 = df_pc1.sort_values(by='PC1', ascending=False)

    fig = px.bar(
        df_pc1,
        x='PC1',
        y='Country',
        orientation='h',
        title='PC1 values of each country',
        labels={'PC1': 'PC1 Values', 'Country': 'Country'},
    )
    fig.update_layout(xaxis_title="PC1 Values", yaxis_title="Country")
    fig.show()


main()
