import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

data = pd.read_csv('properties.csv')
data = data.drop(labels=["hu5", "hu6"], axis="columns")
data.head()


data = MinMaxScaler().fit_transform(data)
pca = PCA(n_components=3)
pca.fit(data)

data_pca = pca.transform(data).T
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)


#https://plotly.com/python/pca-visualization/
pca = PCA(3)
components = pca.fit_transform(data)
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

fig = px.scatter_matrix(
    components,
    labels=labels,
    dimensions=range(3)
    #,color = data_pca[0]
)
fig.update_traces(diagonal_visible=False)
fig.show()