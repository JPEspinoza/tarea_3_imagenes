from scipy.cluster.vq import kmeans
from scipy.sparse import data
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

#leer datos y botar hu5 y 6 que no se piden
df = pd.read_csv("properties.csv")
df = df.drop(labels=["hu5", "hu6"], axis="columns")
column_name = df.columns

df = MinMaxScaler().fit_transform(df)
df = pd.DataFrame(df)
df.columns = column_name


#reducir a 128 observaciones
codebook, _ = kmeans(df, 128)

#guardar en un csv
codebook = pd.DataFrame(codebook)
codebook.columns = column_name
#codebook.to_csv("vq.csv", index=False)

# graficar vq sobre todos puntos en hu4 vs contrast
plt.scatter(df[["hu4"]], df[["contrast"]])
plt.scatter(codebook[["hu4"]], codebook[["contrast"]], c="red")
plt.xlabel("hu4")
plt.ylabel("contrast")
plt.show()

#pca, lo vamos a usar despues
pca = PCA(3)
pca.fit(df)

###graficar datos originales
data_pca = pca.transform(df).T
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(data_pca[0], data_pca[1], data_pca[2])
plt.show()
data_pca = pd.DataFrame(data_pca.T)
sns.pairplot(data_pca)
plt.show()

###graficar datos clusterizados
data_pca = pca.transform(codebook).T
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(data_pca[0], data_pca[1], data_pca[2], c="red")
plt.show()
data_pca = pd.DataFrame(data_pca.T)
sns.pairplot(data_pca)
plt.show()