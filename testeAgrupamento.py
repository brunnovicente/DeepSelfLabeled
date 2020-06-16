import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DSL import DeepSelfLabeled
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sca = MinMaxScaler()

dados = pd.read_csv('D:/Drive UFRN/bases/mnist.csv')

X = sca.fit_transform(dados.drop(['classe'], axis=1))

kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
grupos = kmeans.predict(X)

s = silhouette_score(X, grupos)

dimensao = [X.shape[-1], 250, 250, 500, 10]
dsl = DeepSelfLabeled(n_clusters=10, dims=dimensao, epocas=100, alpha=1.0, t = 0.1, k = 5)
rotulos = dsl.agrupamento(X, optimizer='adam', epocas=200, lote=256)

s1 = silhouette_score(X, rotulos)