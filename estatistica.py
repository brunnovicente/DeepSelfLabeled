import pandas as pd
import numpy as np
from som import SOM
from minisom import MiniSom    
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
sca = MinMaxScaler()

dados = pd.read_csv('d:/basedados/iris.csv')
Y = dados['classe'].values -  1
X = sca.fit_transform(dados.drop(['classe'], axis=1))

som_shape = (5, 6)
som = MiniSom(som_shape[0], som_shape[1], 4, sigma=.5, learning_rate=.5,
              neighborhood_function='gaussian', random_seed=10)
som.train_batch(X, 1000, verbose=True)

winner_coordinates = np.array([som.winner(x) for x in X]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

X = tsne.fit_transform(X)

for c in np.unique(cluster_index):
    plt.scatter(X[cluster_index == c, 0],
                X[cluster_index == c,1], label='cluster='+str(c), alpha=.7)

for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                s=80, linewidths=35, color='k', label='centroid')

Xt = tsne.fit_transform(X)
cores = ['red', 'blue', 'green', 'yellow', 'black', 'pink', 'brown', 'orange', 'gray', 'purple']
for i,xi in enumerate(Xt):
    print(i)
    plt.scatter(xi[0], xi[1], color=cores[Y[i]], s=10)