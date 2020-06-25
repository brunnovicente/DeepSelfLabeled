import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from DAE import DeepAutoEncoder

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
sca = MinMaxScaler()

dados = pd.read_csv('D:/Drive UFRN/bases/mnist64.csv')
#dados = dados[dados['classe'] < 5]
X = dados.drop(['classe'], axis = 1).values
Y = dados['classe'].values
L, U, y, yu = train_test_split(X, Y, train_size=0.1, test_size=0.9, stratify=Y)

U = sca.fit_transform(U)
L = sca.transform(L)

dae = DeepAutoEncoder(np.size(X, axis=1), 10, lote=128, epocas=500)
dae.fit(U)

UZ = dae.encoder.predict(U)
Ul = dae.autoencoder.predict(U)

def softmax(x):
    num = np.exp(x)
    den = np.sum(num)
    return num / den

R = []

for xi in UZ:
    s = softmax(xi)
    R.append(s)

R = np.array(R)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(U[i].reshape(8, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(UZ[i].reshape(1, 10).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n*2)
    plt.imshow(Ul[i].reshape(8, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
tsne_results = tsne.fit_transform(UZ)
grupos = R.argmax(1)
cores = ['black', 'yellow', 'red', 'green', 'gray', 'blue', 'orange', 'pink', 'brown', 'purple']
for i, T in enumerate(tsne_results):
    print(i)
    plt.scatter(T[0], T[1], c=cores[int(grupos[i])], s = 2)