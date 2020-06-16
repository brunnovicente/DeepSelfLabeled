import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DSL import DeepSelfLabeled
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
sca = MinMaxScaler()

dados = pd.read_csv('D:/basedados/mnist64.csv')
#dados = dados[dados['classe'] < 5]
X = dados.drop(['classe'], axis = 1).values
Y = dados['classe'].values
L, U, y, yu = train_test_split(X, Y, train_size=0.1, test_size=0.9, stratify=Y)

U = sca.fit_transform(U)
L = sca.transform(L)

dimensao = [X.shape[-1], 250, 250, 500, 10]
dsl = DeepSelfLabeled(n_clusters=10, dims=dimensao, epocas=100, alpha=1.0, t = 0.1, k = 5)
rotulos = dsl.train(L, U, y, optimizer='adam', epocas=200, lote=256)

#PL.to_csv('PL.csv', index=False)
#PU.to_csv('PU.csv', index=False)

UZ = pd.DataFrame(dsl.encoder.predict(U))
#UZ['y'] = yu
#LZ = pd.DataFrame(dsl.encoder.predict(L))
#LZ['y'] = y
#LZ.to_csv('LZ.csv', index=False)
#UZ.to_csv('UZ.csv', index=False)

encoded_imgs = dsl.autoencoder.predict(L)

tsne_results = tsne.fit_transform(UZ.values)
#grupos = PU['g'].values
cores = ['black', 'yellow', 'red', 'green', 'gray', 'blue', 'orange', 'pink', 'brown', 'purple']
for i, T in enumerate(tsne_results):
    plt.scatter(T[0], T[1], c=cores[int(yu[i])])


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(L[i].reshape(8, 8))
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
    plt.imshow(encoded_imgs[i].reshape(8, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)