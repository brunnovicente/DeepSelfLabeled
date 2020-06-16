import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DSL import DeepSelfLabeled
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from pyitlib import discrete_random_variable as ITL

T = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

PL = pd.read_csv('PL.csv')
PU = pd.read_csv('PU.csv')

indices = np.arange(np.size(PL, axis=0) + np.size(PU, axis=0))
D = pd.DataFrame(pd.concat([PL.drop(['g', 'classe'], axis=1), PU.drop(['g'], axis=1)]).values, index=indices)
D['g'] = pd.concat([PU['g'], PL['g']]).values


def degrau(x):
    if(x > 0.5):
        return 1.
    else:
        return 0.

def calcular_classe(P):
    classe = 0.
    for i,p in enumerate(P):
        classe += i*p
    return classe
    
    
def calcular_I(x, PL, k, t, c):
    X = PL.drop(['classe'], axis=1).values
    
    div = []
    
    #CALCULANDO A DIVERGÊNCIA PARA CADA UMA DAS AMOSTRAS ROTULADAS
    for xe in X:
        div.append(ITL.divergence_kullbackleibler_pmf(x, xe))
    
    
    #CALCULANDO A LISTA ELEGÍVEL A I
    D = PL.copy()
    D['div'] = div
    D = D.sort_values(by='div')
    I = D.iloc[0:k,:]
    
    #TESTA O VALOR DE t
    for i in I['div'].values:
        if(i > t):
           return -1
    
    #CALCULA A CLASSE
    P = []
    for v in np.arange(c):
        q = (I['classe'] == v).sum()
        p = q / k
        P.append(degrau(p))
    
    return int(calcular_classe(P))

k = 5
t = 0.1
c = 3
for g in np.arange(c):
    print('Grupo ', g)
    DU = PU[PU['g'] == g]
    DL = PL[PL['g'] == g]
    
    indices = DU.index.values
    
    DU = DU.drop(['g'], axis=1)
    DL = DL.drop(['g'], axis=1)
    
    respostas = []
    
    for x in DU.values:
        respostas.append(calcular_I(x, DL, k, t, c))
    
    PU.at[indices, 'g'] = respostas

UZ = pd.read_csv('UZ.csv')
T_results = T.fit_transform(UZ.drop(['y'], axis=1).values)

grupos = PU['g'].values
cores = ['black', 'yellow', 'red', 'green', 'gray', 'blue', 'orange', 'pink', 'brown', 'purple']
for i, tn in enumerate(T_results):
    plt.scatter(tn[0], tn[1], c=cores[grupos[i]], s=5)
    
    
    
    
    
    
    
    