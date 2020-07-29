import pandas as pd
import numpy as np
from pyitlib import discrete_random_variable as ITL
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity

sca = MinMaxScaler()

X = pd.read_csv('D:/basedados/iris.csv')
X1 = X[X['classe'] == 1]
X2 = X[X['classe'] == 2]
X3 = X[X['classe'] == 3]
y1 = X1['classe'].values
y2 = X2['classe'].values
y3 = X3['classe'].values

X1 = sca.fit_transform(X1.drop(['classe'], axis=1).values)
X2 = sca.transform(X2.drop(['classe'], axis=1).values)
X3 = sca.transform(X3.drop(['classe'], axis=1).values)


def kernel(a,h):
    x = -((a*a) / 2*(h*h))
    return np.exp(x)

def pdf(x, X, h):
    v = []
    for xi in X:
        xj = x - xi
        v.append(kernel(xj, h))
    return np.sum(v)/np.size(X, 0)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
            
a = X1[6,:]

p1 = pdf(a, X1, 4)
p2 = pdf(a, X2, 4)
p3 = pdf(a, X3, 4)
p = np.array([p1, p2, p3])
c = softmax(p)
