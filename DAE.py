import sys
from time import time
import numpy as np
import pandas as pd
import keras.backend as K
from keras.initializers import RandomNormal
from keras.engine.topology import Layer, InputSpec
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import SGD
from sklearn.preprocessing import normalize
from keras.callbacks import LearningRateScheduler
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.decomposition import PCA
from numpy import linalg
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from keras.utils import np_utils

class DeepAutoEncoder(object):
    
    def __init__(self, dim, k, epocas=100, lote = 256):
        
        self.epocas = epocas
        self.dim = dim
        self.k = k
        self.lote = 256
        
        input_img = Input((dim,))
        #encoded = Dense(32, activation='relu')(input_img)
        #drop = Dropout(0.2)(encoded)
        #encoded = Dense(10, activation='relu')(encoded)
        #drop = Dropout(0.2)(encoded)
        #encoded = Dense(32, activation='relu')(encoded)
        
        Z = Dense(10, activation='relu')(input_img)
        
        #decoded = Dense(32, activation='relu')(Z)
        #drop = Dropout(0.2)(decoded)
        #decoded = Dense(64, activation='relu')(decoded)
        #drop = Dropout(0.2)(decoded)
        #decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(dim, activation='sigmoid')(Z)
                        
        self.encoder = Model(input_img, Z)
        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(loss='mse', optimizer='adadelta')
        self.autoencoder.summary()
        
    def fit(self, X):
        self.autoencoder.fit(X,X, epochs=self.epocas, batch_size=self.lote, shuffle=True)
        