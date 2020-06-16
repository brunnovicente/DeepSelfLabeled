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
from pyitlib import discrete_random_variable as ITL

"""
    ESTRUTURA DO DEEP AUTOENCODER
"""
def DAE(dims, act='relu', init='glorot_uniform'):
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')

"""
    ACURÁCIA DO AGRUPAMENTO
"""
def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

"""
    Classe que define a Camada de Rotulação do Modelo
"""
class LabelingLayer(Layer):
    
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super(LabelingLayer, self).__init__(**kwargs)
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(LabelingLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
        
    def call(self, x, mask=None):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(x, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(LabelingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

"""
    CLASSE QUE DEFINE O MODELO PROPOSTO
"""  
class DeepSelfLabeled(object):
    
    def __init__(self,
                 n_clusters = 10,
                 dims = [],
                 epocas = 100,
                 alpha=1.0,
                 batch_size=256,
                 t = 0.01,
                 k = 5,
                 **kwargs):

        super(DeepSelfLabeled, self).__init__()
        
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1
        self.t = t
        self.k = k
        
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = DAE(self.dims)

        # prepare DEC model
        Labeling_layer = LabelingLayer(self.n_clusters, name='labeling')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=Labeling_layer)
        self.model.summary()
    
    def agrupamento(self, X, optimizer='adam', epocas=200, lote=256):
        #Inicialização do Modelo
        print('INICIALIZANDO...')
        self.pretrain(X,epochs=epocas, batch_size=lote)
        self.gruposU = self.clustering(X)
        return self.gruposU
    
    def train(self, L, U, y, optimizer='adam', epocas=200, lote=256):
        
        #Inicialização do Modelo
        print('INICIALIZANDO...')
        self.pretrain(U,epochs=epocas, batch_size=lote)
        self.gruposU = self.clustering(U)
        self.c = np.size(np.unique(y))
        
        #Gerando as distribuições de probabilidades
        self.gruposL = self.predict(L)
        PU = self.model.predict(U)
        PL = self.model.predict(L)
        
        PL = pd.DataFrame(PL)
        PL['g'] = self.predict(L)
        PL['classe'] = y
        PU = pd.DataFrame(PU)
        PU['g'] = self.predict(U)
        
        print('CALCULANDO AGRUPAMENTO...')        
        #Calcula a primeira etapa da divisão de grupos
        for g in np.arange(self.c):
            print('Grupo ', g)
            DU = PU[PU['g'] == g]
            DL = PL[PL['g'] == g]
            
            indices = DU.index.values
            
            DU = DU.drop(['g'], axis=1)
            DL = DL.drop(['g'], axis=1)
            
            respostas = []
            
            for x in DU.values:
                respostas.append(self.calcular_classe(x, DL, self.k, self.t, self.c))
            
            PU.at[indices, 'g'] = respostas
        
        print('FASE ROTULAÇÃO...')
        #Fase de Rotulação
        D = pd.DataFrame(np.concatenate((U, L), axis=0))
        #D = pd.DataFrame(pd.concat([PL.drop(['g', 'classe'], axis=1), PU.drop(['g'], axis=1)]).values, index=indices)
        D['g'] = pd.concat([PU['g'], PL['g']]).values
        
        for e in np.arange(epocas):
            print('.... Epoca ', e)
            Un = D[D['g'] == -1]
            Ln = D[D['g'] != -1]
            indices = Un.index.values
            
            Un['g'] = self.model.predict(Un.drop(['g'], axis=1).values)
            Ln['g'] = self.model.predict(Ln.drop(['g'], axis=1).values)
            
            for g in np.arange(self.c):
                #print('Grupo ', g)
                DU = Un[Un['g'] == g]
                DL = Ln[Ln['g'] == g]
                
                indices = DU.index.values
                
                DU = DU.drop(['g'], axis=1)
                DL = DL.drop(['g'], axis=1)
                
                respostas = []
                
                for x in DU.values:
                    respostas.append(self.calcular_classe(x, DL, self.k, self.t, self.c))
                
                D.at[indices, 'g'] = respostas            
                
        return D.iloc[0:np.size(U, axis=0) , -1].values
        
    """
        Calcula o vetor I de uma amostra
    """
    def calcular_classe(self, x, PL, k, t, c):
        indices = PL.index.values
        X = PL.drop(['classe'], axis=1).values
        div = []
        
        #CALCULANDO A DIVERGÊNCIA PARA CADA UMA DAS AMOSTRAS ROTULADAS
        for xe in X:
            div.append(ITL.divergence_kullbackleibler_pmf(x, xe))
        
        #CALCULANDO A LISTA ELEGÍVEL A I
        DL = PL.copy()    
        DL['div'] = div
        DL = DL.sort_values(by='div')
        I = DL.iloc[0:k,:]
        
        #TESTA O VALOR DE t
        for i in I['div'].values:
            if(i > t):
               return -1
        
        #CALCULA A CLASSE
        P = []
        for v in np.arange(c):
            q = (I['classe'] == v).sum()
            p = q / k
            P.append(int(self.degrau(p)))
        
        return self.classe(P)
   
    def degrau(self, x):
        if(x > 0.5):
            return 1.
        else:
            return 0.

    def classe(self, P):
        classe = 0.
        for i,p in enumerate(P):
            classe += i*p
        return classe   
            
    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256):
        print('...Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')
        # begin pretraining
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, verbose=False)
        print('Pretraining time: %ds' % round(time() - t0))
        self.pretrained = True
    
    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)
    
    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def clustering(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3, update_interval=140):

        #print('Update interval', update_interval)
        save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        #print('Save interval', save_interval)

        # Step 1: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='labeling').set_weights([kmeans.cluster_centers_])

        p = self.model.predict(x, verbose=0)
        y = q = np_utils.to_categorical(p.argmax(1))
        
        self.compile()
        
        self.model.fit(x, y, epochs=200, verbose=False)

        return self.predict(x)