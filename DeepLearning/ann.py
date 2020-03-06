"""
Simple implementations of regular artificial neural networks
using numpy.

Author: Caio Martins

Date: 2020-03-02
"""

import numpy as np
from sklearn.utils import shuffle

class ANNClassifier:
    """
    ANN scikit-learn-like with softmax.

    Author:
        Caio Martins
    """

    def __init__(self, hidden_layers_dims = None, activation = 'sigmoid', 
                 learning_rate=1e-3, num_epochs=10000, 
                 verbose=True, print_epoch=1000, eps=1e-5,
                 gd_type='full', batch_number = 10, mu = 0,
                 regularization = 0):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.print_epoch = print_epoch
        self.eps = eps
        self.gd_type = gd_type
        self.batch_number = batch_number
        self.mu = mu
        self.activation = activation
        self.regularization = regularization
        self.hidden_layers_dims = []
        if hidden_layers_dims != None:
            self.hidden_layers_dims = hidden_layers_dims

    @staticmethod
    def sigmoid(x):
        s = 1/(1+np.exp(-x))
        return s

    def _feedforward(self, X, best=False):
        self.Zs = [X]
        for i, W, b in list(zip(range(len(self.Ws)),self.Ws, self.bs))[:-1]:
            if self.activation == 'sigmoid':
                self.Zs.append(self.sigmoid(self.Zs[i].dot(W) + b))
            elif self.activation == 'relu':
                tmp = self.Zs[i].dot(W) + b
                self.Zs.append(np.where(tmp < 0, 0, tmp))

        Y_proba = None
        
        if best:
            activation = np.exp(self.Zs[-1].dot(self.best_Ws[-1]) + self.best_bs[-1])
        else:
            activation = np.exp(self.Zs[-1].dot(self.Ws[-1]) + self.bs[-1])
        Y_proba = (activation/activation.sum(axis=1,keepdims=True))
        return np.clip(Y_proba,self.eps,1-self.eps)
    
    def _gradient_weights(self, X, Y_proba, T, layer):
        derror = T - Y_proba
        final_layer = len(self.Zs) - 1
        if layer == final_layer:
            return self.Zs[final_layer].T.dot(derror)
        else:
            for l in range(final_layer,layer,-1):
                if self.activation == 'sigmoid':
                    derror = derror.dot(self.Ws[l].T)*self.Zs[l]*(1-self.Zs[l])
                elif self.activation == 'relu':
                    derror = derror.dot(self.Ws[l].T)*np.sign(self.Zs[l])
            return self.Zs[layer].T.dot(derror)

    def _gradient_biases(self, Y_proba, T, layer):
        derror = T - Y_proba
        final_layer = len(self.Zs) - 1
        if layer == final_layer:
            return derror.sum(axis=0)
        else:
            for l in range(final_layer,layer,-1):
                if self.activation == 'sigmoid':
                    derror = derror.dot(self.Ws_old[l].T)*self.Zs[l]*(1-self.Zs[l])
                elif self.activation == 'relu':
                    derror = derror.dot(self.Ws_old[l].T)*np.sign(self.Zs[l])
            return derror.sum(axis=0)

    def score(self, y, Y_proba):
        Y_pred = np.argmax(Y_proba, axis=1)
        accuracy = (y == Y_pred).sum()/y.shape[0]
        return accuracy

    def _logloss(self, T, Y_proba):
        return (- (T * np.log(np.clip(Y_proba,self.eps,1-self.eps))).sum() +
                   sum([(self.regularization*np.square(w)).sum() for w in self.Ws]) +
                   sum([(self.regularization*np.square(b)).sum() for b in self.bs]))
                   
    def _get_dims(self, X, y):
        self.K = np.unique(y).size
        self.N, self.D = X.shape
        self.layers_dims = [self.D] + self.hidden_layers_dims.copy() + [self.K]
        self.dims = [(self.layers_dims[i],self.layers_dims[i+1])
                     for i in range(len(self.layers_dims)-1)]

    def _initialize_weights(self):
        self.Ws = [np.random.randn(dim1, dim2) / np.sqrt(dim1)
                                    for dim1, dim2 in self.dims]
        self.bs = [np.zeros(dim2) for _, dim2 in self.dims]

    def fit(self, X, y=None):
        self._get_dims(X, y)
        self._initialize_weights()

        T = np.array([[0 if y[j] != i else 1 for i in range(self.K)] for j in range(self.N)])\
                .astype(np.int32)

        self.costs = []
        self.clf_rates = []
        self.best_Ws = self.Ws.copy()
        self.best_bs = self.bs.copy()

        if self.gd_type == 'full':

            for epoch in range(self.num_epochs):
                
                Y_proba = self._feedforward(X)

                self.costs.append(self._logloss(T, Y_proba))
                self.clf_rates.append(self.score(y, Y_proba))
                
                if epoch % self.print_epoch == 0:
                    if self.verbose:
                        print("cost: ", self.costs[-1], "accuracy: ", self.clf_rates[-1])

                if self._logloss(T, Y_proba) <= np.min(self.costs):
                    self.best_Ws = self.Ws.copy()
                    self.best_bs = self.bs.copy()

                for ll in range(len(self.Ws)-1,-1,-1):
                    self.Ws[ll] += self.learning_rate * (self._gradient_weights(X, Y_proba, T, ll) 
                                                         - self.regularization * self.Ws[ll])
                    self.bs[ll] += self.learning_rate * (self._gradient_biases(Y_proba, T, ll)
                                                         - self.regularization * self.bs[ll])

        elif self.gd_type == 'batch':
            
            batch_size = int(self.N/self.batch_number*1.0)

            vW = [0 for i in range(len(self.Ws))]
            vb = [0 for i in range(len(self.Ws))]

            for epoch in range(self.num_epochs):
                
                X, y = shuffle(X,y)
                T = np.array([[0 if y[j] != i else 1 for i in range(self.K)] for j in range(self.N)])

                for l in range(self.batch_number):

                    x = X[batch_size*l:(batch_size)*(l+1)]
                    y_proba = self._feedforward(x)
                    t = T[batch_size*l:(batch_size)*(l+1)]
                  
                    self.Ws_old = self.Ws.copy()

                    for ll in range(len(self.Ws)-1,-1,-1):

                        vW[ll] += self.mu*vW[ll] + self.learning_rate * \
                                                (self._gradient_weights(x, y_proba, t, ll) -
                                                 self.regularization * self.Ws_old[ll])
                        vb[ll] += self.mu*vb[ll] + self.learning_rate * \
                                                (self._gradient_biases(y_proba, t, ll) -
                                                 self.regularization * self.bs[ll])

                        self.Ws[ll] += vW[ll]
                        self.bs[ll] += vb[ll]

                    Y_proba = self._feedforward(X)

                    self.costs.append(self._logloss(T, Y_proba))
                    self.clf_rates.append(self.score(y, Y_proba))

                    if (epoch*self.batch_number+l) % self.print_epoch == 0:
                        if self.verbose:
                            print("cost: ", self.costs[-1], "accuracy: ", self.clf_rates[-1])

                    if self._logloss(T, Y_proba) <= np.min(self.costs):
                        self.best_Ws = self.Ws.copy()
                        self.best_bs = self.bs.copy()            


    def predict_proba(self, X):
        return self._feedforward(X, best=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

        