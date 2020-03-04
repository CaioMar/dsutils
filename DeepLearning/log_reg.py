"""
Simple implementation of multiclass logistic regression
using numpy.

Author: Caio Martins

Date: 2020-03-02
"""

import numpy as np
from sklearn.utils import shuffle

class MulticlassLogisticRegression:
    """
    Multiclass logreg scikit-learn-like with softmax.

    Author:
        Caio Martins
    """

    def __init__(self, learning_rate=1e-3, num_epochs=10000, verbose=True, print_epoch=1000, eps=1e-5,
                 gd_type='full', batch_number = 10, mu = 0):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.print_epoch = print_epoch
        self.eps = eps
        self.gd_type = gd_type
        self.batch_number = batch_number
        self.mu = mu

    def _feedforward(self, X, best=False):
        Y_proba = None
        if best:
            Y_proba = (np.exp(X.dot(self.best_W) + self.best_b)/
                       np.exp(X.dot(self.best_W) + self.best_b).sum(axis=1,keepdims=True))
        else:
            Y_proba = (np.exp(X.dot(self.W) + self.b)/
                       np.exp(X.dot(self.W) + self.b).sum(axis=1,keepdims=True))
        return Y_proba
    
    def _gradient_weights(self, X, Y_proba, T):
        return X.T.dot(T - Y_proba)

    def _gradient_biases(self, Y_proba, T):
        return (T - Y_proba).sum(axis=0)

    def score(self, y, Y_proba):
        Y_pred = np.argmax(Y_proba, axis=1)
        accuracy = (y == Y_pred).sum()/y.shape[0]
        return accuracy

    def _logloss(self, T, Y_proba):
        return - (T * np.log(np.clip(Y_proba,self.eps,1-self.eps))).sum()

    def _get_dims(self, X, y):
        self.K = np.unique(y).size
        self.N, self.D = X.shape

    def _initialize_weights(self):
        self.W = np.random.randn(self.D,self.K) / np.sqrt(self.D)
        self.b = np.zeros(self.K)

    def fit(self, X, y=None):
        self._get_dims(X, y)
        self._initialize_weights()

        T = np.array([[0 if y[j] != i else 1 for i in range(self.K)] for j in range(self.N)])

        self.costs = []
        self.clf_rates = []
        self.best_W = self.W
        self.best_b = self.b

        if self.gd_type == 'full':

            for epoch in range(self.num_epochs):
                
                Y_proba = self._feedforward(X)

                self.costs.append(self._logloss(T, Y_proba))
                self.clf_rates.append(self.score(y, Y_proba))
                
                if epoch % self.print_epoch == 0:
                    if self.verbose:
                        print("cost: ", self.costs[-1], "accuracy: ", self.clf_rates[-1])

                if self._logloss(T, Y_proba) <= np.min(self.costs):
                    self.best_W = self.W
                    self.best_b = self.b

                self.W += self.learning_rate * self._gradient_weights(X, Y_proba, T)
                self.b += self.learning_rate * self._gradient_biases(Y_proba, T)

        elif self.gd_type == 'batch':
            
            batch_size = int(self.N/self.batch_number*1.0)

            vW = 0
            vb = 0

            for epoch in range(self.num_epochs):
                
                X, y = shuffle(X,y)
                T = np.array([[0 if y[j] != i else 1 for i in range(self.K)] for j in range(self.N)])

                for l in range(self.batch_number):

                    x = X[batch_size*l:(batch_size)*(l+1)]
                    y_proba = self._feedforward(x)
                    t = T[batch_size*l:(batch_size)*(l+1)]
                  
                    vW += self.mu*vW + self.learning_rate * self._gradient_weights(x, y_proba, t)
                    vb += self.mu*vb + self.learning_rate * self._gradient_biases(y_proba, t)

                    self.W += vW
                    self.b += vb

                    Y_proba = self._feedforward(X)

                    self.costs.append(self._logloss(T, Y_proba))
                    self.clf_rates.append(self.score(y, Y_proba))

                    if (epoch*self.batch_number+l) % self.print_epoch == 0:
                        if self.verbose:
                            print("cost: ", self.costs[-1], "accuracy: ", self.clf_rates[-1])

                    if self._logloss(T, Y_proba) <= np.min(self.costs):
                        self.best_W = self.W
                        self.best_b = self.b            


    def predict_proba(self, X):
        return self._feedforward(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

        