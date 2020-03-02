"""
Simple implementation of multiclass logistic regression
using numpy.

Author: Caio Martins

Date: 2020-03-02
"""

import numpy as np

class MulticlassLogisticRegression:
    """
    Multiclass logreg scikit-learn-like with softmax.

    Author:
        Caio Martins
    """

    def __init__(self, learning_rate=1e-3, num_epochs=10000, verbose=True, print_epoch=1000):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.print_epoch = print_epoch

    def _feedforward(self, X):
        Y_proba = np.exp(X.dot(self.W) + self.b)/np.exp(X.dot(self.W) + self.b).sum(axis=1,keepdims=True)
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
        return - (T * np.log(Y_proba)).sum()

    def _get_dims(self, X, y):
        self.K = np.unique(y).size
        self.N, self.D = X.shape

    def _initialize_weights(self):
        self.W = np.random.randn(self.D,self.K)
        self.b = np.random.randn(self.K)

    def fit(self, X, y=None):
        self._get_dims(X, y)
        self._initialize_weights()

        T = np.array([[0 if y[j] != i else 1 for i in range(self.K)] for j in range(self.N)])

        self.costs = []
        self.clf_rates = []

        for epoch in range(self.num_epochs):
            
            Y_proba = self._feedforward(X)
            
            if epoch % self.print_epoch == 0:
                self.costs.append(self._logloss(T, Y_proba))
                self.clf_rates.append(self.score(y, Y_proba))
                if self.verbose:
                    print("cost: ", self.costs[-1], "accuracy: ", self.clf_rates[-1])

            self.W += self.learning_rate * self._gradient_weights(X, Y_proba, T)
            self.b += self.learning_rate * self._gradient_biases(Y_proba, T)


    def predict_proba(self, X):
        return self._feedforward(X)

    def predict(self, X):
        Y_proba = self._predict_proba(X)
        return np.argmax(Y_proba, axis=1)

        