"""
Simple implementation of multiclass logistic regression
using tensorflow.

Author: Caio Martins

Date: 2020-03-02
"""

import numpy as np
import tensorflow as tf
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

    def score(self, y, y_pred):
        accuracy = (y == y_pred).sum()/y.shape[0]
        return accuracy

    def _get_dims(self, X, y):
        self.K = np.unique(y).size
        self.N, self.D = X.shape

    def _initialize_weights(self):
        self.W = np.random.randn(self.D,self.K) / np.sqrt(self.D)
        self.b = np.zeros(self.K)

    def fit(self, X, y=None):
        self._get_dims(X, y)
        self._initialize_weights()

        T = np.array([[0 if y[j] != i else 1 for i in range(self.K)] for j in range(self.N)]).astype(np.float32)

        inputs = tf.placeholder(tf.float32, shape=[None,self.D], name='inputs')
        labels = tf.placeholder(tf.float32, shape=[None,self.K], name='labels')
        
        W = tf.Variable(self.W.astype(np.float32), name="W")
        b = tf.Variable(self.b.astype(np.float32), name="b")

        Z = tf.matmul(inputs, W) + b #logits

        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z, labels=labels))

        train_op = tf.train.GradientDescentOptimizer(self.learning_rate, name='train').minimize(cost)
        self.pred_op = tf.arg_max(Z, 1)


        self.costs = []
        self.clf_rates = []

        if self.gd_type == 'full':
    
            init = tf.global_variables_initializer()    
            with tf.Session() as session:

                session.run(init)

                for epoch in range(self.num_epochs):
                    print("epoch: %d" % epoch, X.shape)
                    session.run('train', feed_dict = {'inputs:0': X.reshape((-1,self.D)), 'labels:0': T.reshape((-1,self.K))})
                    y_pred = session.run(self.pred_op, feed_dict = {"inputs:0": X.reshape((-1,self.D))})

                    self.costs.append(cost.eval())
                    self.clf_rates.append(self.score(y, y_pred))
                    
                    if epoch % self.print_epoch == 0:
                        if self.verbose:
                            print("epoch: ", epoch,
                                  "cost: ", self.costs[-1],
                                  "accuracy: ", self.clf_rates[-1])

        # elif self.gd_type == 'batch':
            
        #     batch_size = int(self.N/self.batch_number*1.0)

        #     vW = 0
        #     vb = 0

        #     for epoch in range(self.num_epochs):
                
        #         X, y = shuffle(X,y)
        #         T = np.array([[0 if y[j] != i else 1 for i in range(self.K)] for j in range(self.N)])

        #         for l in range(self.batch_number):

        #             x = X[batch_size*l:(batch_size)*(l+1)]
        #             y_proba = self._feedforward(x)
        #             t = T[batch_size*l:(batch_size)*(l+1)]
                  
        #             vW += self.mu*vW + self.learning_rate * self._gradient_weights(x, y_proba, t)
        #             vb += self.mu*vb + self.learning_rate * self._gradient_biases(y_proba, t)

        #             self.W += vW
        #             self.b += vb

        #             Y_proba = self._feedforward(X)

        #             self.costs.append(self._logloss(T, Y_proba))
        #             self.clf_rates.append(self.score(y, Y_proba))

        #             if (epoch*self.batch_number+l) % self.print_epoch == 0:
        #                 if self.verbose:
        #                     print("cost: ", self.costs[-1], "accuracy: ", self.clf_rates[-1])

        #             if self._logloss(T, Y_proba) <= np.min(self.costs):
        #                 self.best_W = self.W
        #                 self.best_b = self.b            


    # def predict(self, X):
    #     init = tf.global_variables_initializer()
    #     with tf.Session as session:
    #         session.run(init)
    #         y_pred = session.run(self.pred_op, feed_dict={"X":X})
    #     return y_pred

        