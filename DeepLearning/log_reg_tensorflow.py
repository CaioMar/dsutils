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

    def __init__(self, learning_rate=1e-3, num_epochs=10000, verbose=True, print_epoch=1000, eps=1e-10,
                 gd_type='full', batch_number = 10, mu = 0, decay = 0.9):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.print_epoch = print_epoch
        self.eps = eps
        self.gd_type = gd_type
        self.batch_number = batch_number
        self.mu = mu
        self.decay = decay

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

        T = np.array([[0 if y[j] != i else 1 for i in range(self.K)] for j in range(self.N)])

        inputs = tf.placeholder(tf.float32, shape=[None,self.D], name='inputs')
        labels = tf.placeholder(tf.float32, shape=[None,self.K], name='labels')
        
        W = tf.Variable(self.W.astype(np.float32), name="W")
        b = tf.Variable(self.b.astype(np.float32), name="b")

        Z = tf.matmul(inputs, W) + b #logits

        saver = tf.train.Saver()

        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z, labels=labels))

        train_op = tf.train.RMSPropOptimizer(self.learning_rate,
                                             decay=self.decay,
                                             momentum=self.mu,
                                             epsilon=self.eps).minimize(cost)
        self.pred_op = tf.arg_max(Z, 1)


        self.costs = []
        self.clf_rates = []
        
        init = tf.global_variables_initializer()    
        with tf.Session() as session:

            session.run(init)

            if self.gd_type == 'full':

                for epoch in range(self.num_epochs):

                    session.run(train_op, feed_dict = {'inputs:0': X, 'labels:0': T})
                    y_pred = session.run(self.pred_op, feed_dict = {"inputs:0": X})
                    cost_val = session.run(cost, feed_dict = {"inputs:0": X, 'labels:0': T})    
                    
                    self.costs.append(cost_val)
                    self.clf_rates.append(self.score(y, y_pred))
                    
                    if epoch % self.print_epoch == 0:
                        if self.verbose:
                            print("epoch: ", epoch,
                                  "cost: ", self.costs[-1],
                                  "accuracy: ", self.clf_rates[-1])
                
                saver.save(session, './model/logreg_model',global_step=self.num_epochs)

            elif self.gd_type == 'batch':
            
                batch_size = int(self.N/self.batch_number*1.0)

                for epoch in range(self.num_epochs):
                    
                    X, y = shuffle(X,y)
                    T = np.array([[0 if y[j] != i else 1 for i in range(self.K)] for j in range(self.N)])

                    for l in range(self.batch_number):

                        x = X[batch_size*l:(batch_size)*(l+1)]
                        t = T[batch_size*l:(batch_size)*(l+1)]

                        session.run(train_op, feed_dict={'inputs:0':x, 'labels:0':t})  
                        y_pred = session.run(self.pred_op, feed_dict = {"inputs:0": X})
                        cost_val = session.run(cost, feed_dict = {"inputs:0": X, 'labels:0': T})    
                    
                        self.costs.append(cost_val)
                        self.clf_rates.append(self.score(y, y_pred))
                        
                        if (epoch*self.batch_number+l) % self.print_epoch == 0:
                            if self.verbose:
                                print("epoch: ", epoch,
                                    "cost: ", self.costs[-1],
                                    "accuracy: ", self.clf_rates[-1])
                    
                saver.save(session, './model/logreg_model',global_step=self.num_epochs*self.batch_number)

    def predict(self, X):
        with tf.Session() as session:
            if self.gd_type == 'full':
                new_saver = tf.train.import_meta_graph('./model/logreg_model-{}.meta'.format(self.num_epochs))
            else:
                new_saver = tf.train.import_meta_graph('./model/logreg_model-{}.meta'.format(self.num_epochs*self.batch_number))
            new_saver.restore(session, tf.train.latest_checkpoint('./model'))
            y_pred = session.run(self.pred_op, feed_dict={'inputs:0':X})
        return y_pred

        