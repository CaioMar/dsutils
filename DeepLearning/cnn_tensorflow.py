"""
Simple implementations of a convolutional neural network
using tensorflow (LeNet). Pool size (2,2).

Author: Caio Martins

Date: 2020-03-05
"""

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

class ConvPoolLayer:
    """
    Representation of convolution layer with pooling.
    """
    def __init__(self, K1, K2, M1, M2, layernum,
                 activation='relu', dropout_rate=1,
                 batch_normalization=False, alpha=0.9):
        self.M1 = M1
        self.M2 = M2
        self.K1 = K1 #Filter spatial dimensions
        self.K2 = K2 #Filter spatial dimensions
        self.activation = activation

        #Weight initialization
        W = np.random.randn(K1, K2, M1, M2) / np.sqrt(M1*K1*K2 + M2*K1*K2/4.0) #4.0 is due to poolsz
        b = np.zeros(M2)

        self.W = tf.Variable(W.astype(np.float32), name='W%d' %layernum)
        self.b = tf.Variable(b.astype(np.float32), name='b%d' %layernum)
 
        self.logit = None

    def _convpool(self, Input, Filter, b):
        C = tf.nn.conv2d(Input, Filter, strides=[1,1,1,1], padding="SAME")
        C = tf.nn.bias_add(C, b)
        M = tf.nn.max_pool(C, ksize = [1,2,2,1], strides=[1,2,2,1], padding="SAME")
        return M

    def forward(self, Z):

        self.out = self._convpool(Z, self.W, self.b)
        
        if self.activation == 'relu':
            return tf.nn.relu(self.out)
        elif self.activation == 'tanh':
            return tf.nn.tanh(self.out)
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid(self.out)


class HiddenLayer:
    """
    Representation of regular hidden layer.
    """
    def __init__(self, M1, M2, layernum,
                 activation='relu', dropout_rate=1,
                 batch_normalization=False, alpha=0.9):
        self.M1 = M1
        self.M2 = M2
        self.activation = activation
        self.dropout_rate = dropout_rate
        W = np.random.randn(self.M1,self.M2) / np.sqrt(self.M1)
        b = np.zeros(self.M2)

        self.W = tf.Variable(W.astype(np.float32), name='W%d' %layernum)
        self.b = tf.Variable(b.astype(np.float32), name='b%d' %layernum)
 
        self.logit = None

    def get_logit(self, Z):
        return tf.matmul(Z,self.W) + self.b
 
    def forward(self, Z, test=False):
        
        self.logit = self.get_logit(Z)

        if not test:
            self.logit = tf.nn.dropout(self.logit,
                                       keep_prob=self.dropout_rate)

        if self.activation == 'relu':
            return tf.nn.relu(self.logit)
        elif self.activation == 'tanh':
            return tf.nn.tanh(self.logit)
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid(self.logit)


class CNNClassifier:
    """
    CNN scikit-learn-like with softmax.

    Author:
        Caio Martins
    """

    def __init__(self, hidden_layers_sizes = None,
                 convpool_layers_sizes = None, activation = 'sigmoid', 
                 learning_rate=1e-3, num_epochs=10000, 
                 verbose=True, print_epoch=1000, eps=1e-5,
                 gd_type='full', batch_number = 10, mu = 0.9,
                 decay=0.99, dropout_rates = None, filtersz=None):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.print_epoch = print_epoch
        self.eps = eps
        self.gd_type = gd_type
        self.batch_number = batch_number
        self.mu = mu
        self.decay = decay
        self.activation = activation
        self.convpool_layers_sizes = convpool_layers_sizes
        self.filtersz = (3,3)
        if filtersz != None:
            self.filtersz = filtersz
        self.hidden_layers_sizes = []
        if hidden_layers_sizes != None:
            self.hidden_layers_sizes = hidden_layers_sizes
        self.dropout_rates = [1 for i in range(len(self.hidden_layers_sizes)+1)]
        if dropout_rates != None:
            self.dropout_rates = dropout_rates

    def score(self, y, y_pred):
        accuracy = (y == y_pred).sum()/y.shape[0]
        return accuracy

    def _get_dims(self, X, y):
        self.K = np.unique(y).size
        self.N, self.Height, self.Width, self.C = X.shape


    def fit(self, X, y=None, X_test=None, y_test=None):
        tf.reset_default_graph() #important otherwise get errors - boilerplate

        self._get_dims(X, y)

        T = np.array([[0 if y[j] != i else 1 for i in range(self.K)] for j in range(self.N)])
        T_test = np.array([[0 if y_test[j] != i else 1 for i in range(self.K)] for j in range(y_test.shape[0])])
        
        inputs = tf.placeholder(tf.float32, shape=[None, self.Height, self.Width, self.C], name = 'inputs')
        labels = tf.placeholder(tf.float32, shape=[None, self.K], name = 'labels')

        #ConvPool layers
        self.convpool_layers_sizes = [self.C] + self.convpool_layers_sizes.copy()
        Z = inputs
        for ind, _ in enumerate(self.convpool_layers_sizes[:-1]):
            convlayer = ConvPoolLayer(self.filtersz[0],
                                      self.filtersz[1],            
                                      self.convpool_layers_sizes[ind],
                                      self.convpool_layers_sizes[ind+1],
                                      ind,
                                      activation=self.activation)
            Z = convlayer.forward(Z)

        #flattens the features so fully-connected layer can accept it
        Z_shape = Z.get_shape().as_list()
        D = np.prod(Z_shape[1:]) # dense layers input dimension
        Z_flattened = tf.reshape(Z, shape=[-1, D])

        #Dense layers (Fully connected)
        self.hidden_layers_sizes = [D] + self.hidden_layers_sizes.copy() + [self.K]
        self.Zs = [Z_flattened]
        self.Zs_test = [Z_flattened]
        self.hidden_layers = []
        for ind, _ in enumerate(self.hidden_layers_sizes[:-1]):
            hl = HiddenLayer(self.hidden_layers_sizes[ind],
                             self.hidden_layers_sizes[ind+1],
                             ind,
                             activation=self.activation,
                             dropout_rate=self.dropout_rates[ind])
            self.hidden_layers.append(hl)
            self.Zs.append(hl.forward(self.Zs[ind]))
            last_logit = hl.logit
            self.Zs_test.append(hl.forward(self.Zs_test[ind],test=True))
        
        cost = tf.reduce_sum(tf.nn.\
                  softmax_cross_entropy_with_logits_v2(logits=last_logit,
                                                       labels=labels))
        
        train_op = tf.train.RMSPropOptimizer(self.learning_rate,
                                             decay=self.decay,
                                             momentum=self.mu,
                                             epsilon=self.eps).\
                                                 minimize(cost)
        self.pred_op = tf.arg_max(self.Zs_test[-1], 1)    
        
        self.costs = []
        self.costs_test = []
        self.clf_rates = []
        self.clf_rates_test = []
        
        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            if self.gd_type == 'full':

                for epoch in range(self.num_epochs):
                    
                    sess.run(train_op, feed_dict={'inputs:0':X, 'labels:0':T})
                    cost_val = sess.run(cost, feed_dict={'inputs:0':X, 'labels:0':T})
                    y_pred = sess.run(self.pred_op, feed_dict={'inputs:0':X})

                    cost_val_test = sess.run(cost, feed_dict={'inputs:0':X_test, 'labels:0':T_test})
                    y_pred_test = sess.run(self.pred_op, feed_dict={'inputs:0':X_test})

                    self.costs.append(cost_val/self.N)
                    self.costs_test.append(cost_val_test/y_test.shape[0])

                    self.clf_rates.append(self.score(y, y_pred))
                    self.clf_rates_test.append(self.score(y_test, y_pred_test))
                    
                    if epoch % self.print_epoch == 0:
                        if self.verbose:
                            print("epoch: %d" % epoch,
                                  "cost: %.3f" % self.costs[-1],
                                  "accuracy: %.3f" % self.clf_rates[-1],
                                  "cost_test: %.3f" % self.costs_test[-1],
                                  "accuracy_test: %.3f" % self.clf_rates_test[-1])

                saver.save(sess, './model_ann/ann',global_step=self.num_epochs)

            elif self.gd_type == 'batch':
                
                batch_size = int(self.N/self.batch_number*1.0)

                for epoch in range(self.num_epochs):
                    
                    X, y = shuffle(X,y)
                    T = np.array([[0 if y[j] != i else 1 for i in range(self.K)] for j in range(self.N)])

                    for l in range(self.batch_number):

                        x = X[batch_size*l:(batch_size)*(l+1)]
                        t = T[batch_size*l:(batch_size)*(l+1)]

                        sess.run(train_op, feed_dict={'inputs:0':x, 'labels:0':t})
                        
                        cost_val = sess.run(cost, feed_dict={'inputs:0':X, 'labels:0':T})
                        y_pred = sess.run(self.pred_op, feed_dict={'inputs:0':X})
                        cost_val_test = sess.run(cost, feed_dict={'inputs:0':X_test, 'labels:0':T_test})
                        y_pred_test = sess.run(self.pred_op, feed_dict={'inputs:0':X_test})

                        self.costs.append(cost_val/self.N)
                        self.costs_test.append(cost_val_test/y_test.shape[0])

                        self.clf_rates.append(self.score(y, y_pred))
                        self.clf_rates_test.append(self.score(y_test, y_pred_test))
                       
                        if (epoch*self.batch_number+l) % self.print_epoch == 0:
                            if self.verbose:
                                print("epoch: %d" % epoch,
                                      "cost: %.3f" % self.costs[-1],
                                      "accuracy: %.3f" % self.clf_rates[-1],
                                      "cost_test: %.3f" % self.costs_test[-1],
                                      "accuracy_test: %.3f" % self.clf_rates_test[-1])
                    
                saver.save(sess, './model_ann/ann',global_step=self.num_epochs*self.batch_number)

    def predict(self, X):
        with tf.Session() as session:
            if self.gd_type == 'full':
                new_saver = tf.train.import_meta_graph('./model_ann/ann-{}.meta'.format(self.num_epochs))
            else:
                new_saver = tf.train.import_meta_graph('./model_ann/ann-{}.meta'.format(self.num_epochs*self.batch_number))
            new_saver.restore(session, tf.train.latest_checkpoint('./model_ann'))
            y_pred = session.run(self.pred_op, feed_dict={'inputs:0':X})
        return y_pred

        