#coding=utf-8
import numpy as np
import random
import math

class LogisticRegression(object):

    def __init__(self):
        self.w = None


    def loss(self, X_batch, y_batch):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """

        #########################################################################
        # TODO:                                                                 #
        # calculate the loss and the derivative                                 #
        #########################################################################

        loss = 0
        lamda = 1
        N = len(X_batch)
        z = np.dot(X_batch,self.w)#wx+b
        loss = (-np.dot(y_batch.T,(np.log(1.0 / (1.0 + np.exp(-z))))) - np.dot((1.0 - y_batch.T),(np.log(1.0 - 1.0 / (1.0 + np.exp(-z))))) + lamda * (np.dot(self.w[1:].T,self.w[1:]))) / N
        grad = np.zeros_like(self.w)#定义梯度下降矩阵的维度和参数w的维度一样
        grad = (np.dot(X_batch.T,(1.0 / (1.0 + np.exp(-z)) - y_batch)) + 2 * lamda * self.w) / N#(h(xi) - yi)xi / N

        return loss,grad

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=True):

        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape

        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)

        loss_history = []

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            batch = np.random.choice(len(X),batch_size)
            X_batch = X[batch]
            y_batch = y[batch]

            # print(X.shape)
            # print(y.shape)
            # print(X_batch.shape)
            # print(y_batch.shape)

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            #pass
            self.w = self.w - learning_rate * grad
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################

        y_pred = 1.0 / (1.0 + np.exp(-(np.dot(X,self.w))))
        for i in range(len(X)):#遍历每一个样本的预测值
            if y_pred[i] >= 0.50:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred



    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose = True):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        """

        m,n = X.shape
        self.w1 = np.zeros((n,10))#(784,10)
        for i in range(10):#调用10次二分类器
            y_train_label = []#存放标签
            for label in y:#遍历y_train
                if label == i:
                    y_train_label.append(1)#预测值和真实值相等，y_train_label + 1
                else:
                    y_train_label.append(0)#预测值和真实值相等，y_train_label + 0
            y_train_label = np.array(y_train_label)
            self.w = None
            print('i= %d' % i)
            self.train(X,y_train_label,learning_rate,num_iters,batch_size)
            self.w1[:,i] = self.w#存放10个数字的参数


    def one_vs_all_predict(self,X):
        labels = (1.0 / (1.0 + np.exp(-(np.dot(X,self.w1)))))
        y_pred = np.argmax(labels,axis=1)#return max value index
        return y_pred
