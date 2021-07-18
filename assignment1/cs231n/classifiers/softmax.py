from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    classes = np.max(y) + 1
    
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        
        normalizer = np.sum(np.exp(scores))
        softmax_funct = np.exp(correct_class_score)/normalizer
        softmax_loss = -1* np.log(softmax_funct)
        loss += softmax_loss
        
        for j in range(classes):
            if j == y[i]:
                dW[:,j] += -X[i].T + softmax_funct*X[i].T
            else:
                dW[:,j] += np.exp(scores[j])/normalizer*X[i].T
    loss = loss / num_train
    loss += reg*np.sum(W*W)
    
    dW = dW/num_train
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    num_train = X.shape[0]
    normalizer = np.sum(np.exp(scores), axis = 1)
    normalizer = normalizer.reshape(-1,1)
    probs = np.exp(scores[np.arange(num_train),y]).reshape(-1,1)/normalizer
    loss = -1*np.sum(np.log(probs))/num_train
    loss += reg*np.sum(W*W)
    
    
    d_probs = np.zeros((num_train, W.shape[1]))
    d_probs = np.exp(scores)/normalizer
    d_probs[np.arange(num_train), y] -= 1
    dW = X.T.dot(d_probs)
    dW = dW/num_train
    dW += 2*reg*W


    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
