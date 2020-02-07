from __future__ import division
import numpy as np

EPS = 1e-11

class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        return 1 / (2 * input.shape[0]) * np.sum((input - target) ** 2)

    def backward(self, input, target):
        '''Your codes here''' 
        return (input - target) / input.shape[0]


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here''' 
        exponents = np.exp(input)
        softmax = exponents / exponents.sum(axis=1)[:, np.newaxis ]
        return - 1 * (target * np.log(softmax)).sum(axis=1).sum() / input.shape[0]

    def backward(self, input, target):
        '''Your codes here''' 
        exponents = np.exp(input)
        softmax =  exponents / exponents.sum(axis=1)[:, np.newaxis ]
        return 1 * (softmax - target) / input.shape[0] 
