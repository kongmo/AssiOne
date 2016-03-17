# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 11:20:28 2016

@author: aa
"""
import scipy
import numpy as np
import comC


def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=scipy.zeros((num_iters,1))
    
    for i in range(num_iters):
        tmp=np.sum((np.dot(X,theta)-y)*X,axis=0)    
        tmp.shape=(tmp.shape[0],1)
        theta=theta - alpha * (1.0/m) * tmp
        J_history=comC.computeCost(X,y,theta)
    return (theta, J_history)

