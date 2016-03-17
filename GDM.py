# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 15:50:20 2016

@author: aa
"""
import numpy as np
import CCM

def gradientDescentMulti(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=np.zeros((num_iters,1))
    
    for i in range(num_iters):
        y.shape=(y.shape[0],1)
        tmp=np.sum((np.dot(X,theta)-y)*X,axis=0)    
        tmp.shape=(tmp.shape[0],1)
        theta=theta - alpha * (1.0/m) * tmp
        J_history=CCM.computeCostMulti(X,y,theta)
    return (theta, J_history)