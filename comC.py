# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 10:54:25 2016

@author: aa
"""
import scipy

def computeCost(X,y,theta):
    m=len(y)
    J=1.0/(2*m)*sum(scipy.power(scipy.dot(X,theta)-y,2).real)
    return J