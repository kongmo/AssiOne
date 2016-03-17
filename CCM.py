# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 15:53:09 2016

@author: aa
"""

import numpy as np

def computeCostMulti(X,y,theta):
    m=len(y)
    J=1.0/(2*m)*sum((np.power(np.dot(X,theta)-y,2)).real)
    return J