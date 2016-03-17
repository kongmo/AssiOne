# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 15:59:45 2016

@author: aa
"""

import numpy as np

def normalEqn(X,y):
    y.shape=(y.shape[0],1)
    inve=np.linalg.pinv(np.dot(np.transpose(X),X))
    theta=np.dot(np.dot(inve,np.transpose(X)),y)
    return theta
    