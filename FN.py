# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 15:36:49 2016

@author: aa
"""
def featureNormalize(X):
    mu=X.mean()
    sigma=X.std()
    X_norm=(X-mu)/sigma
    return (X_norm,mu,sigma)
    