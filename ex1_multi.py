# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 15:30:45 2016

@author: aa
"""
import numpy as np
import FN
import GDM
import NE

print 'Loading data ...'

data=np.loadtxt('ex1data2.txt',delimiter=',')
X=data[:,0:2]
y=data[:,2]
m=len(y)

print 'Normalizing Features ...'
tmp=FN.featureNormalize(X)
X=tmp[0]
mu=tmp[1]
sigma=tmp[2]

X=np.hstack((np.ones((m,1)), X))

print 'Running gradient descent ...'

alpha = 0.1
num_iters=1000

theta=np.zeros((3,1))

tmp=GDM.gradientDescentMulti(X,y,theta,alpha,num_iters)
theta=tmp[0]
J_history=tmp[1]



print 'Estimate Start ....'
price=np.dot([1,1650,3],theta)
print 'Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ',price

print 'Solving with normal equations...'

data=np.loadtxt('ex1data2.txt',delimiter=',')
X=data[:,0:2]
y=data[:,2]
m=len(y)
X=np.hstack((np.ones((m,1)), X))

theta=NE.normalEqn(X,y)

print 'Theta computed from the normal equations: '
print theta

price=np.dot([1,1650,3],theta)

print 'Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ',price
