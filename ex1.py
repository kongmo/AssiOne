# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 09:04:47 2016

@author: Li Min
"""
import warmUE
import scipy
import comC
import GD
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

B=warmUE.warmUpExercise()

data=scipy.loadtxt('ex1data1.txt',delimiter=',')
X=data[:,0]
X.shape=(X.shape[0],1)
X=X
y=data[:,1]
y.shape=(y.shape[0],1)
y=y
m=len(y)

plt.plot(X,y,'rx')
plt.xlabel('Population of City in 10,000s ')
plt.ylabel('Profit in $ 10,000s ')

print 'Program paused. Press enter to continue.\n'
print 'Running Gradient Descent ... \n'

X=scipy.hstack((scipy.ones((m,1)),X))
theta=scipy.zeros((2,1))

iterations=500
alpha=0.01

J_hist=comC.computeCost(X,y,theta)
print 'The first cost is: '
print J_hist

theta=GD.gradientDescent(X,y,theta,alpha,iterations)[0]
#J_history=GD.gradientDescent(X,y,theta,alpha,iterations)[1]
print 'Theta found by gradient descent: '
print theta[0],theta[1]

tmp=X[:,1]
tmp.shape=(tmp.shape[0],1)
plt.plot(tmp,scipy.dot(X,theta),'-')
plt.legend(['Training data.','Linear regression'],loc=4)

print '  '
print 'Predict values for population sizes of 35,000 and 70,000 '
predict1=scipy.dot([1, 3.5],theta)
print 'For population = 35,000, we predict a profit of {0}'.format(predict1*10000)
predict2=scipy.dot([1, 7],theta)
print 'For population = 70,000, we predict a profit of {0}'.format(predict2*10000)

print 'Visualizing ....'

theta0_vals=np.linspace(-10,10,100)
theta1_vals=np.linspace(-1,4,100)
J_vals=np.zeros((len(theta0_vals),len(theta1_vals)))
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t=np.vstack((theta0_vals[i],theta1_vals[j]))
        J_vals[i,j]=comC.computeCost(X,y,t)

J_vals=np.transpose(J_vals)

X,Y=np.meshgrid(theta1_vals,theta0_vals,sparse=False)
fig=plt.figure()
ax=fig.gca(projection='3d')
ax.set_xlabel(r'$\theta_\mathrm{1}$')
ax.set_ylabel(r'$\theta_\mathrm{0}$')
ax.plot_surface(X,Y,J_vals,cmap='rainbow')
ax.view_init(None,30)
#ax.set_xticks([-1,0,1,2,3,4])
plt.show()

m=J_vals.argmin()
shape=J_vals.shape
row=m//shape[1]-1
col=m % shape[1]-1

plt.figure()
cs=plt.contour(theta0_vals,theta1_vals,J_vals,levels=[0,8,20,50,100,180,300,400])
plt.clabel(cs, inline=1, fontsize=10)
plt.xlabel(r'$\theta_\mathrm{0}$')
plt.ylabel(r'$\theta_\mathrm{1}$')
plt.title('Contour')
plt.text(theta0_vals[col],theta1_vals[row],'x',color='r')
#plt.plot(theta[0],theta[1])