
"""
Created on Sun Feb  5 13:29:51 2017

@author: ps847
"""
# Better initial guess, better result. But how to guess lambda

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io 
from scipy.optimize import fsolve

sig = 0.05 # from SNR
k = 2.0*np.pi
nint=10

np.random.seed(0)

v = scipy.io.loadmat('v.mat')
v = v['v']
v = [val for sublist in v for val in sublist]
dprime = np.asarray(v)
n = len(dprime) # Data length

g = scipy.io.loadmat('g.mat') # access array as g0['g0']
g = g['g']
sizeg = g.shape
m1 = int(np.sqrt(sizeg[1]))
m = int(m1*m1)

def calm(lam):
    model=np.zeros(m)
    sum=0.0
    for j in range(m):        
        arg = np.dot(lam.T,g[:,j])
        model[j]=np.exp(-arg)
        sum+=model[j]
    # Sum of all elements of model = 1
    model/=sum
    return(gamma*model) 

def fun(lam):
    clam = np.sqrt(sig**2**np.dot(lam.T,lam)/(4.0*float(n)))
    e=-lam*sig**2/np.max(2.0*clam,1.0e-9)
    fun = dprime + e - np.dot(g,calm(lam))
    return fun

def jacob(lam):
    clam = np.sqrt(sig**2**np.dot(lam.T,lam)/(4.0*float(n)))
    jacob = -np.identity(n)*sig**2/(2.0*clam)+ \
            np.dot(lam,lam.T)*sig**4/(8.0*float(n)*clam**3)    
    temp=np.zeros((m,n))
    model = calm(lam)
    for j in range(m):
        for k in range(n):
            temp[j,k]=-g[k,j]*model[j]+np.dot(g,model)[k]*model[j]/gamma
    jacob -= np.dot(g,temp)
    return(jacob)           
    
for ints in range(nint):

# maximum entropy solution
    gamma =  99 # m_tot
    # uniform initial model guess
    # chk with true model 
    temp = np.ones(m)/float(m) # Division is for normalization
    temp = temp * gamma # Added this line additional CHK
    err = np.dot(g,temp)-dprime
    tlam = 1.0e-9 # was 1e-2, 1e-5
    lam = -err*2*tlam/sig**2
# find roots of equations
    x = fsolve(fun,lam,fprime=jacob)
#    x = root(fun,lam,method = 'lm')
    model = calm(x)
    model = model.reshape((m1,m1))
    #print model
    print (ints)
    
plt.title('MaxENT inverse')
im = plt.imshow(model, interpolation='bilinear', origin='lower',
                cmap=cm.gray)
#levels = np.arange(-1.2, 1.6, 0.2)
CS = plt.contour(model)

plt.show()

            


