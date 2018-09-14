#GeneralLogisticRegTechX

"""
    x.shape=k*m
    y.shape=k*n
    w.shape=m*n
    b.shape=n

    k: number of samples
    m & n: dimension corresponding to x & y
"""

import numpy as np
from copy import deepcopy

def sigmoid(x):
    return 1/(1+np.exp(-x))

def cost(x,y,w,b):
    p=sigmoid(np.dot(x,w)+b)
    return float(np.sum(-(y*np.log(p)+(1-y)*np.log(1-p))))
                 
def linearR(x,y,threshold=1e-1,lr=1e-5,maximum_repeat):
    k,m=x.shape
    n=y.shape[1]
    w=np.zeros((m,n))
    b=np.zeros(n)
    prevCost=0
    index=0
    while (cost(x,y,w,b)>threshold or abs(cost(x,y,w,b)-prevCost)>threshold) and index<maximum_repeat:
        prevCost=cost(x,y,w,b)
        index+=1
        if index%500==0: print(prevCost)
        prevW=deepcopy(w)
        prevB=deepcopy(b)
        for i in range(0,k):
            for j in range(0,n):
                J=prevB[j]
                for l in range(0,m):
                    J+=x[i][l]*prevW[l][j]
                for l in range(0,m):
                    w[l][j]-=x[i][l]*(sigmoid(J)-y[i][j])*lr
                b[j]-=(sigmoid(J)-y[i][j])*lr
    return w,b
