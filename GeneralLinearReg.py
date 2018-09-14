#GeneralLinearRegTechX

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

def cost(x,y,w,b):
    p=np.dot(x,w)+b
    return float(np.sum((y-p)**2))

def linearR(x,y,threshold=1,lr=1e-4):
    k,m=x.shape
    n=y.shape[1]
    w=np.random.rand(m,n)
    b=np.random.rand(n)
    prevCost=0
    while cost(x,y,w,b)>threshold and cost(x,y,w,b)!=prevCost:
        prevCost=cost(x,y,w,b)
        print(prevCost)
        prevW=deepcopy(w)
        prevB=deepcopy(b)
        for i in range(0,k):
            for j in range(0,n):
                J=prevB[j]
                for l in range(0,m):
                    J+=x[i][l]*prevW[l][j]
                for l in range(0,m):
                    w[l][j]-=(2*(J-y[i][j])*x[i][l])*lr
                b[j]-=(2*(J-y[i][j]))*lr
    return w,b
