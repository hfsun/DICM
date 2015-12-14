import math
import numpy as np
from numpy import *
from sklearn.cluster import KMeans
npa = np.array


def PrincipalNumber(w):
    w_sum = w.sum()
    s=0.0
    for i in xrange(len(w)):
        s += w[i]
        if (s/w_sum)>0.33:
            return i

def DiscriminativeWeight(u,w):
    TT = mat(array(u)*array(u))
    AA = mat(array(w)*array(w)).T
    p = TT.dot(AA)
    return p

def Represent(U,p,x,cover_init=1):
    sigma = 4000
    for i in xrange(len(p)):
        if p[i]>sigma:
            p[i] = 0
        else:
            p[i] = 1 - p[i]/sigma
    weight_matrix = np.diag(array(p)[:,0])
    a = weight_matrix.dot(x)
    ivector = array((U.T).dot(a)).T
    for i in xrange(len(ivector)):
        total = 0.0
        for j in xrange(len(ivector[i])):
            total += ivector[i][j]*ivector[i][j]
        total = math.sqrt(total)
        for j in xrange(len(ivector[i])):
            ivector[i][j] = ivector[i][j]/total
    return ivector

def KmeansProcess(ivec,n):
    knn = KMeans(init='k-means++', n_clusters=n, n_init=25)
    knn.fit_predict(ivec)
    return knn

def DICM(dataset,n_clusters,cover_init=1):
    x = np.load(dataset) #read the input term-document matrix
    U,w,V = np.linalg.svd(x,full_matrices=True)
    principal_number = PrincipalNumber(w)
    w = w[:principal_number]
    U = U[:,:principal_number]
    p = DiscriminativeWeight(U,w)

    ivec = Represent(U,p,x,cover_init=1)
    result = KmeansProcess(ivec,n_clusters)
    print result.labels_