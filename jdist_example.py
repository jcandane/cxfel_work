from jdist import jdist
import numpy as np
import time

A = np.random.rand( 1600, 22000).astype(np.float32)
#A = np.arange( 1600 * 22000 ).reshape( 1600, 22000 )
B = np.arange( 160000 * 22000 ).reshape( 160000, 22000 )

START=time.time()
J = jdist(A,A)
END=time.time()
print(END-START) ##!!!!

from scipy.spatial.distance import cdist

C = cdist(A, A, 'euclidean')

print( np.allclose( J , C ) )

######################################################

trials = np.logspace(2, 5.32 , num=30, base=10).astype(np.int64)

res=[]
for trial in range(len(trials)):

    print(trial)
    pertry=[]
    for j in range(3): ## 3 for each
        A = np.random.rand( trials[trial], 22000).astype(np.float32)
        START=time.time()
        jdist(A,A)
        END=time.time()
        pertry.append( END-START )
    res.append( np.mean( np.asarray(pertry) ) )

print("results on varying N")
print(trials)
print(res)

Ds = np.logspace(2, 6.32 , num=30, base=10).astype(np.int64)

Dres=[]
for D in range(len(Ds)):

    print(D)
    pertry=[]
    for j in range(3): ## 3 for each
        A = np.random.rand( 1000 , Ds[D] ).astype(np.float32)
        START=time.time()
        jdist(A,A)
        END=time.time()
        pertry.append( END-START )
    Dres.append( np.mean( np.asarray(pertry) ) )

print("results on varying D")
print(Ds)
print(Dres)
