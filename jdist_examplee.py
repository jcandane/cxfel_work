from jdist import jdist
import numpy as np
import time

A = np.random.rand( 1600, 22000).astype(np.float32)

START=time.time()
J = jdist(A,A)
END=time.time()
print(END-START) ##!!!!

from scipy.spatial.distance import cdist

C = cdist(A, A, 'euclidean')

print( np.allclose( J , C ) )

###### numpy example
B = np.arange( 100000 * 1000 ).reshape( 100000 , 1000 )

START=time.time()
J = jdist(B,B)
END=time.time()
print(END-START) ##!!!!