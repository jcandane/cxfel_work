import os
os.system("wget https://raw.githubusercontent.com/jcandane/cxfel_work/main/jdist.py")  

from jdist import jdist
from scipy.sparse import coo_matrix
from scipy.signal import convolve2d

def sna(A, c):
    return convolve2d(A, np.eye(c), mode='valid')

def k_neighbor(D_IJ, k):

    ### find k-nearest-neighbors
    j     = np.argsort( D_IJ , axis=1 , kind="stable")
    Ds_IJ =    np.sort( D_IJ , axis=1 , kind="stable")

    ### get sparse data
    j    = j[:,:k].reshape(-1)
    i    = (np.arange( Ds_IJ.shape[0] )[:,None] * np.ones(k, dtype=j.dtype)[None,:]).reshape(-1)
    data = Ds_IJ[:,:k].reshape(-1)
    return coo_matrix((data, (i,j)), shape=D_IJ.shape)

def total(R_ix, c, k):
    D_ij = jdist( R_ix, R_ix )
    D_IJ = sna( D_ij, c )
    return k_neighbor(D_IJ, k)

total(R_ix, c=6, k=102)
