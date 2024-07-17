from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

######################################################

devices = mesh_utils.create_device_mesh((4, 1))
mesh    = Mesh(devices, axis_names=('i', 'j'))

@jax.jit
def kernel(v_x, v_y):
    #return jnp.dot(v_x, v_y)
    return jnp.linalg.norm(v_x - v_y)

@jax.jit
def embed_array(small, big):
    """ embed a small array into a bigger array
    """
    start  = tuple( 0          for i in range( small.ndim )       )
    end    = tuple( s + d      for s,d in zip(start, small.shape) )
    slices = tuple( slice(s,e) for s,e in zip(start, end)         )

    return big.at[slices].set(small)

@jax.jit
def parallel_pair(A, B, kernel=kernel): ## !! alreaady chose A as the larger one

    #A_split = pmap_dist_prep(A)
    ### place A & B on the GPU with the right shapes

    ###### define pairwise operation
    @jax.jit
    def pairwise(R_ix, R_jx, kernel=kernel):
        """ compute the entire covariance-matrix
        GET > Σ_ij
        """

        @jax.jit
        def matrix_row(v_x, R_ix):
            """ compute 1 row of the covariance-matrix
            covariance_vector(v_x, R_ix) = Σ_i
            """
            return jax.vmap(kernel, in_axes=(None, 0))(v_x, R_ix)

        @jax.jit
        def update_matrix(carry, i):
            """
            Given the entire covariance-matrix, update the "i"th row
            with the covariance_vector function
            """
            R_ij, R_ix, R_jx = carry
            R_ij = R_ij.at[i].set(matrix_row(R_ix[i], R_jx))
            return (R_ij, R_ix, R_jx), None

        N    = R_ix.shape[0]
        M    = R_jx.shape[0]
        Σ_ij = jax.numpy.zeros((N, M)) ### define empty covaraince-matrix
        (Σ_ij, _, __), _ = jax.lax.scan(update_matrix, (Σ_ij, R_ix, R_jx), jax.numpy.arange(N))
        return Σ_ij
    ######

    @partial(shard_map, mesh=mesh, in_specs=(P('i'), P(None)), out_specs=P('i'))
    def batched_pairwise(A_batch, B):
        return pairwise(A_batch, B)

    C_split = batched_pairwise(A, B)

    return C_split

def jjdist(a, b):
    """
    GIVEN > a,b : jnp.ndarray[:,:]
    GET >  c : jnp.ndarray[:,:]
    """
    de    = jnp.ceil( jnp.asarray( a.shape ) / jnp.array( list(mesh.shape.values()) ) ).astype(int)
    large = jax.numpy.zeros( jnp.array( list(mesh.shape.values()) )  * de , dtype=a.dtype )
    return parallel_pair( embed_array(a, large) , b)[:a.shape[0],:]

import numpy

def jdist(A,B, par=16):
    """
    GIVEN > A,B : np.ndarray[N,D] (N samples & D features)
            par : int (the number of seqential partitions)
    GET > distance_matrix : np.ndarray[:,:]
    """

    C=[]
    for i in range(par): ## divide A into quarters
        a = jnp.asarray( A )
        b = jnp.asarray( B[(B.shape[0]*(i))//par:(B.shape[0]*(i+1))//par,:] )
        C.append( numpy.asarray( jjdist(a, b) ) )
    return numpy.concatenate(C, axis=1)