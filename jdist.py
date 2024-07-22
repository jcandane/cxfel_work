import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from functools import partial

####
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(4)
print(jax.devices("cpu"))

######################################################

devices = mesh_utils.create_device_mesh((len(jax.devices()), 1))
mesh    = Mesh(devices, axis_names=('i', 'j'))

@jax.jit
def kernel(v_x, v_y, hps=jnp.array([])):
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

#### mesh has to be defined before defining the entire function!!!
@jax.jit
def parallel_pair(A, B, kernel=kernel):

    devices = mesh_utils.create_device_mesh((len(jax.devices()), 1))
    mesh    = Mesh(devices, axis_names=('i', 'j'))

    @partial(jax.vmap, in_axes=(None, 0))
    def matrix_row(v_x, R_ix):
        return kernel(v_x, R_ix)

    @partial(shard_map, mesh=mesh, in_specs=(P('i'), P(None)), out_specs=P('i'))
    def batched_pairwise(A_batch, B):
        return jax.vmap(matrix_row, in_axes=(0, None))(A_batch, B)

    return batched_pairwise(A, B)







def jjdist(a, b): ### NOT needed if taken into accout by jdist : Callable[numpy]
    """
    GIVEN > a,b : jnp.ndarray[:,:]
    GET >  c : jnp.ndarray[:,:]
    """
    de    = jnp.ceil( jnp.asarray( a.shape ) / jnp.array( list(mesh.shape.values()) ) ).astype(int)
    large = jax.numpy.zeros( jnp.array( list(mesh.shape.values()) )  * de , dtype=a.dtype )

    return parallel_pair( embed_array(a, large) , b )[:a.shape[0],:]

import numpy

def jdist(A,B, par=16): ### now we have to determine the partition-size,
    ### est. max memory per device....
    ### perhaps a for-loop to try different partitions to see what works.... aim for par=2
    """
    GIVEN > A,B : np.ndarray[N,D] (N samples & D features)
            par : int (the number of seqential partitions)
    GET > distance_matrix : np.ndarray[:,:]
    """

    ### create jax Device-Mesh
    #devices = mesh_utils.create_device_mesh((4, 1))
    #mesh    = Mesh(devices, axis_names=('i', 'j'))

    C=[]
    for i in range(par): ## divide B into pieces (each which divide 4?)
        ## CPU -> GPU
        a = jnp.asarray( A )
        b = jnp.asarray( B[(B.shape[0]*(i))//par:(B.shape[0]*(i+1))//par,:] )
        
        ## carry back from GPU -> CPU
        C.append( numpy.asarray( jjdist(a, b) ) )
    return numpy.concatenate(C, axis=1)