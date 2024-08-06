import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from functools import partial

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

import numpy


@jax.jit
def parallel_pair(A, B): ### 7/31/2024

    devices = mesh_utils.create_device_mesh((len(jax.devices()), 1), devices=jax.devices())
    mesh    = Mesh(devices, axis_names=('i', 'j'))

    @jax.jit
    def kernel(v_x, v_y, hps=jnp.array([])):
        return -2.*jnp.dot(v_x, v_y)
        #return jnp.linalg.norm(v_x - v_y)

    @partial(jax.vmap, in_axes=(None, 0))
    def matrix_row(v_x, R_ix):
        return kernel(v_x, R_ix) #+ v_x + R_ix ## add v_x[None,:] 

    @partial(shard_map, mesh=mesh, in_specs=(P('i'), P(None)), out_specs=P('i'))
    def batched_pairwise(A_batch, B):
        return jax.vmap(matrix_row, in_axes=(0, None))(A_batch, B)

    return batched_pairwise(A, B)

def sharded_padding(A, l):
    """
    A:jax.Array 
    l:int (sharding dimension)
    GET>
    B:jax.Array (such that B.shape[0] % l = 0)
    """

    correction = (l - (A.shape[0] % l)) + A.shape[0]
    B = jnp.zeros((correction, A.shape[1]), dtype=A.dtype)
    B = B.at[:A.shape[0],:].set(A)
    return B

######==========
def jdist(A, B, Dtype=jnp.float16):

    ### move numpy.arrays (on CPU) -> jax.Arrays (on CPU)
    R  = jax.device_put(A.astype(Dtype), jax.devices("cpu")[0])
    B  = jax.device_put(B.astype(Dtype), jax.devices("cpu")[0])
    
    mesh_    = Mesh(numpy.array(jax.devices()).reshape((len(jax.devices()), 1)), axis_names=('i', 'j'))

    def cpu_calculation():
        return jnp.sum(R ** 2, axis=1)[:, None] + jnp.sum(B ** 2, axis=1)[None, :]

    ### shard & replicate on 4 GPUs version "_" jax.Arrays
    R_ = jax.device_put(sharded_padding(R, len(jax.devices())), NamedSharding(mesh_, P('i', 'j'))) ### pad then place for both first dimensions....
    B_ = jax.device_put(B, NamedSharding(mesh_, P(None)))

    D_ij  = cpu_calculation().block_until_ready() ### numpy is too slow, do this operation in jax
    D_ij += jax.device_put( parallel_pair(R_, B_).block_until_ready(), jax.devices("cpu")[0])[:R.shape[0],:B.shape[0]]
    return numpy.asarray(D_ij)
