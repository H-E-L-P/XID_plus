import jax
import pickle
from xidplus.numpyro_fit.neuralnet_models import CIGALE_emulator, CIGALE_emulator_kasia
import numpy as np

@jax.partial(jax.jit, static_argnums=(2))
def sp_matmul(A, B, shape):
    """
    http://gcucurull.github.io/deep-learning/2020/06/03/jax-sparse-matrix-multiplication/
    Arguments:
        A: (N, M) sparse matrix represented as a tuple (indexes, values)
        B: (M,K) dense matrix
        shape: value of N
    Returns:
        (N, K) dense matrix
    """
    #assert B.ndim == 2
    indexes, values = A
    rows, cols = indexes
    in_ = B.take(cols, axis=-2)
    prod = in_*values[:, None]
    res = jax.ops.segment_sum(prod, rows, shape)
    return res

def load_emulator(filename):
    #read in net params
    x=np.load(filename, allow_pickle=True)
    net_init,net_apply=CIGALE_emulator()
    return {'net_init':net_init,'net_apply':net_apply,'params':x['arr_0'].tolist()}

def load_emulatorII(filename):
    #read in net params
    x=np.load(filename, allow_pickle=True)
    net_init,net_apply=CIGALE_emulator_kasia()
    return {'net_init':net_init,'net_apply':net_apply,'params':x['arr_0'].tolist()}

