from jax import jit # for compiling functions for speedup
from jax.experimental import stax # neural network library
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax, LeakyRelu # neural network layers

def CIGALE_emulator():
    output_cols=['spire_250','spire_350','spire_500']
    net_init, net_apply = stax.serial(
        Dense(128), LeakyRelu,
        Dense(128), LeakyRelu,
        Dense(len(output_cols))
    )
    return net_init,net_apply