import numpy as np

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigDerivative(x):
    return sig(x) * (1 - sig(x))

def tanh(x):
    return np.tanh(x)

def tanhDerivative(x):
    return 1 - np.tanh(x) ** 2

