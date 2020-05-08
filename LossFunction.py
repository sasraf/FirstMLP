import numpy as np

def meanSquaredError(actual, expected):
    """Mean Squared Error - mean of (actual - expected)^2"""
    return np.mean(np.power(actual - expected, 2))

def meanSquaredErrorDerivative(actual, expected):
    return 2 * (expected - actual) / actual.size
