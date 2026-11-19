import numpy as np


def numerical_gradient(values, objective, eps=1e-5):
    """Центральная разность для скалярной функции на маленьком массиве float64."""
    gradient = np.empty_like(values)
    for index in np.ndindex(values.shape):
        original = values[index]
        values[index] = original + eps
        plus = objective()
        values[index] = original - eps
        minus = objective()
        values[index] = original
        gradient[index] = (plus - minus) / (2 * eps)
    return gradient
