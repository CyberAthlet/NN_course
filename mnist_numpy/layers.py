"""Этапы 1–2. От одного нейрона к полносвязному слою."""

import numpy as np


def relu(z):
    """Применяет ReLU поэлементно к скаляру или массиву."""
    raise NotImplementedError("Этап 1: relu")


class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.asarray(weights)
        self.bias = bias

    def forward(self, x):
        """Вход x формы (D,), выход — скаляр после ReLU."""
        raise NotImplementedError("Этап 1: Neuron.forward")


class Dense:
    def __init__(self, n_inputs, n_outputs, rng):
        """Создаёт weights (D, H) и bias (H,); rng — np.random.Generator."""
        raise NotImplementedError("Этап 2: Dense.__init__")

    def forward(self, x):
        """Преобразует (B, D) в (B, H) и запоминает x для backward."""
        raise NotImplementedError("Этап 2: Dense.forward")

    def backward(self, d_z):
        """По d_z (B, H) вычисляет self.d_weights, self.d_bias и возвращает d_x (B, D)."""
        raise NotImplementedError("Этап 2: Dense.backward")


class ReLU:
    def forward(self, z):
        """Применяет relu и сохраняет информацию, нужную backward."""
        raise NotImplementedError("Этап 2: ReLU.forward")

    def backward(self, d_a):
        """Возвращает d_z той же формы, что и d_a. Производная в нуле равна 0."""
        raise NotImplementedError("Этап 2: ReLU.backward")
