import unittest

import numpy as np

from mnist_numpy.loss import softmax_cross_entropy
from mnist_numpy.model import MLP
from tests.helpers import numerical_gradient


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = MLP(3, 2, 3, np.random.default_rng(21))
        self.model.hidden.weights = np.array([[1.0, -1.0], [2.0, 1.0], [1.0, 2.0]])
        self.model.hidden.bias = np.array([-1.25, 0.5])
        self.model.output.weights = np.array([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]])
        self.model.output.bias = np.array([0.25, -0.5, 1.0])
        self.x = np.array([[1.0, 2.0, -1.0], [-1.0, 0.0, 2.0]])
        self.labels = np.array([0, 2])

    def test_forward_returns_logits(self):
        logits = self.model.forward(self.x)
        np.testing.assert_allclose(logits, [[3.0, -0.5, -1.75], [0.25, 10.5, 6.5]])

    def test_backward_and_step_for_both_layers(self):
        _, gradient = softmax_cross_entropy(self.model.forward(self.x), self.labels)
        d_x = self.model.backward(gradient)
        layers = (self.model.hidden, self.model.output)
        params = [(layer.weights, layer.d_weights) for layer in layers]
        params += [(layer.bias, layer.d_bias) for layer in layers]

        def objective():
            return softmax_cross_entropy(self.model.forward(self.x), self.labels)[0]

        np.testing.assert_allclose(d_x, numerical_gradient(self.x, objective), rtol=1e-5, atol=1e-7)
        for values, analytical in params:
            numerical = numerical_gradient(values, objective)
            np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-7)

        rate = 0.1
        expected = [values - rate * gradient for values, gradient in params]
        self.model.step(rate)
        actual = [layer.weights for layer in layers] + [layer.bias for layer in layers]
        for result, target in zip(actual, expected):
            np.testing.assert_allclose(result, target)
