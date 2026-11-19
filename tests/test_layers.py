import unittest

import numpy as np

from mnist_numpy.layers import Dense, Neuron, ReLU
from tests.helpers import numerical_gradient


class TestLayers(unittest.TestCase):
    def setUp(self):
        self.layer = Dense(3, 2, np.random.default_rng(11))

    def test_parameter_shapes_and_reproducibility(self):
        weights, bias = self.layer.weights, self.layer.bias
        self.assertEqual(weights.shape, (3, 2))
        self.assertEqual(bias.shape, (2,))
        self.assertTrue(np.isfinite(weights).all() and np.isfinite(bias).all())
        self.assertFalse(np.array_equal(weights[:, 0], weights[:, 1]))
        same_layer = Dense(3, 2, np.random.default_rng(11))
        np.testing.assert_array_equal(weights, same_layer.weights)
        np.testing.assert_array_equal(bias, same_layer.bias)

    def test_dense_relu_agrees_with_individual_neurons(self):
        self.layer.weights = np.array([[1.0, -1.0], [2.0, 1.0], [1.0, 2.0]])
        self.layer.bias = np.array([-1.0, 0.5])
        x = np.array([[1.0, 2.0, -1.0], [-1.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
        neurons = [Neuron(self.layer.weights[:, j], self.layer.bias[j]) for j in range(2)]
        expected = np.array([[neuron.forward(row) for neuron in neurons] for row in x])
        actual = ReLU().forward(self.layer.forward(x))
        self.assertEqual(actual.shape, (3, 2))
        np.testing.assert_allclose(actual, expected)

    def test_relu_gradient(self):
        layer = ReLU()
        layer.forward(np.array([[-1.0, 0.0, 1.0], [3.0, -1.0, 0.0]]))
        upstream = np.array([[2.0, 3.0, 4.0], [-2.0, -3.0, -4.0]])
        np.testing.assert_array_equal(layer.backward(upstream), [[0.0, 0.0, 4.0], [-2.0, 0.0, 0.0]])

    def test_dense_gradients_against_finite_differences(self):
        self.layer.weights = np.array([[0.1, 0.2], [0.3, -0.2], [0.5, 0.1]])
        self.layer.bias = np.array([-0.2, 0.5])
        x = np.array([[1.0, -2.0, 0.5], [0.3, 0.2, -1.0]])
        upstream = np.array([[0.4, -0.1], [0.2, 0.6]])
        self.layer.forward(x)
        d_x = self.layer.backward(upstream)

        def objective():
            return np.sum(self.layer.forward(x) * upstream)

        for name, analytical, inputs in (
            ("weights", self.layer.d_weights, self.layer.weights),
            ("bias", self.layer.d_bias, self.layer.bias),
            ("x", d_x, x),
        ):
            with self.subTest(gradient=name):
                numerical = numerical_gradient(inputs, objective)
                np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-7)
