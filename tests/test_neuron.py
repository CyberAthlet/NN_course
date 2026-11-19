import unittest

import numpy as np

from mnist_numpy.layers import Neuron, relu


class TestNeuron(unittest.TestCase):
    def test_relu_is_elementwise(self):
        np.testing.assert_array_equal(relu(np.array([-2.0, 0.0, 3.0])), [0.0, 0.0, 3.0])

    def test_example_from_lecture(self):
        neuron = Neuron(np.array([1.2]), bias=-0.5)
        self.assertAlmostEqual(float(neuron.forward(np.array([-1.0]))), 0.0)
        self.assertAlmostEqual(float(neuron.forward(np.array([1.0]))), 0.7)

    def test_multiple_inputs_and_bias(self):
        neuron = Neuron(np.array([0.5, -1.0, 2.0]), bias=0.25)
        actual = neuron.forward(np.array([2.0, -1.0, 0.5]))
        self.assertEqual(np.shape(actual), ())
        self.assertAlmostEqual(float(actual), 3.25)
