import unittest

import numpy as np

from mnist_numpy.loss import softmax_cross_entropy
from tests.helpers import numerical_gradient


class TestLoss(unittest.TestCase):
    def test_uniform_logits(self):
        logits = np.zeros((2, 3))
        labels = np.array([0, 2])
        loss, gradient = softmax_cross_entropy(logits, labels)
        self.assertAlmostEqual(float(loss), float(np.log(3)))
        np.testing.assert_allclose(gradient, [[-1 / 3, 1 / 6, 1 / 6], [1 / 6, 1 / 6, -1 / 3]])
        np.testing.assert_array_equal(logits, np.zeros((2, 3)))

    def test_large_logits(self):
        logits = np.array([[1000.0, -1000.0], [1000.0, 1000.0]])
        loss, gradient = softmax_cross_entropy(logits, np.array([1, 0]))
        self.assertAlmostEqual(float(loss), float((2000 + np.log(2)) / 2))
        self.assertTrue(np.isfinite(gradient).all())

    def test_gradient_against_finite_differences(self):
        logits = np.array([[0.2, -0.8, 0.7], [1.2, 0.3, -1.0]])
        labels = np.array([0, 2])
        _, gradient = softmax_cross_entropy(logits, labels)
        numerical = numerical_gradient(logits, lambda: softmax_cross_entropy(logits, labels)[0])
        np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-7)
