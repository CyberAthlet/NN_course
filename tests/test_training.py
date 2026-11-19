import unittest

import numpy as np

from mnist_numpy.loss import softmax_cross_entropy
from mnist_numpy.model import MLP
from mnist_numpy.training import train_epoch


class RecordingModel:
    def __init__(self):
        self.calls = []

    def forward(self, x):
        self.calls.append(("forward", x[:, 0].copy()))
        return np.column_stack((x[:, 0], np.zeros(len(x))))

    def backward(self, gradient):
        self.calls.append(("backward", gradient.copy()))

    def step(self, learning_rate):
        self.calls.append(("step", learning_rate))


class TestTraining(unittest.TestCase):
    def test_minibatch_order_labels_and_short_last_batch(self):
        x = np.arange(7, dtype=float).reshape(-1, 1)
        labels = np.arange(7) % 2
        model = RecordingModel()
        rate = 0.1
        actual = train_epoch(model, x, labels, 3, rate, np.random.default_rng(17))
        logits = np.column_stack((x[:, 0], np.zeros(len(x))))
        expected, _ = softmax_cross_entropy(logits, labels)
        self.assertAlmostEqual(float(actual), float(expected))
        self.assertEqual([name for name, _ in model.calls], ["forward", "backward", "step"] * 3)

        batches = [indices.astype(int) for _, indices in model.calls[::3]]
        self.assertEqual([len(batch) for batch in batches], [3, 3, 1])
        visited = np.concatenate(batches)
        np.testing.assert_array_equal(np.sort(visited), np.arange(len(x)))
        self.assertFalse(np.array_equal(visited, np.arange(len(x))))
        for indices, (_, gradient), (_, step_rate) in zip(batches, model.calls[1::3], model.calls[2::3]):
            _, expected_gradient = softmax_cross_entropy(logits[indices], labels[indices])
            np.testing.assert_allclose(gradient, expected_gradient, atol=1e-7)
            self.assertEqual(step_rate, rate)

    def test_network_learns_small_dataset(self):
        rng = np.random.default_rng(10)
        labels = np.repeat(np.arange(3), 16)
        centers = np.array([[-1.5, -1.0], [1.5, -1.0], [0.0, 1.5]])
        x = centers[labels] + rng.normal(scale=0.15, size=(len(labels), 2))
        model = MLP(2, 12, 3, rng)
        initial_loss, _ = softmax_cross_entropy(model.forward(x), labels)

        for _ in range(80):
            train_epoch(model, x, labels, 13, 0.1, rng)

        logits = model.forward(x)
        final_loss, _ = softmax_cross_entropy(logits, labels)
        self.assertLess(final_loss, 0.3 * initial_loss)
        self.assertGreaterEqual(np.mean(np.argmax(logits, axis=1) == labels), 0.95)
