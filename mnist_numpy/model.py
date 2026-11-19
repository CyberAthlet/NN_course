"""Этап 4. Полносвязная сеть с одним скрытым слоем."""

from .layers import Dense, ReLU


class MLP:
    def __init__(self, n_inputs, hidden_size, n_classes, rng):
        """Собирает self.hidden (Dense), self.activation (ReLU) и self.output (Dense)."""
        raise NotImplementedError("Этап 4: MLP.__init__")

    def forward(self, x):
        """Принимает батч (B, n_inputs) и возвращает логиты (B, n_classes)."""
        raise NotImplementedError("Этап 4: MLP.forward")

    def backward(self, d_logits):
        """Передаёт градиент через слои в обратном порядке и возвращает d_x."""
        raise NotImplementedError("Этап 4: MLP.backward")

    def step(self, learning_rate):
        """Одним шагом градиентного спуска обновляет веса и смещения обоих слоёв."""
        raise NotImplementedError("Этап 4: MLP.step")
