"""Этап 4. Обучение на мини-батчах."""

import numpy as np

from .loss import softmax_cross_entropy


def train_epoch(model, x, labels, batch_size, learning_rate, rng):
    """Перемешивает объекты и проводит один проход по обучающей выборке.

    Для каждого мини-батча считает логиты, потери и градиенты, затем
    обновляет модель. Возвращает среднюю потерю по всем объектам эпохи,
    вычисленную по потерям мини-батчей перед соответствующими обновлениями.
    """
    raise NotImplementedError("Этап 4: train_epoch")
