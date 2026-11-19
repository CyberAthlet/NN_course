"""Этап 3. Кросс-энтропия для многоклассовой классификации."""

import numpy as np


def softmax_cross_entropy(logits, labels):
    """Принимает логиты (B, C) и целые метки (B,) в диапазоне [0, C).

    Возвращает (loss, d_logits). loss — среднее по батчу, d_logits —
    градиент этого среднего по каждому элементу logits, форма (B, C).
    Вычисления должны оставаться устойчивыми для больших логитов.
    """
    raise NotImplementedError("Этап 3: softmax_cross_entropy")
