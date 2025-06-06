"""
Модуль train
------------
Демонстрационный «скелет» функции тренировки.
Здесь модель не обучается по-настоящему,
просто эмулируем несколько шагов.
"""

import random
from typing import List, Tuple


def train_model(epochs: int, data: List[int]) -> Tuple[float, float]:
    """
    Примитивная функция «тренировки»: эмулируем обучение,
    возвращая фиктивные метрики.

    Args:
        epochs (int): Число «эпох» (итераций).
        data (List[int]): Входные данные (например, списки чисел).

    Returns:
        Tuple[float, float]: Кортеж (loss, accuracy). Поскольку это заглушка,
        возвращаем случайные «метрики» в допустимых диапазонах.
    """
    if epochs <= 0:
        raise ValueError("epochs должно быть положительным")
    if not data:
        raise ValueError("data не может быть пустым списком")

    # Эмуляция: чем больше epochs, тем ниже «loss» и выше «accuracy».
    base_loss = 1.0 / epochs
    base_acc = min(1.0, 0.5 + 0.05 * epochs)

    # Добавим небольшие случайные «шумы»
    loss = base_loss + random.uniform(-0.01, 0.01)
    accuracy = base_acc + random.uniform(-0.02, 0.02)

    # Гарантируем, что loss ≥ 0, accuracy ∈ [0, 1]
    loss = max(loss, 0.0)
    accuracy = max(min(accuracy, 1.0), 0.0)
    return loss, accuracy
