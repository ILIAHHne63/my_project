import pytest

from src.train import train_model


def test_train_model_valid():
    """
    Для небольшого числа эпох и простых данных проверяем,
    что возвращаемые значения — кортеж из двух чисел (float, float)
    и что loss в диапазоне [0, ∞), accuracy ∈ [0, 1].
    """
    data = [1, 2, 3, 4, 5]
    loss, acc = train_model(epochs=3, data=data)
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert loss >= 0
    assert 0.0 <= acc <= 1.0


def test_train_model_negative_epochs():
    with pytest.raises(ValueError):
        train_model(epochs=0, data=[1, 2, 3])  # epochs <= 0


def test_train_model_empty_data():
    with pytest.raises(ValueError):
        train_model(epochs=5, data=[])  # пустой список данных
