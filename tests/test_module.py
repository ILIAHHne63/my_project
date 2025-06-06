import pytest

from src.module import gcd, is_prime, lcm


@pytest.mark.parametrize(
    "n, expected",
    [
        (2, True),
        (3, True),
        (4, False),
        (17, True),
        (18, False),
    ],
)
def test_is_prime_valid(n, expected):
    assert is_prime(n) == expected


def test_is_prime_invalid():
    with pytest.raises(ValueError):
        is_prime(1)  # n < 2 вызывает ошибку


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (0, 0, 0),
        (0, 5, 5),
        (6, 3, 3),
        (54, 24, 6),
        (-8, 12, 4),
    ],
)
def test_gcd(a, b, expected):
    assert gcd(a, b) == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (0, 5, 0),
        (7, 3, 21),
        (6, 4, 12),
        (-4, 6, 12),
    ],
)
def test_lcm(a, b, expected):
    assert lcm(a, b) == expected
