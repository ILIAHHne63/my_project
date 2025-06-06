"""
Модуль module
-------------
Простейшие функции для арифметики и проверки чисел.
"""


def is_prime(n: int) -> bool:
    """
    Проверяет, является ли число простым.

    Args:
        n (int): Входное число (n ≥ 2).

    Returns:
        bool: True, если n — простое, иначе False.

    Raises:
        ValueError: Если n < 2.
    """
    if n < 2:
        raise ValueError("n должно быть ≥ 2")
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def gcd(a: int, b: int) -> int:
    """
    Находит наибольший общий делитель (НОД) двух целых чисел.

    Args:
        a (int): Первое число.
        b (int): Второе число.

    Returns:
        int: НОД(a, b). Если оба нуля, возвращает 0.
    """
    a, b = abs(a), abs(b)
    if a == 0 and b == 0:
        return 0
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """
    Находит наименьшее общее кратное (НОК) двух целых чисел.

    Args:
        a (int): Первое число (может быть 0).
        b (int): Второе число (может быть 0).

    Returns:
        int: НОК(a, b). Если хотя бы одно из чисел равно 0, возвращает 0.
    """
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)
