"""
Модуль data_loader
------------------
Содержит функции для загрузки и простого препроцессинга «данных».
"""

import csv
from typing import Dict, List


def load_csv(path: str) -> List[Dict[str, str]]:
    """
    Загружает CSV-файл и возвращает список словарей,
    где ключи — столбцы, а значения — строки.

    Args:
        path (str): Путь до CSV-файла.

    Returns:
        List[Dict[str, str]]: Список, где каждый элемент – словарь
        {имя_столбца: значение_в_строке}.
    """
    data = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def filter_by_column(
    data: List[Dict[str, str]], column: str, value: str
) -> List[Dict[str, str]]:
    """
    Фильтрует загруженные данные по значению в указанном столбце.

    Args:
        data (List[Dict[str, str]]): Входной список словарей (из load_csv).
        column (str): Имя столбца, по которому идёт фильтрация.
        value (str): Строковое значение, по которому фильтруем.

    Returns:
        List[Dict[str, str]]: Отфильтрованный список (только те строки,
        где data[i][column] == value).
    """
    if column not in data[0]:
        raise KeyError(f"Столбец '{column}' отсутствует в данных")
    return [row for row in data if row[column] == value]
