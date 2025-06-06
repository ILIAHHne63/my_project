import csv

import pytest

from src.data_loader import filter_by_column, load_csv


@pytest.fixture
def sample_csv_path(tmp_path):
    """
    Создаёт во временной папке небольшой CSV-файл и возвращает путь до него.
    """
    data = [
        {"name": "Alice", "age": "30", "city": "Moscow"},
        {"name": "Bob", "age": "25", "city": "Berlin"},
        {"name": "Charlie", "age": "30", "city": "Moscow"},
    ]
    file = tmp_path / "people.csv"
    with open(file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "age", "city"])
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    return str(file)


def test_load_csv(sample_csv_path):
    rows = load_csv(sample_csv_path)
    assert isinstance(rows, list)
    assert len(rows) == 3
    # Проверяем ключи и значения в первом словаре
    assert rows[0]["name"] == "Alice"
    assert rows[1]["age"] == "25"
    assert rows[2]["city"] == "Moscow"


def test_filter_by_column_success(sample_csv_path):
    data = load_csv(sample_csv_path)
    filtered = filter_by_column(data, column="city", value="Moscow")
    # В нашем примере должно вернуть 2 строки, где city == "Moscow"
    assert len(filtered) == 2
    for row in filtered:
        assert row["city"] == "Moscow"


def test_filter_by_column_missing_column(sample_csv_path):
    data = load_csv(sample_csv_path)
    with pytest.raises(KeyError):
        filter_by_column(data, column="nonexistent", value="xxx")
