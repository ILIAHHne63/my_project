# UNet-Prod: Flood Segmentation with UNet

## Описание проекта

**UNet-Prod** — это модульный проект для семантической сегментации затопленных
участков на спутниковых изображениях с использованием архитектуры UNet. Цель
проекта — предоставить полный конвейер от подготовки данных до реализации
тренировки и инференса:

- **Предобработка данных**: загрузка и подготовка спутниковых изображений и
  масок.
- **Обучение модели**: конфигурируемая тренировка UNet с возможностью выбора
  гиперпараметров.
- **Инференс**: скрипт для применения обученной модели к изображениям.

### Основные модули

- `src/data`

  - Скрипты и утилиты для загрузки, валидации и препроцессинга данных.
  - Интеграция с DVC для управления версиями датасета.

- `src/models`

  - Определение архитектуры UNet, варианты энкодеров (ResNet, EfficientNet и
    т. д.).
  - Компоненты потерь, метрики (IoU, Dice, Accuracy).

- `src/trainers`

  - Модуль `train`: загрузка конфигурации, формирование DataLoader, обучение,
    логирование в MLflow/TensorBoard.
  - Модуль `inference`: загрузка чекпоинта, применение модели к новым данным,
    сохранение предсказаний.

- `configs`

  - YAML-файлы с настройками для разных экспериментов (настройки обучения, пути
    к данным, гиперпараметры).

- `experiments`

  - Автоматически создаваемая структура для хранения чекпоинтов, логов и
    итоговых артефактов.

- `plots/mlflow_logs`

  - Каталог для хранения данных MLflow (база данных SQLite + артефакты).

> **Важно:** все CLI-скрипты (`train`, `inference`) используют конфигурацию из
> папки `configs` и могут автоматически скачивать данные с помощью DVC, если
> указан флаг `--need_data_download True`.

---

## Технические детали

### 📦 Setup

1. **Клонирование репозитория**

   ```bash
   git clone https://github.com/ILIAHHne63/unet-prod.git
   cd unet-prod
   ```

2. **Установка Poetry** (если ещё не установлен)

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Установка зависимостей**

   ```bash
   poetry install
   ```

4. **Активация виртуального окружения** _(опционально, но рекомендуется)_

   ```bash
   poetry shell
   ```

   Либо вручную:

   ```bash
   source $(poetry env info --path)/bin/activate
   ```

5. **(Опционально) Настройка MLflow для отслеживания экспериментов** Для запуска
   локального сервера MLflow:

   ```bash
   poetry run mlflow server \
     --backend-store-uri sqlite:///mlruns.db \
     --default-artifact-root ./plots/mlflow_logs \
     --host 127.0.0.1 \
     --port 5001
   ```

   После запуска можно открыть [http://127.0.0.1:5001](http://127.0.0.1:5001) в
   браузере для просмотра экспериментов.

6. **(При необходимости) Установка DVC**

   ```bash
   pip install dvc
   ```

   Репозиторий уже содержит `.dvc`-файлы и `dvc.yaml`. Для скачивания датасета
   используется флаг `--need_data_download True` в скриптах.

### 🎓 Train

Запуск обучения модели включает следующие шаги: загрузка данных (через DVC, если
требуется), препроцессинг, обучение, логирование метрик.

1. **Подготовка данных** Если у вас ещё нет локальной копии датасета, запуск
   скрипта тренировки с флагом `--need_data_download True` автоматически скачает
   данные в папку `data/`:

   - Структура после скачивания:

     ```
     data/
       ├── train/
       │     ├── images/    (*.png, *.jpg)
       │     └── masks/     (*.png, *.jpg)
       └── test/
             ├── images/    (*.png, *.jpg)
             └── masks/     (*.png, *.jpg)
     ```

2. **Запуск тренировки**

   ```bash
   poetry run python -m src.trainers.train \
     --config configs/floodnet_unet.yaml \
     --need_data_download True
   ```

   - `--config` — путь к YAML-конфигу (например, `configs/floodnet_unet.yaml`).
   - `--need_data_download` — если `True`, то скачивает данные в папку `data/`
     (использует DVC).
   - Пример ключевых параметров в YAML:

     ```yaml
     data:
       train_dir: data/train/images
       train_mask_dir: data/train/masks
     model:
       encoder: resnet34
       pretrained: True
       in_channels: 3
       out_classes: 1
     training:
       batch_size: 8
       epochs: 50
       lr: 1e-4
       scheduler:
         name: CosineAnnealingLR
         T_max: 10
     logging:
       use_mlflow: True
       mlflow_uri: http://127.0.0.1:5001
       experiment_name: floodnet_unet
     ```

3. **Что происходит в процессе**

   - Скрипт читает конфигурационный файл `configs/floodnet_unet.yaml`.
   - Если `--need_data_download True`, выполняется команда `dvc pull` для
     актуализации датасета.
   - Строятся PyTorch DataLoader’ы для `train` и `val`.
   - Инициализируется модель UNet с заданным энкодером и гиперпараметрами.
   - Выполняется обучение, в ходе которого:

     - Логируются метрики (loss, IoU, Dice) в TensorBoard
       (`tb_logs/floodnet_unet/`).
     - Если включён MLflow, метрики и артефакты сохраняются в MLflow.
     - По окончании каждой эпохи сохраняется чекпоинт в папку
       `experiments/floodnet_unet/checkpoints/` (файлы `unet-epoch=<N>.ckpt`).

4. **Результаты**

   - **Чекпоинты**: `experiments/floodnet_unet/checkpoints/unet-epoch=<N>.ckpt`
   - **Логи TensorBoard**: `tb_logs/floodnet_unet/`
   - **MLflow артефакты** (если запущен MLflow Server):

     - SQLite БД `mlruns.db` в корне проекта.
     - Каталог артефактов `plots/mlflow_logs/`.

### 📦 Production Preparation

После успешного обучения модели необходимо подготовить её для
продакшен-окружения. В рамках данного проекта рекомендуется выполнить следующие
шаги:

1. **Экспорт модели в ONNX** Для упрощения интеграции в различные
   продакшен-окружения (Docker, GPU/CPU inference) можно конвертировать
   сохранённый чекпоинт в формат ONNX. Например, создайте скрипт
   `export_to_onnx.py`:

   ```python
   import torch
   from src.models.unet import UNet  # пример импорта модели
   from omegaconf import OmegaConf

   # Загрузка конфигурации и чекпоинта
   cfg = OmegaConf.load("configs/floodnet_unet.yaml")
   model = UNet(
       encoder_name=cfg.model.encoder,
       encoder_weights='imagenet' if cfg.model.pretrained else None,
       in_channels=cfg.model.in_channels,
       classes=cfg.model.out_classes
   )
   ckpt = torch.load("experiments/floodnet_unet/checkpoints/unet-epoch=best.ckpt")
   model.load_state_dict(ckpt["state_dict"])
   model.eval()

   # Создание фиктивного входа (размер batch=1, каналы=3, H×W из конфига)
   dummy_input = torch.randn(1, cfg.model.in_channels, 256, 256)

   # Экспорт в ONNX
   torch.onnx.export(
       model,
       dummy_input,
       "experiments/floodnet_unet/unet_floodnet.onnx",
       input_names=["input"],
       output_names=["output"],
       opset_version=11
   )
   ```

   Запуск:

   ```bash
   poetry run python export_to_onnx.py
   ```

   В результате получится файл `unet_floodnet.onnx`, готовый для загрузки в
   любые ONNX Runtime окружения.

2. **Подготовка окружения для inference** Для минимизации зависимостей
   продакшен-скрипта рекомендуется собрать с помощью Poetry отдельный
   минимальный `pyproject.toml` или указать в README, какие библиотеки
   необходимы для инференса (например, `torch`, `onnxruntime`, `opencv-python` и
   т. д.). Пример секции в `pyproject.toml` для продакшен-окружения:

   ```toml
   [tool.poetry.group.prod.dependencies]
   python = "^3.11"
   onnxruntime = "^1.13.0"
   opencv-python = "^4.7.0"
   numpy = "^1.24.0"
   ```

   После этого можно установить зависимости:

   ```bash
   poetry install --with prod
   ```

3. **Артефакты для поставки**

   - `unet_floodnet.onnx` (или PyTorch чекпоинт, если планируете запускать
     inference в PyTorch).
   - Скрипт/модуль инференса (например, `src/trainers/inference.py`).
   - Минимальный `requirements.txt` или `pyproject.toml` для
     продакшен-окружения.
   - Примеры входных изображений и структура папок `data/test/…/`.

### 🔍 Infer

После тренировки и подготовки модели к продакшен-формату необходимо предоставить
команду для запуска инференса на новых данных. Для этого предусмотрён отдельный
скрипт, который минимально зависит от остальных компонентов.

1. **Структура входных данных**

   ```
   data/
     └── test/
         ├── images/   (*.png, *.jpg)  — изображения для предсказания
         └── masks/    (*.png, *.jpg)  — (опционально) реальные маски для валидации
   ```

2. **Запуск inf­erence (PyTorch-версия)**

   ```bash
   poetry run python -m src.trainers.inference \
     --ckpt experiments/floodnet_unet/checkpoints/unet-epoch=18.ckpt \
     --need_data_download False
   ```

   - `--ckpt`: путь к чекпоинту модели (можно указать любой файл из
     `experiments/.../checkpoints/`).
   - `--need_data_download`: если `True`, скачивает данные (DVC) в папку
     `data/`; иначе ожидает, что папка уже есть.
   - В результате работы скрипта:

     - Предсказанные маски сохранятся в папку `outputs/predicted/` (формат
       `.png`).
     - Исходные (ground truth) маски автоматически копируются в `outputs/gt/`
       (если они есть в `data/test/masks/`).

3. **Запуск inf­erence (ONNX-версия)** Если вы экспортировали модель в ONNX и
   подготовили минимальный продакшен-скрипт, его можно запускать независимо:

   ```bash
   poetry run python onnx_inference.py \
     --model_path experiments/floodnet_unet/unet_floodnet.onnx \
     --input_dir data/test/images \
     --output_dir outputs/predicted_onnx
   ```

   Пример аргументов:

   - `--model_path` — путь к ONNX-модели.
   - `--input_dir` — папка с новыми изображениями.
   - `--output_dir` — папка для сохранения предсказаний.

   **Пример простого `onnx_inference.py`:**

   ```python
   import os
   import argparse
   import cv2
   import numpy as np
   import onnxruntime as ort

   def preprocess(img_path, input_size=(256, 256)):
       img = cv2.imread(img_path)
       img = cv2.resize(img, input_size)
       img = img.astype(np.float32) / 255.0
       img = np.transpose(img, (2, 0, 1))
       return img[np.newaxis, :]

   def postprocess(mask, orig_shape):
       mask = mask.squeeze()
       mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
       mask = (mask > 0.5).astype(np.uint8) * 255
       return mask

   def run_inference(model_path, input_dir, output_dir):
       os.makedirs(output_dir, exist_ok=True)
       session = ort.InferenceSession(model_path)
       input_name = session.get_inputs()[0].name

       for fname in os.listdir(input_dir):
           if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
               continue
           img_path = os.path.join(input_dir, fname)
           orig = cv2.imread(img_path)
           inp = preprocess(img_path)
           pred = session.run(None, {input_name: inp})[0]
           mask = postprocess(pred, orig.shape[:2])
           cv2.imwrite(os.path.join(output_dir, fname), mask)

   if __name__ == "__main__":
       parser = argparse.ArgumentParser()
       parser.add_argument("--model_path", type=str, required=True)
       parser.add_argument("--input_dir", type=str, required=True)
       parser.add_argument("--output_dir", type=str, required=True)
       args = parser.parse_args()

       run_inference(args.model_path, args.input_dir, args.output_dir)
   ```

   Запуск:

   ```bash
   poetry run python onnx_inference.py \
     --model_path experiments/floodnet_unet/unet_floodnet.onnx \
     --input_dir data/test/images \
     --output_dir outputs/predicted_onnx
   ```

   В результате получаем папку `outputs/predicted_onnx/` с бинарными масками в
   формате `.png`.

---

## Полезные команды

- **Установка зависимостей и активация среды**

  ```bash
  poetry install
  poetry shell
  ```

- **Запуск MLflow Server**

  ```bash
  poetry run mlflow server \
    --backend-store-uri sqlite:///mlruns.db \
    --default-artifact-root ./plots/mlflow_logs \
    --host 127.0.0.1 \
    --port 5001
  ```

- **Тренировка модели**

  ```bash
  poetry run python -m src.trainers.train \
    --config configs/floodnet_unet.yaml \
    --need_data_download True
  ```

- **Инференс (PyTorch-чекпоинт)**

  ```bash
  poetry run python -m src.trainers.inference \
    --ckpt experiments/floodnet_unet/checkpoints/unet-epoch=18.ckpt \
    --need_data_download False
  ```

- **Экспорт модели в ONNX**

  ```bash
  poetry run python export_to_onnx.py
  ```

- **Инференс (ONNX-модель)**

  ```bash
  poetry run python onnx_inference.py \
    --model_path experiments/floodnet_unet/unet_floodnet.onnx \
    --input_dir data/test/images \
    --output_dir outputs/predicted_onnx
  ```

---

Теперь любой новый участник команды сможет быстро склонировать репозиторий,
настроить окружение, обучить модель и запустить инференс на новых данных.
Удачной работы!
