# ============================================
# Файл: src/trainers/inference.py
# Назначение: инференс (вывод предсказаний) сохранённой
# модели UNet на новом наборе данных
# ============================================

import argparse
import os

import numpy as np
import torch
import yaml
from PIL import Image

from data.datamodule import FloodNetDataModule
from models.unet_lightning import UNetLitModule
from utils.seed import seed_everything


def save_mask(mask_tensor: torch.Tensor, output_path: str):
    """
    Сохраняет одномерный тензор меток (H×W) в PNG-файл с палитрой
    (каждый класс — свой цвет).
    mask_tensor: torch.LongTensor, shape [H, W], значения от 0
    до num_classes-1
    output_path: путь, куда сохранить mask.png
    """
    # Преобразуем тензор на CPU и в NumPy
    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
    # Здесь можно наложить цветовую палитру; для простоты
    # сохраняем оттенки серого
    img = Image.fromarray(mask_np, mode="L")
    img.save(output_path)


def main(config_path: str, checkpoint_path: str, output_dir: str):
    # ----------------------------------------
    # 1. Загрузка конфига и установка сида
    # ----------------------------------------
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["seed"])

    # ----------------------------------------
    # 2. Создание выходной директории
    # ----------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------
    # 3. Инициализируем DataModule для теста
    # ----------------------------------------
    dm = FloodNetDataModule(
        data_dir=cfg["data"]["data_dir"],
        img_size=cfg["data"]["img_size"],
        batch_size=1,  # Для инференса обычно batch_size=1
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
    )
    dm.prepare_data()
    dm.setup(stage="test")  # загружаем только test-дату

    # ----------------------------------------
    # 4. Загружаем модель из чекпоинта
    # ----------------------------------------
    # Метод load_from_checkpoint автоматически подаст на вход
    # тот же cfg, что был сохранён
    lit_model = UNetLitModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    lit_model.eval()
    lit_model.freeze()

    # ----------------------------------------
    # 5. Настраиваем Trainer для предсказания
    # ----------------------------------------
    # Здесь не нужен логгер, но оставим TensorBoardLogger
    # для совместимости (либо None)
    # tb_logger = TensorBoardLogger(
    #     save_dir=cfg["logger"]["save_dir"], name=cfg["logger"]["name"]
    # )

    # trainer = Trainer(
    #     accelerator=cfg["trainer"]["accelerator"],
    #     devices=cfg["trainer"]["devices"],
    #     logger=tb_logger,
    #     enable_progress_bar=True,
    # )

    # ----------------------------------------
    # 6. Запускаем предсказания
    # ----------------------------------------
    # predict() вернёт список тензоров [batch_idx][логиты], но поскольку
    # batch_size=1, удобнее сделать цикл вручную:
    test_loader = dm.test_dataloader()
    for idx, batch in enumerate(test_loader):
        images, _ = batch
        images = images.to(lit_model.device)

        # Получаем предсказания (логиты) [1, C, H, W]
        with torch.no_grad():
            logits = lit_model(images)

        # Берём argmax по каналам, чтобы получить метки [1, H, W]
        preds = torch.argmax(logits, dim=1).squeeze(0)  # [H, W] как LongTensor

        # Сохраняем маску в файл
        out_mask_path = os.path.join(output_dir, f"prediction_{idx:04d}.png")
        save_mask(preds, out_mask_path)

    print(f"Инференс завершён. Предсказания сохранены в папке {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloodNet UNet Inference")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/floodnet_unet.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        "-ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="inference_outputs",
        help="Directory to save predicted masks",
    )

    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output)
