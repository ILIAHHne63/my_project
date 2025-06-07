import argparse
import os

import numpy as np
import torch
import yaml
from PIL import Image

from ..data.datamodule import FloodNetDataModule
from ..data.dataset_download import download_data_from_gdrive_folder
from ..models.unet_lightning import UNetLitModule
from ..utils.seed import seed_everything

# Цветовая палитра для классов FloodNet
PALETTE = {
    0: (0, 0, 0),  # фон
    1: (128, 0, 0),  # класс 1
    2: (0, 128, 0),  # класс 2
    3: (128, 128, 0),  # класс 3
    4: (0, 0, 128),  # класс 4
    5: (128, 0, 128),  # класс 5
    6: (0, 128, 128),  # класс 6
    7: (128, 128, 128),  # класс 7
}


def apply_palette(mask_np: np.ndarray, palette: dict) -> Image.Image:
    """
    Преобразует 2D-массив меток в RGB-изображение по палитре.
    """
    h, w = mask_np.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in palette.items():
        color_img[mask_np == cls_id] = color
    return Image.fromarray(color_img)


def save_mask(mask_tensor: torch.Tensor, output_path: str, palette: dict):
    """
    Сохраняет тензор меток в цветное PNG-изображение.
    """
    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
    img = apply_palette(mask_np, palette)
    img.save(output_path)


def main(
    config_path: str, checkpoint_path: str, output_dir: str, need_data_download: bool
):
    # Загрузка конфига и фиксация сидов
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    seed_everything(cfg["seed"])

    # Создаем папки для GT и предсказаний
    gt_dir = os.path.join(output_dir, "gt")
    pred_dir = os.path.join(output_dir, "predicted")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # Скачать данные при необходимости
    if need_data_download:
        download_data_from_gdrive_folder()

    # Инициализация DataModule
    dm = FloodNetDataModule(
        data_dir=cfg["data"]["data_dir"],
        img_size=cfg["data"]["img_size"],
        batch_size=1,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
    )
    dm.prepare_data()
    dm.setup(stage="test")

    # Загрузка модели
    lit_model = UNetLitModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    lit_model.eval()
    lit_model.freeze()

    # Инференс
    test_loader = dm.test_dataloader()
    for idx, batch in enumerate(test_loader):
        images, gt_masks = batch
        images = images.to(lit_model.device)

        with torch.no_grad():
            logits = lit_model(images)
        preds = torch.argmax(logits, dim=1).squeeze(0)
        gt = gt_masks.squeeze(0)

        # Пути сохранения масок
        gt_path = os.path.join(gt_dir, f"gt_{idx:04d}.png")
        pred_path = os.path.join(pred_dir, f"pred_{idx:04d}.png")

        # Сохранение
        save_mask(gt, gt_path, PALETTE)
        save_mask(preds, pred_path, PALETTE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloodNet UNet Inference with Palette")
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
        default="outputs",
        help="Directory to save GT and predicted masks",
    )
    parser.add_argument(
        "--need_data_download",
        "-n",
        type=bool,
        help="Download dataset if needed",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output, args.need_data_download)
