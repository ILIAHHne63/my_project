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


def save_mask(mask_tensor: torch.Tensor, output_path: str):
    """
    Сохраняет одномерный тензор меток (H×W) в PNG-файл с палитрой
    (каждый класс — свой цвет).
    mask_tensor: torch.LongTensor, shape [H, W], значения от 0
    до num_classes-1
    output_path: путь, куда сохранить mask.png
    """
    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
    img = Image.fromarray(mask_np, mode="L")
    img.save(output_path)


def main(
    config_path: str, checkpoint_path: str, output_dir: str, need_data_download: bool
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["seed"])
    os.makedirs(output_dir, exist_ok=True)

    if need_data_download:
        download_data_from_gdrive_folder()

    dm = FloodNetDataModule(
        data_dir=cfg["data"]["data_dir"],
        img_size=cfg["data"]["img_size"],
        batch_size=1,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
    )
    dm.prepare_data()
    dm.setup(stage="test")

    lit_model = UNetLitModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    lit_model.eval()
    lit_model.freeze()

    test_loader = dm.test_dataloader()
    for idx, batch in enumerate(test_loader):
        images, _ = batch
        images = images.to(lit_model.device)

        with torch.no_grad():
            logits = lit_model(images)

        preds = torch.argmax(logits, dim=1).squeeze(0)
        out_mask_path = os.path.join(output_dir, f"prediction_{idx:04d}.png")
        save_mask(preds, out_mask_path)


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
    parser.add_argument(
        "--need_data_download",
        "-n",
        type=bool,
        default=False,
        help="Do I need to download the dataset",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output, args.need_data_download)
