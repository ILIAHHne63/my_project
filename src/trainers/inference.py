# src/trainers/inference.py
from pathlib import Path

import fire
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from ..data.datamodule import FloodNetDataModule
from ..data.dataset_download import download_data_from_gdrive_folder
from ..models.unet_lightning import UNetLitModule
from ..utils.seed import seed_everything


def apply_palette(mask_np: np.ndarray, palette: dict) -> Image.Image:
    h, w = mask_np.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in palette.items():
        color_img[mask_np == int(cls_id)] = color
    return Image.fromarray(color_img)


def save_mask(mask_tensor: torch.Tensor, output_path: Path, palette: dict):
    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
    img = apply_palette(mask_np, palette)
    img.save(str(output_path))


def run_inference(cfg: DictConfig):
    """Вся логика инференса, cfg уже готов."""
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)

    out_root = Path(cfg.inference.output_dir)
    gt_dir = out_root / "gt"
    pred_dir = out_root / "predicted"
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    if cfg.inference.need_data_download:
        download_data_from_gdrive_folder(cfg)

    dm = FloodNetDataModule(
        data_dir=cfg.data.data_dir,
        img_size=cfg.data.img_size,
        batch_size=cfg.inference.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    dm.prepare_data()
    dm.setup(stage="test")

    lit_model = UNetLitModule.load_from_checkpoint(cfg.model.checkpoint_path, cfg=cfg)
    lit_model.eval()
    lit_model.freeze()

    for idx, batch in enumerate(dm.test_dataloader()):
        images, gt_masks = batch
        images = images.to(lit_model.device)

        with torch.no_grad():
            logits = lit_model(images)
        preds = torch.argmax(logits, dim=1).squeeze(0)
        gt = gt_masks.squeeze(0)

        gt_path = gt_dir / f"gt_{idx:04d}.png"
        pred_path = pred_dir / f"pred_{idx:04d}.png"

        save_mask(gt, gt_path, cfg.palette)
        save_mask(preds, pred_path, cfg.palette)


def inference(checkpoint_path: str):
    """
    Точка входа для fire: принимает checkpoint_path,
    остальное подтягивается из Hydra-конфигов.
    """
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config")

    cfg.model.checkpoint_path = checkpoint_path
    run_inference(cfg)


if __name__ == "__main__":
    fire.Fire(inference)
