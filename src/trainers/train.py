import argparse
import os

import yaml
from pytorch_lightning import Trainer

from ..callbacks.callbacks import get_callbacks
from ..data.datamodule import FloodNetDataModule
from ..data.dataset_download import download_data_from_gdrive_folder
from ..loggers.tensorboard_logger import get_tensorboard_logger
from ..models.unet_lightning import UNetLitModule
from ..utils.seed import seed_everything


def main(config_path: str, need_data_download: bool):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["seed"])
    output_dir = cfg["experiment"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "used_config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    if need_data_download:
        download_data_from_gdrive_folder()

    dm = FloodNetDataModule(
        data_dir=cfg["data"]["data_dir"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )
    dm.prepare_data()
    dm.setup()

    lit_model = UNetLitModule(cfg)

    logger_type = cfg["logger"]["type"]
    loggers = []
    if logger_type == "tensorboard":
        tb_logger = get_tensorboard_logger(cfg["logger"])
        loggers.append(tb_logger)

    logger_collection = loggers or None
    callbacks = get_callbacks(cfg)

    trainer_params = {
        "max_epochs": cfg["trainer"]["max_epochs"],
        "accelerator": cfg["trainer"]["accelerator"],
        "devices": cfg["trainer"]["devices"],
        "deterministic": cfg["trainer"]["deterministic"],
        "benchmark": cfg["trainer"]["benchmark"],
        "val_check_interval": cfg["trainer"]["val_check_interval"],
        "precision": cfg["trainer"]["precision"],
        "default_root_dir": cfg["trainer"]["default_root_dir"],
        "logger": logger_collection,
        "callbacks": callbacks,
        "enable_progress_bar": True,
    }

    trainer = Trainer(**trainer_params)
    trainer.fit(lit_model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloodNet UNet Training")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/floodnet_unet.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--need_data_download",
        "-n",
        type=bool,
        default=False,
        help="Do I need to download the dataset",
    )
    args = parser.parse_args()
    main(args.config, args.need_data_download)
