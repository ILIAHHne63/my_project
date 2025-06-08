import os

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from ..callbacks.callbacks import get_callbacks
from ..data.datamodule import FloodNetDataModule
from ..data.dataset_download import download_data_from_gdrive_folder
from ..loggers.logger import get_mlflow_logger
from ..models.unet_lightning import UNetLitModule
from ..utils.seed import seed_everything


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # Для отладки

    seed_everything(cfg.seed)

    output_dir = cfg.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "used_config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    if cfg.trainer.need_data_download:
        download_data_from_gdrive_folder(cfg)

    dm = FloodNetDataModule(
        data_dir=cfg.data.data_dir,
        img_size=cfg.data.img_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    dm.prepare_data()
    dm.setup()

    lit_model = UNetLitModule(cfg)

    loggers = []
    ml_cfg = cfg["logger"]["mlflow"]
    if ml_cfg.get("enable", False):
        loggers.append(get_mlflow_logger(cfg))

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        deterministic=cfg.trainer.deterministic,
        benchmark=cfg.trainer.benchmark,
        val_check_interval=cfg.trainer.val_check_interval,
        precision=cfg.trainer.precision,
        default_root_dir=cfg.trainer.default_root_dir,
        logger=loggers or None,
        callbacks=get_callbacks(cfg),
        enable_progress_bar=True,
    )

    trainer.fit(lit_model, datamodule=dm)


if __name__ == "__main__":
    main()
