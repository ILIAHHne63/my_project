import argparse
import os

import yaml
from pytorch_lightning import Trainer

from ..callbacks.callbacks import get_callbacks
from ..data.datamodule import FloodNetDataModule
from ..loggers.tensorboard_logger import get_tensorboard_logger
from ..models.unet_lightning import UNetLitModule
from ..utils.seed import seed_everything

# Если понадобится Wandb:
# from loggers.wandb_logger import get_wandb_logger


def main(config_path: str):
    # 1. Загружаем конфиг
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Устанавливаем сид для воспроизводимости
    seed_everything(cfg["seed"])

    # 3. Создаём выходную директорию эксперимента
    output_dir = cfg["experiment"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    # Сохраним копию конфига туда
    with open(os.path.join(output_dir, "used_config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # 4. Инициализируем DataModule
    dm = FloodNetDataModule(
        data_dir=cfg["data"]["data_dir"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )
    dm.prepare_data()
    dm.setup()

    # 5. Инициализируем LightningModule
    lit_model = UNetLitModule(cfg)

    # 6. Выбираем логгер
    logger_type = cfg["logger"]["type"]
    loggers = []
    if logger_type == "tensorboard":
        tb_logger = get_tensorboard_logger(cfg["logger"])
        loggers.append(tb_logger)
    # elif logger_type == "wandb":
    #     wandb_logger = get_wandb_logger(cfg["logger"])
    #     loggers.append(wandb_logger)

    logger_collection = loggers or None

    # 7. Колбэки
    callbacks = get_callbacks(cfg)

    # 8. Параметры Trainer
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

    # 9. Запускаем обучение
    trainer.fit(lit_model, datamodule=dm)

    # 10. Тестирование (если нужно)
    # trainer.test(lit_model, datamodule=dm)

    # 11. Опционально: predict
    # trainer.predict(lit_model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloodNet UNet Training")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/floodnet_unet.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    main(args.config)
