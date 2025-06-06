from pytorch_lightning.loggers import TensorBoardLogger


def get_tensorboard_logger(cfg: dict):
    """
    CFG из раздела logger в конфиге YAML.
    Ожидает поля:
      save_dir: где сохранять логи
      name: имя эксперимента
    """
    return TensorBoardLogger(save_dir=cfg["save_dir"], name=cfg["name"])
