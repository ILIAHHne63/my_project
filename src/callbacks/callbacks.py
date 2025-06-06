from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def get_callbacks(cfg: dict):
    callbacks = []

    # ModelCheckpoint
    ckpt_cfg = cfg["callbacks"]["checkpoint"]
    if ckpt_cfg["enabled"]:
        checkpoint = ModelCheckpoint(
            monitor=ckpt_cfg["monitor"],
            save_top_k=ckpt_cfg["save_top_k"],
            mode=ckpt_cfg["mode"],
            dirpath=ckpt_cfg["dirpath"],
            filename=ckpt_cfg["filename"],
        )
        callbacks.append(checkpoint)

    # EarlyStopping
    es_cfg = cfg["callbacks"]["early_stop"]
    if es_cfg.get("enabled", False):
        early_stop = EarlyStopping(
            monitor=es_cfg["monitor"], patience=es_cfg["patience"], mode=es_cfg["mode"]
        )
        callbacks.append(early_stop)

    # LearningRateMonitor
    lr_cfg = cfg["callbacks"]["lr_monitor"]
    if lr_cfg.get("enabled", False):
        lr_mon = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_mon)

    # DeviceStatsMonitor
    ds_cfg = cfg["callbacks"]["device_stats"]
    if ds_cfg.get("enabled", False):
        ds_mon = DeviceStatsMonitor(
            log_gpu_memory=ds_cfg.get("log_gpu_memory", False),
            log_cpu_stats=ds_cfg.get("log_cpu_stats", False),
        )
        callbacks.append(ds_mon)

    return callbacks
