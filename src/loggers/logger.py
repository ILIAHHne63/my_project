# src/utils/logger.py

import subprocess

from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger


def get_commit_id() -> str:
    """Возвращает текущий git commit id проекта."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
    except Exception:
        commit = "unknown"
    return commit


def get_loggers(cfg: dict):
    """
    Создаёт и возвращает список логгеров в зависимости от конфигурации:
      - MLFlowLogger (если enabled в cfg["logger"]["mlflow"])
      - TensorBoardLogger (если enabled в cfg["logger"]["tensorboard"])
    Каждый логгер получает одинаковый набор гиперпараметров + git commit id.
    """
    loggers = []

    # --- MLflow Logger ---
    ml_cfg = cfg.get("logger", {}).get("mlflow", {})
    if ml_cfg.get("enable", False):
        ml_logger = MLFlowLogger(
            experiment_name=ml_cfg["experiment_name"],
            tracking_uri=ml_cfg["tracking_uri"],
            artifact_location=ml_cfg["save_dir"],
        )
        # Логируем гиперпараметры и git commit id
        hparams = {
            **cfg.get("training", {}),
            **cfg.get("model", {}),
            "git_commit_id": get_commit_id(),
        }
        ml_logger.log_hyperparams(hparams)
        loggers.append(ml_logger)

    # --- TensorBoard Logger ---
    tb_cfg = cfg.get("logger", {}).get("tensorboard", {})
    if tb_cfg.get("enable", False):
        tb_logger = TensorBoardLogger(save_dir=tb_cfg["save_dir"], name=tb_cfg["name"])
        # Логируем те же гиперпараметры
        tb_logger.log_hyperparams(
            {
                **cfg.get("training", {}),
                **cfg.get("model", {}),
                "git_commit_id": get_commit_id(),
            }
        )
        loggers.append(tb_logger)

    return loggers
