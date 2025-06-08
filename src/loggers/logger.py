import subprocess

from pytorch_lightning.loggers import MLFlowLogger


def get_commit_id() -> str:
    """Возвращает текущий git commit id проекта."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
    except Exception:
        commit = "unknown"
    return commit


def get_mlflow_logger(cfg: dict) -> MLFlowLogger:
    """
    Создает и возвращает MLflow логгер:
      - experiment_name
      - tracking_uri
      - artifact_location (директория для arifacts)
    Логирует гиперпараметры и git commit id.
    """
    ml_cfg = cfg["logger"]["mlflow"]
    ml_logger = MLFlowLogger(
        experiment_name=ml_cfg["experiment_name"],
        tracking_uri=ml_cfg["tracking_uri"],
        artifact_location=ml_cfg.get("save_dir"),
    )
    # Собираем гиперпараметры
    hparams = {
        **cfg.get("trainer", {}),
        **cfg.get("model", {}),
        "git_commit_id": get_commit_id(),
    }
    ml_logger.log_hyperparams(hparams)
    return ml_logger
