import shutil
import subprocess
import tarfile
from pathlib import Path

import gdown
from omegaconf import DictConfig


def download_data_from_gdrive_folder(cfg: DictConfig):
    """
    Скачивает файлы из Google Drive, распаковывает их
    и добавляет через DVC.
    Параметры берутся из cfg.download и cfg.dvc.
    """
    download_cfg = cfg.download
    folder_id = download_cfg.folder_id
    dest_dir = Path(download_cfg.dest_dir)
    tar_path = dest_dir / download_cfg.tar_name

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Скачиваем tar.gz
    url = f"https://drive.google.com/uc?id={folder_id}&export=download"
    print(f"Downloading {url} → {tar_path}")
    gdown.download(url=url, output=str(tar_path), quiet=False)

    # Распаковываем
    with tarfile.open(tar_path, mode="r:*") as tar:
        tar.extractall(path=dest_dir)

    # Перемещаем поддиректории train/test
    extracted = dest_dir / download_cfg.tar_name.replace(".tar.gz", "")
    for subset in download_cfg.subsets:
        src = extracted / subset
        dst = dest_dir / subset
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            src.rename(dst)
        else:
            print(f"Warning: subset folder not found: {src}")

    # Удаляем распакованную папку
    if extracted.exists():
        shutil.rmtree(extracted)

    # Добавляем данные в DVC
    for path in cfg.dvc.add_paths:
        try:
            print(f"Running: dvc add {path}")
            subprocess.run(["dvc", "add", path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running `dvc add {path}`: {e}")
            raise
