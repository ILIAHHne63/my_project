import shutil
import subprocess
import tarfile
from pathlib import Path

import gdown


def download_data_from_gdrive_folder(
    folder_id: str = "1AJ8Ufs9J4QgIjRoet9DG5TFrNAgmtQJ0", dest_dir: str = "data"
):
    """
    Скачивает все файлы из Google Drive-папки (folder_id) в локальную папку dest_dir,
    затем упаковывает ее содержимое в tar-файл и добавляет его в DVC.

    Параметры:
    - folder_id: str — идентификатор Google Drive-папки (из URL).
    - dest_dir: str — папка, куда сохранять файлы (по умолчанию "data/raw").
    """
    base_path = Path(dest_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    download_url = f"https://drive.google.com/uc?id={folder_id}&export=download"
    tar_path = base_path / "tiny-floodnet-challenge.tar.gz"

    gdown.download(url=download_url, output=str(tar_path), quiet=False)
    with tarfile.open(tar_path, mode="r:*") as tar:
        tar.extractall(path=base_path)

    gdown.download_folder(
        id=folder_id, output="data/tiny-floodnet-challenge.tar.gz", quiet=False
    )
    extracted = base_path / "tiny-floodnet-challenge"
    for subset in ["train", "test"]:
        src = extracted / subset
        dst = base_path / subset
        if src.exists():
            print(f"Moving {src} → {dst}")
            # если dst уже есть — можно удалить или перезаписать
            if dst.exists():
                shutil.rmtree(dst)
            src.rename(dst)
        else:
            print(f"Папка не найдена: {src}")

    if extracted.exists():
        print(f"Removing directory {extracted}")
        shutil.rmtree(extracted)
    try:
        subprocess.run(
            ["dvc", "add", "data/test"],
            check=True,
        )
        subprocess.run(["dvc", "add", "data/train"], check=True)
    except subprocess.CalledProcessError as e:
        print("Ошибка при выполнении 'dvc add' или 'git commit':", e)
        raise
