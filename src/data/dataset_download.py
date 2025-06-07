import subprocess
from pathlib import Path

import gdown


def download_data_from_gdrive_folder(
    folder_id: str = "17fVno9Dgm6RrLuntWnV4OhL-BCYQnmjp", dest_dir: str = "data"
):
    """
    Скачивает все файлы из Google Drive-папки (folder_id) в локальную папку dest_dir,
    затем упаковывает ее содержимое в tar-файл и добавляет его в DVC.

    Параметры:
    - folder_id: str — идентификатор Google Drive-папки (из URL).
    - dest_dir: str — папка, куда сохранять файлы (по умолчанию "data/raw").
    """
    raw_path = Path(dest_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    gdown.download_folder(id=folder_id, output=str(raw_path), quiet=False)
    try:
        subprocess.run(["dvc", "add", "my-project/data/test"], check=True)
        subprocess.run(["dvc", "add", "my-project/data/train"], check=True)
        subprocess.run(["git", "add", "my-project/data/test.dvc"], check=True)
        subprocess.run(["git", "add", "my-project/data/train.dvc"], check=True)
        subprocess.run(
            ["git", "commit", "-m", "Добавили данные из Google Drive в DVC"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("Ошибка при выполнении 'dvc add' или 'git commit':", e)
        raise

    try:
        subprocess.run(["dvc", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print("Ошибка при выполнении 'dvc push':", e)
        raise


# if __name__ == "__main__":
#     download_data_from_gdrive_folder(
#         folder_id="17fVno9Dgm6RrLuntWnV4OhL-BCYQnmjp",
#         dest_dir="data/raw",
#         tar_name="data_raw.tar",
#     )
