import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.dataset import FloodNetDataset


class FloodNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        img_size: int,
        batch_size: int = 4,
        num_workers: int = 2,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Самые важные датасеты заведомо инициализируем в setup()
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        # Здесь можно прописать логику скачивания/распаковки данных, если надо.
        pass

    def setup(self, stage=None):
        # stage может быть 'fit', 'validate', 'test' или None
        # При fit создаём train/val
        if stage in (None, "fit"):
            self.train_dataset = FloodNetDataset(
                data_path=self.data_dir,
                phase="train",
                img_size=self.img_size,
                augment=True,  # аугментации только для train
            )
            # Поскольку у нас нет отдельной папки val,
            # будем валидировать на тех же 'test' данных
            # (либо сделайте split вручную)
            self.val_dataset = FloodNetDataset(
                data_path=self.data_dir,
                phase="test",
                img_size=self.img_size,
                augment=False,
            )

        # Если захотим тестировать отдельно:
        if stage in ("test",):
            self.test_dataset = FloodNetDataset(
                data_path=self.data_dir,
                phase="test",
                img_size=self.img_size,
                augment=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
