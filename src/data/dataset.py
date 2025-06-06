import os

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2 as ToTensor
from PIL import Image
from torch.utils.data import Dataset


class FloodNetDataset(Dataset):
    """
    Классы:
      0: Background, 1: Building, 2: Road, 3: Water,
      4: Tree, 5: Vehicle, 6: Pool, 7: Grass
    """

    def __init__(
        self,
        data_path: str,
        phase: str,
        img_size: int,
        augment: bool = False,
    ):
        super().__init__()
        self.num_classes = 8
        self.data_path = data_path
        self.phase = phase  # 'train' или 'test'
        self.img_size = img_size

        # Список ID изображений (без расширения)
        image_dir = os.path.join(self.data_path, self.phase, "image")
        self.items = [fname.split(".")[0] for fname in os.listdir(image_dir)]

        # Определяем аугментации
        if augment and phase == "train":
            self.transform = A.Compose(
                [
                    A.RandomResizedCrop(height=self.img_size, width=self.img_size),
                    ToTensor(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.CenterCrop(height=self.img_size, width=self.img_size),
                    ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item_id = self.items[index]
        image_path = os.path.join(self.data_path, self.phase, "image", f"{item_id}.jpg")
        mask_path = os.path.join(self.data_path, self.phase, "mask", f"{item_id}.png")

        # Загружаем как numpy-массивы
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        # Применяем аугментации / обрезку + ToTensor
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        # Если mask — numpy.ndarray, приводим к torch.LongTensor
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.astype(np.int64))
        else:
            mask = mask.long()

        if self.phase == "train":
            assert image.shape == (3, self.img_size, self.img_size)
            assert mask.shape == (self.img_size, self.img_size)

        return image, mask
