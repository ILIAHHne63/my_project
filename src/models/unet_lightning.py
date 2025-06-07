import pytorch_lightning as pl
import torch
from torch import nn

from .unet_model import UNet


def calc_val_data(preds: torch.Tensor, masks: torch.Tensor, num_classes: int):
    """
    Берёт логиты [B, C, H, W] и ground-truth [B, H, W],
    возвращает intersection, union, target
    """
    preds = torch.argmax(preds, dim=1)  # [B, H, W]
    B, H, W = preds.shape

    intersection = torch.zeros(B, num_classes, H, W, device=preds.device)
    union = torch.zeros(B, num_classes, H, W, device=preds.device)
    target = torch.zeros(B, num_classes, H, W, device=preds.device)

    for i in range(num_classes):
        pred_mask = (preds == i).float()
        gt_mask = (masks == i).float()
        intersection[:, i, :, :] = pred_mask * gt_mask
        union[:, i, :, :] = pred_mask + gt_mask - (pred_mask * gt_mask)
        target[:, i, :, :] = gt_mask

    return intersection, union, target


def calc_val_loss(
    intersection: torch.Tensor,
    union: torch.Tensor,
    target: torch.Tensor,
    num_batches: int,
    eps: float = 1e-7,
):
    """
    Принимает:
      intersection: [total_examples, C, H, W]
      union      : [total_examples, C, H, W]
      target     : [total_examples, C, H, W]
      num_batches: число обработанных мини-батчей на валидации
    Возвращает: (mean_iou, mean_recall, mean_acc)
    """
    # Суммируем по пространственным осям и по batch’ам
    intersection_sum = intersection.sum(dim=(0, 2, 3))  # [C]
    union_sum = union.sum(dim=(0, 2, 3))  # [C]
    target_sum = target.sum(dim=(0, 2, 3))  # [C]

    # iou и recall на класс [C]
    iou_per_class = (intersection_sum + eps) / (union_sum + eps)
    recall_per_class = (intersection_sum + eps) / (target_sum + eps)

    mean_iou = iou_per_class.mean().item()
    mean_recall = recall_per_class.mean().item()

    total_intersection = intersection_sum.sum()  # скаляр
    # получаем B × H × W × num_batches
    # Но нам нужно знать число пикселей:
    # total_pixels = (H * W) * (число примеров).
    # Мы знаем, что intersection.shape = [N, C, H, W],
    # где N = total_examples
    N, _, H, W = intersection.shape
    total_pixels = N * H * W
    mean_acc = ((total_intersection + eps) / (total_pixels + eps)).item()

    return mean_iou, mean_recall, mean_acc


class UNetLitModule(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        # Сохраняем все параметры в hyperparameters
        # (автоматически подтянет в логгер)
        self.save_hyperparameters(cfg)

        self.num_classes = cfg["model"]["num_classes"]
        self.lr = cfg["model"]["lr"]

        # Создаём модель и loss
        self.model = UNet(num_classes=self.num_classes)
        self.criterion = nn.CrossEntropyLoss()

        # Накопители для валидации
        self.val_intersection = []
        self.val_union = []
        self.val_target = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)

        # Логируем train_loss
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)

        # Логируем валидационный лосс (по эпохе)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Считаем матрицы intersection/union/target (на CPU)
        inter, uni, tar = calc_val_data(logits, masks, self.num_classes)
        self.val_intersection.append(inter.detach().cpu())
        self.val_union.append(uni.detach().cpu())
        self.val_target.append(tar.detach().cpu())

    def on_validation_epoch_end(self):
        # Склеиваем накопленные батчи
        intersection = torch.cat(
            self.val_intersection, dim=0
        )  # [total_examples, C, H, W]
        union = torch.cat(self.val_union, dim=0)
        target = torch.cat(self.val_target, dim=0)

        # Вычисляем метрики
        mean_iou, mean_recall, mean_acc = calc_val_loss(
            intersection, union, target, num_batches=len(self.val_intersection)
        )

        # Логируем mIoU, mRecall, mAcc
        self.log("val_mIoU", mean_iou, prog_bar=True)
        self.log("val_mRecall", mean_recall, prog_bar=True)
        self.log("val_mAcc", mean_acc, prog_bar=True)

        # Сбросим накопители
        self.val_intersection.clear()
        self.val_union.clear()
        self.val_target.clear()

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)

        # Логируем test loss & метрики
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        inter, uni, tar = calc_val_data(logits, masks, self.num_classes)
        mean_iou, mean_recall, mean_acc = calc_val_loss(
            inter.cpu(), uni.cpu(), tar.cpu(), num_batches=1
        )
        self.log("test_mIoU", mean_iou, prog_bar=True)
        self.log("test_mRecall", mean_recall, prog_bar=True)
        self.log("test_mAcc", mean_acc, prog_bar=True)
