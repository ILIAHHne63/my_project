# src/utils/plot_callback.py

import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl


class SaveMetricsPlotCallback(pl.Callback):
    """
    Callback, который по окончании тренировки строит графики
    train_loss, val_loss, val_IoU и сохраняет их в указанную папку.
    """

    def __init__(self, save_dir: str):
        """
        :param save_dir: папка, куда сохранить PNG-файлы
        """
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.val_ious = []

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        На каждом конце валидации собираем train_loss, val_loss, val_IoU.
        Предполагается, что LightningModule логирует:
          self.log("train_loss", ...),
          self.log("val_loss", ...),
          self.log("val_IoU", ...)
        """
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.train_losses.append(metrics["train_loss"].cpu().item())
        if "val_loss" in metrics:
            self.val_losses.append(metrics["val_loss"].cpu().item())
        if "val_mIoU" in metrics:
            self.val_ious.append(metrics["val_mIoU"].cpu().item())

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        В конце тренировки строим и сохраняем три графика.
        """
        # 1) train_loss vs epoch
        if self.train_losses:
            plt.figure(figsize=(6, 4))
            plt.plot(self.train_losses, label="train_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Train Loss")
            plt.legend()
            path = os.path.join(self.save_dir, "train_loss.png")
            plt.savefig(path)
            plt.close()

        # 2) val_loss vs epoch
        if self.val_losses:
            plt.figure(figsize=(6, 4))
            plt.plot(self.val_losses, label="val_loss", color="orange")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Validation Loss")
            plt.legend()
            path = os.path.join(self.save_dir, "val_loss.png")
            plt.savefig(path)
            plt.close()

        # 3) val_IoU vs epoch
        if self.val_ious:
            plt.figure(figsize=(6, 4))
            plt.plot(self.val_ious, label="val_mIoU", color="green")
            plt.xlabel("Epoch")
            plt.ylabel("mIoU")
            plt.title("Validation mIoU")
            plt.legend()
            path = os.path.join(self.save_dir, "val_mIoU.png")
            plt.savefig(path)
            plt.close()
