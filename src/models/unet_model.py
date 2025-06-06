import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class UpBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class BottleNeck(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class UNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Encoder
        self.enc_conv1 = DownBlock(3, 64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv2 = DownBlock(64, 128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv3 = DownBlock(128, 256)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv4 = DownBlock(256, 512)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = BottleNeck(512, 1024)

        # Decoder
        self.transpose_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv1 = UpBlock(1024, 512)

        self.transpose_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv2 = UpBlock(512, 256)

        self.transpose_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv3 = UpBlock(256, 128)

        self.transpose_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv4 = UpBlock(128, 64)

        # Final conv
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.clone().float()
        B, C, H_in, W_in = x.shape

        # Если изображение не 256×256 (или любое другое), масштабируем к 256 при помощи interpolate
        if H_in != 256 or W_in != 256:
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            reshaped = True
        else:
            reshaped = False

        # Encoder
        conv1 = self.enc_conv1(x)  # [B, 64, 256, 256]
        x = self.max_pool1(conv1)  # [B, 64, 128, 128]

        conv2 = self.enc_conv2(x)  # [B, 128, 128, 128]
        x = self.max_pool2(conv2)  # [B, 128, 64, 64]

        conv3 = self.enc_conv3(x)  # [B, 256, 64, 64]
        x = self.max_pool3(conv3)  # [B, 256, 32, 32]

        conv4 = self.enc_conv4(x)  # [B, 512, 32, 32]
        x = self.max_pool4(conv4)  # [B, 512, 16, 16]

        # Bottleneck
        x = self.bottleneck(x)  # [B, 1024, 16, 16]

        # Decoder
        x = self.transpose_conv1(x)  # [B, 512, 32, 32]
        x = torch.cat([x, conv4], dim=1)
        x = self.dec_conv1(x)  # [B, 512, 32, 32]

        x = self.transpose_conv2(x)  # [B, 256, 64, 64]
        x = torch.cat([x, conv3], dim=1)
        x = self.dec_conv2(x)  # [B, 256, 64, 64]

        x = self.transpose_conv3(x)  # [B, 128, 128, 128]
        x = torch.cat([x, conv2], dim=1)
        x = self.dec_conv3(x)  # [B, 128, 128, 128]

        x = self.transpose_conv4(x)  # [B, 64, 256, 256]
        x = torch.cat([x, conv1], dim=1)
        x = self.dec_conv4(x)  # [B, 64, 256, 256]

        logits = self.final_conv(x)  # [B, num_classes, 256, 256]

        if reshaped:
            logits = F.interpolate(
                logits, size=(H_in, W_in), mode="bilinear", align_corners=False
            )

        assert logits.shape == (
            B,
            self.num_classes,
            H_in,
            W_in,
        ), "Неправильная форма выходного тензора"
        return logits
