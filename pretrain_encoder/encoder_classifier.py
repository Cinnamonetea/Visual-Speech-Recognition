"""
encoder_classifier.py — Предобучение визуального CNN энкодера
через классификацию слов (BPE-токенов) по кадрам губ.

Архитектура:
  MBConv CNN (из model_mbconv.py) → AdaptiveAvgPool → Linear(num_classes)

Подход:
  Каждый pkl-файл содержит frames и input_ids.
  Кадры равномерно распределяются по токенам.
  Для каждого окна из WINDOW_SIZE кадров определяется «центральный» токен.
  CNN обучается предсказывать этот токен.

После обучения сохраняются веса stem + mbconv_blocks + pool — они
загружаются в VisualEncoderMBConv для fine-tuning в полной модели.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python_mbconv"))
from EfficientNet_lib import MBConvBlock


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class VisualEncoderClassifier(nn.Module):
    """
    MBConv CNN энкодер + классификационная голова.

    Вход:  [B, WINDOW_SIZE, 3, H, W]  — окно из нескольких кадров
    Выход: [B, num_classes]            — логиты по классам (токенам)

    CNN обрабатывает каждый кадр, затем усредняет по времени.
    """

    def __init__(self, encoder_cfg, num_classes, dropout=0.5):
        super().__init__()

        stem_channels = encoder_cfg.get("stem_channels", 24)
        drop_path_rate = encoder_cfg.get("drop_path_rate", 0.15)
        mbconv_cfgs = encoder_cfg["mbconv_blocks"]

        # ── CNN backbone (идентичен VisualEncoderMBConv) ──
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
        )

        num_blocks = len(mbconv_cfgs)
        dp_rates = [drop_path_rate * i / max(num_blocks - 1, 1) for i in range(num_blocks)]

        blocks = []
        for i, cfg in enumerate(mbconv_cfgs):
            block = MBConvBlock(
                in_channels=cfg["in_channels"],
                out_channels=cfg["out_channels"],
                kernel_size=cfg.get("kernel_size", 3),
                stride=cfg.get("stride", 1),
                expand_ratio=cfg.get("expand_ratio", 4),
                se_ratio=cfg.get("se_ratio", 0.25),
                drop_path_rate=dp_rates[i],
                use_se=cfg.get("use_se", True),
            )
            blocks.append(block)
        self.mbconv_blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Классификационная голова ──
        last_channels = mbconv_cfgs[-1]["out_channels"]
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_channels, num_classes),
        )

    def forward_cnn(self, x):
        """Прогнать один кадр [B, 3, H, W] через CNN → [B, C]."""
        x = self.stem(x)
        x = self.mbconv_blocks(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        """
        x: [B, T, 3, H, W] — окно из T кадров.
        Возвращает: [B, num_classes] — логиты.
        """
        B, T, C, H, W = x.size()

        # Покадрово через CNN
        x = x.view(B * T, C, H, W)
        features = self.forward_cnn(x)  # [B*T, last_channels]
        features = features.view(B, T, -1)  # [B, T, last_channels]

        # Усреднение по времени
        pooled = features.mean(dim=1)  # [B, last_channels]

        return self.classifier(pooled)

    def get_cnn_state_dict(self):
        """Извлечь только веса CNN (stem + mbconv + pool) для загрузки в полную модель."""
        state = {}
        for name, param in self.named_parameters():
            if name.startswith(("stem.", "mbconv_blocks.", "pool.")):
                state[name] = param.data.clone()
        for name, buf in self.named_buffers():
            if name.startswith(("stem.", "mbconv_blocks.", "pool.")):
                state[name] = buf.clone()
        return state
