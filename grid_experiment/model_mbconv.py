"""
model_mbconv.py — LipReadingTransformer с MBConv backbone (EfficientNet-style).

Отличия от model_2d.py:
  - CNN backbone заменён на MBConv блоки из EfficientNet_lib.py
  - Stem (Conv2d 3→32) + MBConv блоки + AdaptiveAvgPool2d
  - SE attention для channel-wise перевзвешивания
  - Skip connections для улучшения градиентного потока
  - DropPath (Stochastic Depth) для регуляризации
  - SiLU/Swish активация в CNN, GELU в Transformer
  - Decoder и Transformer Encoder идентичны model_2d.py
"""

import torch
import torch.nn as nn
import math
import json
from pathlib import Path

from EfficientNet_lib import MBConvBlock


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class VisualEncoderMBConv(nn.Module):
    """
    Пофреймовый MBConv backbone + Transformer Encoder.

    Вход:  [B, T, 3, H, W]
    Выход: [B, T, d_model]

    Структура CNN:
      Stem: Conv2d(3 → stem_channels, k=3, s=2) + BN + SiLU
      MBConv блоки (из конфига): depthwise separable + SE + skip + DropPath
      AdaptiveAvgPool2d(1,1)
      Linear(last_channels → d_model)
    """

    def __init__(self, encoder_cfg, d_model, nhead, num_layers,
                 dim_feedforward, dropout, max_frames):
        super().__init__()

        stem_channels = encoder_cfg.get("stem_channels", 32)
        drop_path_rate = encoder_cfg.get("drop_path_rate", 0.1)
        mbconv_cfgs = encoder_cfg["mbconv_blocks"]

        # ── Stem: обычная свёртка для первичного извлечения признаков ──
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
        )

        # ── MBConv блоки ──
        # Линейно нарастающий drop_path_rate (как в EfficientNet)
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

        # Проекция последнего выхода → d_model
        last_channels = mbconv_cfgs[-1]["out_channels"]
        self.fc_proj = nn.Linear(last_channels, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_len=max(max_frames * 2, 128))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        # x: [B, T, 3, H, W]
        B, T, C, H, W = x.size()

        # Покадрово через CNN (shared weights)
        x = x.view(B * T, C, H, W)
        x = self.stem(x)
        x = self.mbconv_blocks(x)
        x = self.pool(x)            # [B*T, last_channels, 1, 1]
        x = x.view(B, T, -1)        # [B, T, last_channels]

        x = self.fc_proj(x)          # [B, T, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)

        memory = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 dim_feedforward, dropout, max_tokens):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_tokens)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_padding_mask=None, memory_padding_mask=None):
        B, T_tgt = tgt.size()
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T_tgt).to(tgt.device)

        out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.fc_out(out)


class LipReadingTransformerMBConv(nn.Module):
    """
    LipReadingTransformer с MBConv визуальным энкодером.
    Конфиг: model_config_mbconv.json
    """

    def __init__(self, config):
        super().__init__()
        if isinstance(config, (str, Path)):
            config = load_config(config)
        self.config = config

        vocab_size      = config["vocab_size"]
        d_model         = config["d_model"]
        nhead           = config["nhead"]
        num_layers      = config["num_layers"]
        dim_feedforward = config.get("dim_feedforward", d_model * 4)
        dropout         = config["dropout"]
        max_frames      = config["max_frames"]
        max_tokens      = config["max_tokens"]

        # Поддержка раздельного числа слоёв encoder/decoder
        num_encoder_layers = config.get("num_encoder_layers", num_layers)
        num_decoder_layers = config.get("num_decoder_layers", num_layers)

        self.encoder = VisualEncoderMBConv(
            config["encoder"], d_model, nhead, num_encoder_layers,
            dim_feedforward, dropout, max_frames,
        )
        self.decoder = TextDecoder(
            vocab_size, d_model, nhead, num_decoder_layers,
            dim_feedforward, dropout, max_tokens,
        )

    def forward(self, src_video, tgt_tokens, tgt_padding_mask=None, src_padding_mask=None):
        memory, memory_padding_mask = self.encoder(src_video, src_key_padding_mask=src_padding_mask)
        output = self.decoder(tgt_tokens, memory, tgt_padding_mask, memory_padding_mask=memory_padding_mask)
        return output
