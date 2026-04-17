"""
model_2d.py — LipReadingTransformer с пофреймовым 2D CNN backbone.

Отличия от model.py (3D CNN версии):
  - 3D CNN frontend полностью убран
  - Вместо него: глубокий 2D CNN, применяемый к каждому кадру независимо
    (веса разделяются по временному измерению)
  - Активация: GELU вместо ReLU — лучший градиентный поток при малом числе каналов
  - GELU также в TransformerEncoderLayer и TransformerDecoderLayer
  - Нет интерполяции padding-маски (T не меняется после CNN)

Конфиг: model_config_v6.json (нет секции cnn_frontend, только cnn_backbone)
"""

import torch
import torch.nn as nn
import math
import json
from pathlib import Path


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_tuple(x):
    return tuple(x) if isinstance(x, list) else x


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


class VisualEncoder2D(nn.Module):
    """
    Пофреймовый 2D CNN + Transformer Encoder.

    Вход:  [B, T, 3, H, W]  — T кадров на батч
    Выход: [B, T, d_model]  — контекстные векторы (memory)

    CNN применяется к каждому кадру одинаково (shared weights по T).
    Благодаря этому число параметров CNN не зависит от длины видео.
    """

    def __init__(self, encoder_cfg, d_model, nhead, num_layers,
                 dim_feedforward, dropout, max_frames):
        super().__init__()

        # ── Per-frame 2D CNN (GELU активации) ──
        backbone_layers = []
        for layer in encoder_cfg["cnn_backbone"]:
            backbone_layers += [
                nn.Conv2d(
                    in_channels=layer["in_channels"],
                    out_channels=layer["out_channels"],
                    kernel_size=_as_tuple(layer["kernel_size"]),
                    stride=_as_tuple(layer["stride"]),
                    padding=_as_tuple(layer["padding"]),
                    bias=False,
                ),
                nn.BatchNorm2d(layer["out_channels"]),
                nn.GELU(),
            ]
        backbone_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.cnn_backbone = nn.Sequential(*backbone_layers)

        last_channels = encoder_cfg["cnn_backbone"][-1]["out_channels"]
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

        # Применяем 2D CNN покадрово (общие веса)
        x = x.view(B * T, C, H, W)
        x = self.cnn_backbone(x)   # [B*T, last_channels, 1, 1]
        x = x.view(B, T, -1)       # [B, T, last_channels]

        x = self.fc_proj(x)         # [B, T, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)

        memory = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # T не меняется — маска паддинга остаётся корректной
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
        return self.fc_out(out)  # [B, T_tgt, vocab_size]


class LipReadingTransformer2D(nn.Module):
    """
    LipReadingTransformer с пофреймовым 2D CNN энкодером.
    Конфиг: model_config_v6.json
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

        self.encoder = VisualEncoder2D(
            config["encoder"], d_model, nhead, num_layers,
            dim_feedforward, dropout, max_frames,
        )
        self.decoder = TextDecoder(
            vocab_size, d_model, nhead, num_layers,
            dim_feedforward, dropout, max_tokens,
        )

    def forward(self, src_video, tgt_tokens, tgt_padding_mask=None, src_padding_mask=None):
        memory, memory_padding_mask = self.encoder(src_video, src_key_padding_mask=src_padding_mask)
        output = self.decoder(tgt_tokens, memory, tgt_padding_mask, memory_padding_mask=memory_padding_mask)
        return output
