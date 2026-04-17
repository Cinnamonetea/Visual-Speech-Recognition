import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from pathlib import Path


def load_config(path):
    """Загрузить JSON-конфиг модели."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_tuple(x):
    """В JSON нет кортежей — только списки. Conv-слои принимают и int, и tuple."""
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
        # x shape: (Batch, Seq_Len, d_model)
        return x + self.pe[:, :x.size(1)]


class VisualEncoder(nn.Module):
    def __init__(self, encoder_cfg, d_model, nhead, num_layers,
                 dim_feedforward, dropout, max_frames):
        super().__init__()

        # ── CNN Frontend (Conv3d + MaxPool3d) ──
        fe = encoder_cfg["cnn_frontend"]
        conv3d_cfg = fe["conv3d"]
        pool_cfg = fe["maxpool3d"]

        self.cnn_frontend = nn.Sequential(
            nn.Conv3d(
                in_channels=conv3d_cfg["in_channels"],
                out_channels=conv3d_cfg["out_channels"],
                kernel_size=_as_tuple(conv3d_cfg["kernel_size"]),
                stride=_as_tuple(conv3d_cfg["stride"]),
                padding=_as_tuple(conv3d_cfg["padding"]),
                bias=False,
            ),
            nn.BatchNorm3d(conv3d_cfg["out_channels"]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=_as_tuple(pool_cfg["kernel_size"]),
                stride=_as_tuple(pool_cfg["stride"]),
                padding=_as_tuple(pool_cfg["padding"]),
            ),
        )

        # ── CNN Backbone (список Conv2d-слоёв) ──
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
                nn.ReLU(inplace=True),
            ]
        backbone_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.cnn_backbone = nn.Sequential(*backbone_layers)

        # Размерность последнего выхода backbone -> проекция в d_model
        last_channels = encoder_cfg["cnn_backbone"][-1]["out_channels"]
        self.fc_proj = nn.Linear(last_channels, d_model)
        # Запас в PE на случай, если свёртки по времени изменят длину (padding/stride)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max(max_frames * 2, 128))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        # x: [B, T, 3, H, W] -> Conv3d ждёт [B, 3, T, H, W]
        x = x.transpose(1, 2)

        x = self.cnn_frontend(x)

        # Подготовка для 2D Backbone
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

        x = self.cnn_backbone(x)
        x = x.view(B, T, -1)

        x = self.fc_proj(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Если Conv3d изменил длину времени, приводим маску паддинга к новому T
        if src_key_padding_mask is not None and src_key_padding_mask.size(1) != T:
            mask_f = src_key_padding_mask.float().unsqueeze(1)        # [B, 1, T_in]
            mask_f = F.interpolate(mask_f, size=T, mode="nearest")     # [B, 1, T]
            src_key_padding_mask = mask_f.squeeze(1).bool()

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
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_padding_mask=None, memory_padding_mask=None):
        # tgt: (Batch, Seq_Len) - токены
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

        logits = self.fc_out(out)  # [Batch, Seq_Len, Vocab_Size]
        return logits


class LipReadingTransformer(nn.Module):
    """
    Модель конструируется из JSON-конфига (dict или путь к файлу).
    См. model_config.json для структуры.
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

        self.encoder = VisualEncoder(
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
