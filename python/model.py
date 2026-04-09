import torch
import torch.nn as nn
import math
import numpy as np

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
    def __init__(self, d_model=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()

        # Вход: [B, 3, 40, 96, 64] (Batch, Channels, Time, Height, Width)
        self.cnn_frontend = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        self.cnn_backbone = nn.Sequential(
            # [B, 64, 40, 24, 16]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # [B, 128, 40, 12, 18]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Проекция в размерность трансформера
        self.fc_proj = nn.Linear(256, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=40)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        # x: [B, 40, 3, 96, 64] -> меняем для Conv3d на [B, 3, 40, 96, 64]
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

        memory = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        return memory

class TextDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=12)

        # Блок Nx слоев декодера (Masked Multi-Head + Cross-Attention + FF)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_padding_mask=None, memory_padding_mask=None):
        # tgt: (Batch, Seq_Len) - токены
        # memory: выход энкодера

        B, T_tgt = tgt.size()
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T_tgt).to(tgt.device)

        # Декодер принимает текущие токены и "память" из энкодера
        out = self.transformer_decoder(
            tgt = tgt_emb,
            memory = memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask
        )

        logits = self.fc_out(out)  # [Batch, 12, Vocab_Size]
        return logits

class LipReadingTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()
        self.encoder = VisualEncoder(d_model, nhead, num_layers, dropout)
        self.decoder = TextDecoder(vocab_size, d_model, nhead, num_layers, dropout)

    def forward(self, src_video, tgt_tokens, tgt_padding_mask=None, src_padding_mask=None):
        # Энкодер: Видео -> Контекстные векторы
        memory = self.encoder(src_video, src_key_padding_mask=src_padding_mask)

        # Декодер: Токены + Память энкодера -> Прогнозы
        output = self.decoder(tgt_tokens, memory, tgt_padding_mask, memory_padding_mask=src_padding_mask)
        return output