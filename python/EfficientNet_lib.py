import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

# ====================== Вспомогательные компоненты ======================

class SqueezeExcitation(nn.Module):
    """
    Блок Squeeze-and-Excitation (SE) для перевзвешивания каналов
    """
    def __init__(self, in_channels: int, reduced_dim: int):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),  # Swish активация (SiLU)
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Инициализация весов блока Squeeze-and-Excitation
        """
        # Слой сжатия (редукция каналов) - Conv2d с активацией SiLU
        # se[1] - это первый Conv2d слой (in_channels -> reduced_dim)
        nn.init.xavier_normal_(
            self.se[1].weight,
            gain=nn.init.calculate_gain('relu')  # Для активации SiLU
        )
        if self.se[1].bias is not None:
            nn.init.constant_(self.se[1].bias, 0)
        
        # Слой расширения (экспансия каналов) - Conv2d с активацией Sigmoid
        # se[3] - это второй Conv2d слой (reduced_dim -> in_channels)
        nn.init.xavier_normal_(
            self.se[3].weight,
            gain=1.0  # Для активации Sigmoid (масштаб [0, 1])
        )
        if self.se[3].bias is not None:
            nn.init.constant_(self.se[3].bias, 0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class DropPath(nn.Module):
    """
    Stochastic Depth (DropPath) для регуляризации
    """
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        
        # Сохраняем только (1 - drop_prob) образцов
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Бинаризация
        output = x.div(keep_prob) * random_tensor
        return output


# ====================== Основной блок MBConv ======================

class MBConvBlock(nn.Module):
    """
    MBConv блок из EfficientNet (Mobile Inverted Bottleneck Conv)
    
    Args:
        in_channels: количество входных каналов
        out_channels: количество выходных каналов
        kernel_size: размер ядра (3 или 5)
        stride: шаг свертки (1 или 2)
        expand_ratio: коэффициент расширения каналов (обычно 1, 4, 6)
        se_ratio: коэффициент сжатия для SE блока (0.25 для EfficientNet)
        drop_path_rate: вероятность DropPath (Stochastic Depth)
        use_se: использовать ли Squeeze-and-Excitation
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 4,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.,
        use_se: bool = True,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super().__init__()
        
        # Сохраняем параметры для инициализации
        self.init_params = {
            'expand_gain': nn.init.calculate_gain('relu'),
            'depthwise_gain': nn.init.calculate_gain('relu'),
            'se_reduce_gain': nn.init.calculate_gain('relu'),
            'se_expand_gain': 1.0,  # после sigmoid
            'project_gain': 1.0,
            'skip_gain': 1.0
        }
        
        # Настройки по умолчанию
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # Swish активация
        
        # Параметры блока
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.has_skip = (stride == 1 and in_channels == out_channels)
        
        # Вычисление скрытой размерности (расширение)
        hidden_dim = int(in_channels * expand_ratio)
        
        # Фаза расширения (если expand_ratio > 1)
        self.expand = None
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                norm_layer(hidden_dim),
                activation_layer(inplace=True)
            )
        
        # Depthwise convolution (свертка по глубине)
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                hidden_dim if expand_ratio != 1 else in_channels,
                hidden_dim if expand_ratio != 1 else in_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=hidden_dim if expand_ratio != 1 else in_channels,
                bias=False
            ),
            norm_layer(hidden_dim if expand_ratio != 1 else in_channels),
            activation_layer(inplace=True)
        )
        
        # Squeeze-and-Excitation блок
        self.use_se = use_se
        if use_se:
            reduced_dim = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcitation(
                hidden_dim if expand_ratio != 1 else in_channels,
                reduced_dim
            )
        
        # Фаза сжатия (проекция)
        self.project = nn.Sequential(
            nn.Conv2d(
                hidden_dim if expand_ratio != 1 else in_channels,
                out_channels,
                1,
                bias=False
            ),
            norm_layer(out_channels)
        )
        
        # DropPath для регуляризации
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # Проекция для skip connection (если размеры не совпадают)
        self.skip_proj = None
        if stride != 1 or in_channels != out_channels:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                norm_layer(out_channels)
            )
    
    def _initialize_weights(self):
        """
        Метод инициализации весов блока MBConv
        Вызывается после создания блока для правильной инициализации
        """
        # Инициализация слоя расширения (если есть)
        if self.expand is not None:
            # expand[0] - Conv2d слой
            nn.init.xavier_normal_(
                self.expand[0].weight, 
                gain=self.init_params['expand_gain']
            )
            # BatchNorm инициализируется отдельно
        
        # Инициализация depthwise свертки
        nn.init.xavier_normal_(
            self.depthwise[0].weight,
            gain=self.init_params['depthwise_gain']
        )

        # Инициализация проекционного слоя
        nn.init.xavier_normal_(
            self.project[0].weight,
            gain=self.init_params['project_gain']
        )
        
        # Инициализация skip connection проекции (если есть)
        if self.skip_proj is not None:
            nn.init.xavier_normal_(
                self.skip_proj[0].weight,
                gain=self.init_params['skip_gain']
            )
        
        # Инициализация BatchNorm слоев
        self._init_batchnorm()
    
    def _init_batchnorm(self):
        """
        Инициализация BatchNorm слоев в блоке
        """
        # Инициализация BatchNorm в expand
        if self.expand is not None:
            nn.init.constant_(self.expand[1].weight, 1)
            nn.init.constant_(self.expand[1].bias, 0)
        
        # Инициализация BatchNorm в depthwise
        nn.init.constant_(self.depthwise[1].weight, 1)
        nn.init.constant_(self.depthwise[1].bias, 0)
        
        # Инициализация BatchNorm в project
        nn.init.constant_(self.project[1].weight, 1)
        nn.init.constant_(self.project[1].bias, 0)
        
        # Инициализация BatchNorm в skip_proj (если есть)
        if self.skip_proj is not None:
            nn.init.constant_(self.skip_proj[1].weight, 1)
            nn.init.constant_(self.skip_proj[1].bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Фаза расширения
        if self.expand is not None:
            x = self.expand(x)
        
        # Depthwise convolution
        x = self.depthwise(x)
        
        # Squeeze-and-Excitation
        if self.use_se:
            x = self.se(x)
        
        # Фаза сжатия
        x = self.project(x)
        
        # Skip connection
        if self.has_skip:
            x = self.drop_path(x) + identity
        elif self.skip_proj is not None:
            x = x + self.skip_proj(identity)
        
        return x


# ====================== Примеры использования ======================

class CustomEfficientNetLike(nn.Module):
    """
    Пример кастомной CNN с использованием MBConv блоков
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Начальные слои (как в EfficientNet)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Последовательность MBConv блоков (упрощенная версия EfficientNet-B0)
        self.blocks = nn.Sequential(
            # Stage 1: 32 -> 16
            MBConvBlock(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            
            # Stage 2: 16 -> 24
            MBConvBlock(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            MBConvBlock(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            
            # Stage 3: 24 -> 40
            MBConvBlock(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            MBConvBlock(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            
            # Stage 4: 40 -> 80
            MBConvBlock(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            MBConvBlock(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            MBConvBlock(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            
            # Stage 5: 80 -> 112
            MBConvBlock(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConvBlock(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConvBlock(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            
            # Stage 6: 112 -> 192
            MBConvBlock(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            
            # Stage 7: 192 -> 320
            MBConvBlock(192, 320, kernel_size=3, stride=1, expand_ratio=6),
        )
        
        # Финальные слои
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class MBConvConfig:
    """
    Конфигурация для построения EfficientNet-подобной архитектуры
    """
    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2):
        # Базовая конфигурация EfficientNet-B0
        self.base_config = [
            # (expand_ratio, channels, repeats, stride, kernel_size)
            (1, 16, 1, 1, 3),   # stage 1
            (6, 24, 2, 2, 3),   # stage 2
            (6, 40, 2, 2, 5),   # stage 3
            (6, 80, 3, 2, 3),   # stage 4
            (6, 112, 3, 1, 5),  # stage 5
            (6, 192, 4, 2, 5),  # stage 6
            (6, 320, 1, 1, 3),  # stage 7
        ]
        
        # Применяем коэффициенты масштабирования
        self.config = []
        for expand_ratio, channels, repeats, stride, kernel_size in self.base_config:
            # Масштабирование ширины (каналов)
            channels = self._round_channels(channels * width_mult)
            repeats = int(repeats * depth_mult)
            self.config.append((expand_ratio, channels, repeats, stride, kernel_size))
        
        self.dropout_rate = dropout_rate
    
    def _round_channels(self, channels: float, divisor: int = 8) -> int:
        """Округление количества каналов до ближайшего кратного divisor"""
        rounded = max(divisor, int(channels + divisor / 2) // divisor * divisor)
        if rounded < 0.9 * channels:
            rounded += divisor
        return rounded


def build_efficientnet_from_config(config: MBConvConfig, in_channels=3, num_classes=1000):
    """
    Строит EfficientNet-подобную модель из конфигурации
    """
    layers = []
    
    # Stem
    out_channels = config._round_channels(32)
    layers.append(nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True)
    ))
    
    in_channels = out_channels
    
    # MBConv блоки
    drop_path_rates = torch.linspace(0, 0.2, sum(c[2] for c in config.config)).tolist()
    idx = 0
    
    for expand_ratio, channels, repeats, stride, kernel_size in config.config:
        for i in range(repeats):
            # Только первый блок в каждом этапе имеет stride > 1
            block_stride = stride if i == 0 else 1
            drop_path_rate = drop_path_rates[idx]
            
            layers.append(MBConvBlock(
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=block_stride,
                expand_ratio=expand_ratio,
                se_ratio=0.25,
                drop_path_rate=drop_path_rate,
                use_se=True
            ))
            
            in_channels = channels
            idx += 1
    
    # Head
    final_channels = config._round_channels(1280)
    layers.append(nn.Sequential(
        nn.Conv2d(in_channels, final_channels, 1, bias=False),
        nn.BatchNorm2d(final_channels),
        nn.SiLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(config.dropout_rate),
        nn.Linear(final_channels, num_classes)
    ))
    
    return nn.Sequential(*layers)


# ====================== Тестирование ======================

if __name__ == "__main__":
    # Тест отдельного блока
    print("Тестирование MBConv блока...")
    
    # Создаем тестовый тензор (batch_size=4, channels=32, H=64, W=64)
    x = torch.randn(4, 32, 64, 64)
    
    # Тестируем разные конфигурации блока
    block1 = MBConvBlock(32, 16, kernel_size=3, stride=1, expand_ratio=1)
    block2 = MBConvBlock(32, 64, kernel_size=5, stride=2, expand_ratio=6)
    
    y1 = block1(x)
    y2 = block2(x)
    
    print(f"Вход: {x.shape}")
    print(f"Выход блока1 (stride=1, expand=1): {y1.shape}")
    print(f"Выход блока2 (stride=2, expand=6): {y2.shape}")
    
    # Тест кастомной модели
    print("\nТестирование кастомной модели...")
    model = CustomEfficientNetLike(num_classes=10)
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    
    # Прямой проход
    test_input = torch.randn(2, 3, 224, 224)
    output = model(test_input)
    print(f"Вход модели: {test_input.shape}")
    print(f"Выход модели: {output.shape}")
    
    # Тест сборки модели из конфигурации
    print("\nТестирование сборки из конфигурации...")
    config = MBConvConfig(width_mult=1.0, depth_mult=1.0)
    efficientnet_model = build_efficientnet_from_config(config, num_classes=10)
    
    total_params = sum(p.numel() for p in efficientnet_model.parameters())
    print(f"EfficientNet-B0 параметров: {total_params:,}")
    
    # Проверка прямого прохода
    test_input = torch.randn(2, 3, 224, 224)
    output = efficientnet_model(test_input)
    print(f"Выход EfficientNet: {output.shape}")