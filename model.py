"""CNN模型定义模块"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 卷积步长
        """
        super(ResidualBlock, self).__init__()
        
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 残差连接（shortcut）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 当维度不匹配时，使用1x1卷积调整维度
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [B, in_channels, H, W]
        Returns:
            输出张量 [B, out_channels, H', W']
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # 残差连接
        out = self.relu(out)
        
        return out


class CIFAR10CNN(nn.Module):
    """CIFAR-10 CNN分类器"""
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        """
        Args:
            num_classes: 分类类别数
            dropout_rate: Dropout比率
        """
        super(CIFAR10CNN, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差层
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout和分类器
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int) -> nn.Sequential:
        """构建残差层"""
        layers = []
        
        # 第一个块可能需要下采样
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # 后续块保持维度不变
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [B, 3, 32, 32]
        Returns:
            分类logits [B, num_classes]
        """
        # 初始卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 残差层
        out = self.layer1(out)  # [B, 64, 32, 32]
        out = self.layer2(out)  # [B, 128, 16, 16]
        out = self.layer3(out)  # [B, 256, 8, 8]
        out = self.layer4(out)  # [B, 512, 4, 4]
        
        # 全局平均池化
        out = self.avgpool(out)  # [B, 512, 1, 1]
        out = torch.flatten(out, 1)  # [B, 512]
        
        # 分类
        out = self.dropout(out)
        out = self.fc(out)  # [B, num_classes]
        
        return out
