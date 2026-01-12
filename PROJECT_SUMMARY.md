# CIFAR-10 CNN 项目总结

## 项目概述

本项目实现了一个基于ResNet风格的CNN网络，用于CIFAR-10图像分类任务，支持8张NVIDIA 3090 GPU的分布式训练。

## 核心文件

### Python模块
- `model.py` - CNN模型定义（ResidualBlock + CIFAR10CNN）
- `data_loader.py` - CIFAR-10数据加载和预处理
- `trainer.py` - 分布式训练器（支持DDP和AMP）
- `train.py` - 主训练脚本
- `visualize.py` - 训练曲线可视化

### 配置文件
- `requirements.txt` - Python依赖包列表

### 启动脚本
- `setup_env.sh` - Conda环境创建脚本
- `train_distributed.sh` - 8 GPU分布式训练启动脚本
- `train_single_gpu.sh` - 单GPU训练脚本

### 文档
- `README.md` - 完整的项目文档

## 快速开始

```bash
# 1. 激活环境（已创建）
conda activate cifar10_cnn

# 2. 开始训练
bash train_distributed.sh  # 8 GPU训练
# 或
bash train_single_gpu.sh   # 单GPU训练
```

## 模型架构

- **ResNet风格**: 4层残差网络（64→128→256→512通道）
- **每层**: 2个残差块
- **特性**: BatchNorm + Dropout(0.5) + 残差连接
- **参数量**: 约11M

## 训练配置

| 配置项 | 值 |
|--------|-----|
| Batch Size | 128 × 8 = 1024 |
| Learning Rate | 0.1 (余弦退火) |
| Optimizer | SGD (momentum=0.9, weight_decay=5e-4) |
| Epochs | 200 |
| 混合精度 | 启用 |

## 训练输出

训练完成后在 `./outputs` 目录生成：
- `cifar10_cnn_best.pth` - 最佳模型
- `cifar10_cnn_latest.pth` - 最新模型
- `training_curves.png` - 训练曲线图

## 进度显示

训练过程中会显示：
- 训练配置信息表格
- 实时进度条（显示loss/accuracy）
- 每个epoch的详细信息
- 🌟 标记最佳模型

## 数据集

CIFAR-10数据集位于 `./cifar-10-batches-py`：
- 训练集: 50,000张图像
- 测试集: 10,000张图像
- 类别数: 10

## 性能优化

- ✅ 分布式数据并行（DDP）
- ✅ 混合精度训练（AMP）
- ✅ 数据增强（随机裁剪、水平翻转）
- ✅ 学习率余弦退火
- ✅ 残差连接

## 依赖

- Python 3.10
- PyTorch 2.9.1
- torchvision 0.24.1
- numpy, matplotlib, tqdm

## 环境

- Conda环境: `cifar10_cnn`
- GPU: 8 × NVIDIA 3090
- CUDA: 支持

## 项目特点

1. **完整实现**: 从数据加载到模型训练的完整流程
2. **分布式支持**: 原生支持多GPU训练
3. **美观输出**: 实时进度条和格式化信息
4. **易于使用**: 一键启动脚本
5. **高性能**: 混合精度训练加速

## 预期性能

在CIFAR-10数据集上，经过200个epoch的训练，预期可达到：
- 测试准确率: ~85-90%
- 训练时间: 约2-3小时（8×3090）

## 注意事项

1. 确保数据集在正确位置（`./cifar-10-batches-py`）
2. 确保有足够的GPU显存（建议每张GPU至少8GB）
3. 训练过程中会自动保存最佳模型
4. 可通过命令行参数调整超参数

## 后续改进方向

- 尝试更深的网络结构
- 添加更多数据增强策略
- 实验不同的学习率调度策略
- 添加模型集成
- 实现测试时增强（TTA）
