# CIFAR-10 CNN图像分类器

基于ResNet风格的CNN网络，用于CIFAR-10数据集的图像分类任务，支持8张NVIDIA 3090 GPU的分布式训练。

## 项目结构

```
.
├── model.py                 # CNN模型定义（ResidualBlock + CIFAR10CNN）
├── data_loader.py           # CIFAR-10数据加载和预处理
├── trainer.py               # 分布式训练器
├── train.py                 # 主训练脚本
├── visualize.py             # 训练曲线可视化
├── requirements.txt         # 依赖包列表
├── setup_env.sh             # Conda环境创建脚本
├── train_distributed.sh     # 8 GPU分布式训练启动脚本
├── train_single_gpu.sh      # 单GPU训练脚本
└── README.md                # 本文件
```

## 环境配置

### 方法1: 使用脚本自动创建环境

```bash
# 创建并配置conda环境
bash setup_env.sh

# 激活环境
conda activate cifar10_cnn
```

### 方法2: 手动创建环境

```bash
# 创建conda环境
conda create -n cifar10_cnn python=3.10 -y

# 激活环境
conda activate cifar10_cnn

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 模型架构

### 网络结构

- **输入**: [B, 3, 32, 32] RGB图像
- **初始卷积**: 3 → 64通道
- **残差层1**: 64 → 64 (2个残差块)
- **残差层2**: 64 → 128 (2个残差块, stride=2下采样)
- **残差层3**: 128 → 256 (2个残差块, stride=2下采样)
- **残差层4**: 256 → 512 (2个残差块, stride=2下采样)
- **全局平均池化**: [B, 512, 4, 4] → [B, 512]
- **Dropout**: 0.5
- **全连接层**: 512 → 10
- **输出**: [B, 10] 分类logits

### 超参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Batch Size | 128 × 8 = 1024 | 每GPU 128，总共1024 |
| Learning Rate | 0.1 | 初始学习率 |
| Optimizer | SGD | 动量0.9，权重衰减5e-4 |
| LR Scheduler | CosineAnnealing | 余弦退火到0 |
| Epochs | 200 | 训练轮数 |
| Dropout | 0.5 | 分类器前的Dropout |
| 混合精度 | 启用 | 使用AMP加速训练 |

## 训练

### 训练进度显示

训练过程中会显示：
- 美观的训练配置信息表格
- 每个epoch的实时进度条（显示当前loss/accuracy）
- 每个epoch完成后的详细信息（Loss、Accuracy、Best Accuracy、Learning Rate）
- 🌟 标记表示新的最佳模型
- 训练完成后的总结信息

### 8 GPU分布式训练（推荐）

```bash
# 确保在cifar10_cnn环境中
conda activate cifar10_cnn

# 启动8 GPU训练
bash train_distributed.sh
```

### 单GPU训练

```bash
# 确保在cifar10_cnn环境中
conda activate cifar10_cnn

# 启动单GPU训练
bash train_single_gpu.sh
```

### 自定义训练参数

```bash
# 使用torchrun启动（8 GPU）
torchrun --nproc_per_node=8 train.py \
    --data_dir ./cifar-10-batches-py \
    --batch_size 128 \
    --epochs 200 \
    --lr 0.1 \
    --num_workers 4 \
    --save_dir ./outputs

# 单GPU训练
python train.py \
    --data_dir ./cifar-10-batches-py \
    --batch_size 128 \
    --epochs 200 \
    --lr 0.1 \
    --num_workers 4 \
    --save_dir ./outputs
```

## 输出

训练完成后，在`./outputs`目录下会生成：

- `cifar10_cnn_best.pth`: 最佳模型权重
- `cifar10_cnn_latest.pth`: 最新模型权重
- `training_curves.png`: 训练Loss和测试Accuracy曲线图

## 数据集

CIFAR-10数据集应位于`./cifar-10-batches-py`目录，包含：

- `data_batch_1` ~ `data_batch_5`: 训练数据（共50000张）
- `test_batch`: 测试数据（10000张）
- `batches.meta`: 元数据

## 性能优化

- **分布式数据并行**: 使用PyTorch DDP在8张GPU上并行训练
- **混合精度训练**: 使用AMP减少显存占用并加速训练
- **数据增强**: 随机裁剪和水平翻转提升泛化能力
- **学习率调度**: 余弦退火策略优化收敛
- **残差连接**: 支持更深的网络结构

## 依赖

- Python >= 3.10
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- tqdm >= 4.65.0

## 许可

本项目仅供学习和研究使用(Students Help Students)。
