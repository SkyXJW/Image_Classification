#!/bin/bash
# 8 GPU分布式训练启动脚本

# 训练参数
DATA_DIR="./cifar-10-batches-py"
BATCH_SIZE=128  # 每个GPU的batch size
EPOCHS=200
LR=0.1
NUM_WORKERS=4
SAVE_DIR="./outputs"

# GPU配置
NUM_GPUS=8

echo "=========================================="
echo "启动CIFAR-10 CNN分布式训练"
echo "=========================================="
echo "GPU数量: ${NUM_GPUS}"
echo "总Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "训练轮数: ${EPOCHS}"
echo "初始学习率: ${LR}"
echo "=========================================="

# 使用torchrun启动分布式训练
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    train.py \
    --data_dir ${DATA_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --num_workers ${NUM_WORKERS} \
    --save_dir ${SAVE_DIR}

echo "=========================================="
