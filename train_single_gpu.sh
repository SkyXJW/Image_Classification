#!/bin/bash
# 单GPU训练启动脚本（用于测试）

# 训练参数
DATA_DIR="./cifar-10-batches-py"
BATCH_SIZE=128
EPOCHS=200
LR=0.1
NUM_WORKERS=4
SAVE_DIR="./outputs"

echo "=========================================="
echo "启动CIFAR-10 CNN单GPU训练"
echo "=========================================="

python train.py \
    --data_dir ${DATA_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --num_workers ${NUM_WORKERS} \
    --save_dir ${SAVE_DIR}

echo "=========================================="
echo "训练完成!"
echo "=========================================="
