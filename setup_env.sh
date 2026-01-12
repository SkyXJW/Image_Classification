#!/bin/bash
# Conda环境创建脚本

ENV_NAME="cifar10_cnn"

echo "=========================================="
echo "创建CIFAR-10 CNN训练环境"
echo "=========================================="

# 检查conda是否已安装
if ! command -v conda &> /dev/null; then
    echo "错误: conda未安装，请先安装Anaconda或Miniconda"
    exit 1
fi

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 ${ENV_NAME} 已存在"
    read -p "是否删除并重新创建? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除现有环境..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "使用现有环境"
        exit 0
    fi
fi

# 创建conda环境
echo "创建conda环境: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=3.10 -y

# 激活环境
echo "激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# 安装依赖
echo "安装依赖包..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=========================================="
echo "环境创建完成!"
echo "使用以下命令激活环境:"
echo "  conda activate ${ENV_NAME}"
echo "=========================================="
