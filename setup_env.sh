#!/bin/bash
# Conda环境创建脚本

# 错误处理设置
set -e          # 任何命令失败时立即退出
set -o pipefail # 管道中任何命令失败都会导致整个管道失败
set -u          # 使用未定义变量时退出

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
    set +e  # 临时禁用错误退出，允许用户取消操作
    read -p "是否删除并重新创建? (y/n) " -n 1 -r
    set -e  # 重新启用错误退出
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
if ! conda create -n ${ENV_NAME} python=3.10 -y; then
    echo "错误: conda环境创建失败"
    exit 1
fi

# 激活环境
echo "激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
if ! conda activate ${ENV_NAME}; then
    echo "错误: 无法激活conda环境 ${ENV_NAME}"
    exit 1
fi

# 安装依赖
echo "安装依赖包..."
if ! pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple; then
    echo "错误: 依赖包安装失败"
    exit 1
fi

echo "=========================================="
echo "环境创建完成!"
echo "使用以下命令激活环境:"
echo "  conda activate ${ENV_NAME}"
echo "=========================================="
