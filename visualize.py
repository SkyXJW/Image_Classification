"""训练可视化模块"""
import os
from typing import List
import matplotlib.pyplot as plt


def plot_training_curves(
    losses: List[float],
    accuracies: List[float],
    save_path: str = "./outputs/training_curves.png"
) -> None:
    """
    绘制训练曲线并保存
    
    Args:
        losses: 训练损失列表
        accuracies: 测试准确率列表
        save_path: 保存路径
    """
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(losses) + 1)
    
    # 绘制Loss曲线
    ax1.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 绘制Accuracy曲线
    ax2.plot(epochs, accuracies, 'r-', linewidth=2, label='Test Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Test Accuracy Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 1])
    
    # 添加最佳准确率标注
    best_acc = max(accuracies)
    best_epoch = accuracies.index(best_acc) + 1
    ax2.axhline(y=best_acc, color='g', linestyle='--', alpha=0.5)
    ax2.text(len(epochs) * 0.7, best_acc + 0.02, 
             f'Best: {best_acc:.4f} (Epoch {best_epoch})',
             fontsize=10, color='g')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存至: {save_path}")
