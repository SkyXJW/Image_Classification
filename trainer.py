"""分布式训练器模块"""
import os
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据相关
    data_dir: str = "./cifar-10-batches-py"
    batch_size: int = 128  # 每个GPU的batch size
    num_workers: int = 4
    
    # 模型相关
    num_classes: int = 10
    dropout_rate: float = 0.5
    
    # 训练相关
    epochs: int = 200
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    
    # 分布式相关
    use_amp: bool = True  # 混合精度训练
    
    # 输出相关
    save_dir: str = "./outputs"
    model_name: str = "cifar10_cnn"


class DistributedTrainer:
    """分布式训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: TrainingConfig,
        rank: int = 0,
        world_size: int = 1
    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            config: 训练配置
            rank: 当前进程的rank
            world_size: 总进程数
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # 模型
        self.model = model.to(self.device)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])
        
        # 数据加载器
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )
        
        # 混合精度训练
        if config.use_amp and torch.cuda.is_available():
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
        
        # 记录
        self.train_losses = []
        self.test_accuracies = []
        self.best_acc = 0.0
        
        # 创建输出目录
        if rank == 0:
            os.makedirs(config.save_dir, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 只在主进程显示进度条
        if self.rank == 0:
            pbar = tqdm(self.train_loader, 
                       desc=f'Epoch {epoch}/{self.config.epochs} [Train]',
                       ncols=100,
                       leave=False)
        else:
            pbar = self.train_loader
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 混合精度训练
            if self.config.use_amp and self.scaler is not None:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条显示当前loss
            if self.rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, epoch: int) -> float:
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        # 只在主进程显示进度条
        if self.rank == 0:
            pbar = tqdm(self.test_loader,
                       desc=f'Epoch {epoch}/{self.config.epochs} [Eval]',
                       ncols=100,
                       leave=False)
        else:
            pbar = self.test_loader
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 更新进度条显示当前准确率
                if self.rank == 0:
                    current_acc = correct / total
                    pbar.set_postfix({'acc': f'{current_acc:.4f}'})
        
        accuracy = correct / total
        return accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点"""
        if self.rank != 0:
            return
        
        model_state = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'train_losses': self.train_losses,
            'test_accuracies': self.test_accuracies
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.config.save_dir, f'{self.config.model_name}_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config.save_dir, f'{self.config.model_name}_best.pth')
            torch.save(checkpoint, best_path)
    
    def train(self) -> Tuple[List[float], List[float]]:
        """完整训练流程"""
        if self.rank == 0:
            print("=" * 80)
            print(f"{'CIFAR-10 CNN Training':^80}")
            print("=" * 80)
            print(f"{'Epochs:':<20} {self.config.epochs}")
            print(f"{'Device:':<20} {self.device}")
            print(f"{'World Size:':<20} {self.world_size}")
            print(f"{'Batch Size:':<20} {self.config.batch_size} × {self.world_size} = {self.config.batch_size * self.world_size}")
            print(f"{'Learning Rate:':<20} {self.config.learning_rate}")
            print(f"{'Mixed Precision:':<20} {'Enabled' if self.config.use_amp else 'Disabled'}")
            print("=" * 80)
        
        for epoch in range(1, self.config.epochs + 1):
            # 训练
            avg_loss = self.train_epoch(epoch)
            self.train_losses.append(avg_loss)
            
            # 评估
            accuracy = self.evaluate(epoch)
            self.test_accuracies.append(accuracy)
            
            # 更新学习率
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            
            # 保存最佳模型
            is_best = accuracy > self.best_acc
            if is_best:
                self.best_acc = accuracy
            
            self.save_checkpoint(epoch, is_best)
            
            # 打印每轮训练信息
            if self.rank == 0:
                status = "🌟 NEW BEST!" if is_best else ""
                print(f"Epoch [{epoch:3d}/{self.config.epochs}] "
                      f"Loss: {avg_loss:.4f} | "
                      f"Acc: {accuracy:.4f} | "
                      f"Best: {self.best_acc:.4f} | "
                      f"LR: {current_lr:.6f} "
                      f"{status}")
        
        if self.rank == 0:
            print("=" * 80)
            print(f"{'Training Completed!':^80}")
            print(f"{'Best Accuracy:':<20} {self.best_acc:.4f}")
            print("=" * 80)
        
        return self.train_losses, self.test_accuracies
