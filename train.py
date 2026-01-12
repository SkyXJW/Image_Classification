"""主训练脚本"""
import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import CIFAR10CNN
from data_loader import CIFAR10DataModule
from trainer import DistributedTrainer, TrainingConfig
from visualize import plot_training_curves


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # 初始化进程组
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # 单GPU或CPU模式
        return 0, 1, 0


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN训练')
    parser.add_argument('--data_dir', type=str, default='./cifar-10-batches-py',
                        help='CIFAR-10数据集路径')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='每个GPU的batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='初始学习率')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数')
    parser.add_argument('--save_dir', type=str, default='./outputs',
                        help='模型保存目录')
    parser.add_argument('--no_amp', action='store_true',
                        help='禁用混合精度训练')
    
    args = parser.parse_args()
    
    # 设置分布式环境
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print("=" * 60)
        print("CIFAR-10 CNN分类器训练")
        print("=" * 60)
        print(f"数据目录: {args.data_dir}")
        print(f"Batch size: {args.batch_size} × {world_size} = {args.batch_size * world_size}")
        print(f"训练轮数: {args.epochs}")
        print(f"初始学习率: {args.lr}")
        print(f"混合精度训练: {not args.no_amp}")
        print(f"GPU数量: {world_size}")
        print("=" * 60)
    
    # 创建训练配置
    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.lr,
        use_amp=not args.no_amp,
        save_dir=args.save_dir
    )
    
    # 创建数据加载器
    data_module = CIFAR10DataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    train_loader = data_module.get_train_loader(
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size
    )
    test_loader = data_module.get_test_loader()
    
    # 创建模型
    model = CIFAR10CNN(
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate
    )
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数总数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print("=" * 60)
    
    # 创建训练器
    trainer = DistributedTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        rank=rank,
        world_size=world_size
    )
    
    # 训练
    try:
        losses, accuracies = trainer.train()
        
        # 绘制训练曲线（仅在主进程）
        if rank == 0:
            plot_training_curves(
                losses,
                accuracies,
                save_path=os.path.join(config.save_dir, 'training_curves.png')
            )
            print("训练完成!")
    
    except KeyboardInterrupt:
        if rank == 0:
            print("\n训练被中断")
    
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
