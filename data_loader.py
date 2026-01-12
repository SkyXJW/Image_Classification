"""CIFAR-10数据加载和预处理模块"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms


class CIFAR10Dataset(Dataset):
    """CIFAR-10数据集类"""
    
    def __init__(self, data_dir: str, train: bool = True, transform=None):
        """
        Args:
            data_dir: CIFAR-10数据目录路径
            train: 是否为训练集
            transform: 数据变换
        """
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        
        # 加载数据
        self.data, self.labels = self._load_data()
    
    def _load_data(self):
        """从本地文件加载CIFAR-10数据"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        data_list = []
        labels_list = []
        
        if self.train:
            # 加载训练数据 (data_batch_1 到 data_batch_5)
            for i in range(1, 6):
                file_path = os.path.join(self.data_dir, f'data_batch_{i}')
                batch_data, batch_labels = self._load_batch(file_path)
                data_list.append(batch_data)
                labels_list.extend(batch_labels)
        else:
            # 加载测试数据
            file_path = os.path.join(self.data_dir, 'test_batch')
            batch_data, batch_labels = self._load_batch(file_path)
            data_list.append(batch_data)
            labels_list.extend(batch_labels)
        
        # 合并所有批次
        data = np.concatenate(data_list, axis=0)
        labels = np.array(labels_list)
        
        return data, labels
    
    def _load_batch(self, file_path: str):
        """加载单个批次文件"""
        try:
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            
            data = batch[b'data']
            labels = batch[b'labels']
            
            # 重塑数据: (N, 3072) -> (N, 3, 32, 32)
            data = data.reshape(-1, 3, 32, 32)
            
            return data, labels
        except Exception as e:
            raise RuntimeError(f"加载数据文件失败 {file_path}: {str(e)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        # 转换为float并归一化到[0, 1]
        img = torch.from_numpy(img).float() / 255.0
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class CIFAR10DataModule:
    """CIFAR-10数据加载模块"""
    
    # CIFAR-10数据集的均值和标准差
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2470, 0.2435, 0.2616]
    
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 4):
        """
        Args:
            data_dir: 数据集目录路径
            batch_size: 每个GPU的batch size
            num_workers: 数据加载的工作进程数
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 训练数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(self.MEAN, self.STD)
        ])
        
        # 测试数据预处理
        self.test_transform = transforms.Compose([
            transforms.Normalize(self.MEAN, self.STD)
        ])
    
    def get_train_loader(self, distributed: bool = False, rank: int = 0, world_size: int = 1) -> DataLoader:
        """获取训练数据加载器"""
        train_dataset = CIFAR10Dataset(
            self.data_dir,
            train=True,
            transform=self.train_transform
        )
        
        sampler = None
        shuffle = True
        
        if distributed:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            shuffle = False
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader
    
    def get_test_loader(self) -> DataLoader:
        """获取测试数据加载器"""
        test_dataset = CIFAR10Dataset(
            self.data_dir,
            train=False,
            transform=self.test_transform
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return test_loader
