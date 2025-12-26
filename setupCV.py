import torch
from torchvision import datasets, transforms
import numpy as np
import random
from client import Client_CV
from server import Server
from models import CNNCifar

def get_cifar10(datapath):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(root=datapath, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=datapath, train=False, download=True, transform=transform)
    return trainset, testset

def add_noise(dataset, noise_rate, seed=None):
    """
    模拟噪声率：随机翻转部分标签
    """
    if noise_rate <= 0:
        return dataset
    
    np.random.seed(seed)
    targets = np.array(dataset.targets)
    n_samples = len(targets)
    n_noise = int(noise_rate * n_samples)
    noise_idx = np.random.choice(n_samples, n_noise, replace=False)
    
    # 随机替换为其他类别
    for idx in noise_idx:
        current_label = targets[idx]
        new_label = np.random.randint(0, 10)
        while new_label == current_label:
            new_label = np.random.randint(0, 10)
        targets[idx] = new_label
        
    dataset.targets = targets.tolist()
    return dataset

def split_noniid(trainset, num_clients, alpha=0.5):
    """
    使用 Dirichlet 分布模拟类别多样性差异 (Non-IID)
    alpha 越小，Non-IID 程度越高（即每个客户端拥有的类别越少）
    """
    n_classes = 10
    label_distribution = np.random.dirichlet([alpha]*num_clients, n_classes)
    # label_distribution: [n_classes, num_clients]
    
    class_indices = [np.where(np.array(trainset.targets) == i)[0] for i in range(n_classes)]
    
    client_indices = [[] for _ in range(num_clients)]
    
    for c, fracs in zip(range(n_classes), label_distribution):
        # 对每个类别，按比例分配给客户端
        total_size = len(class_indices[c])
        split_points = (np.cumsum(fracs) * total_size).astype(int)[:-1]
        indices_split = np.split(class_indices[c], split_points)
        
        for i in range(num_clients):
            client_indices[i] += indices_split[i].tolist()
            
    return client_indices

def setup_devices_cv(datapath, args):
    trainset, testset = get_cifar10(datapath)
    
    # 1. 模拟类别多样性 (Class Diversity) -> Non-IID 划分
    client_indices = split_noniid(trainset, args.num_clients, alpha=args.alpha)
    
    clients = []
    
    # 预先生成每个客户端的噪声率配置，如果 args.noise_rate 是一个列表则直接使用，否则随机生成
    # 这里假设 args.noise_rate 是一个基础值，我们让部分客户端噪声大，部分小，以体现 Contribution 差异
    
    for i in range(args.num_clients):
        # 为每个客户端创建一个数据子集
        subset_indices = client_indices[i]
        
        # 必须深拷贝，因为我们要修改 target 制造噪声，不能影响原始数据集
        # 注意：Torchvision Dataset 深拷贝比较耗内存，实际工程中建议重写Dataset类只存索引和labels
        # 这里为了简化直接使用 Subset + Copy 逻辑模拟
        from torch.utils.data import Subset
        client_trainset = Subset(trainset, subset_indices)
        
        # 处理数据对象以便修改 targets (Subset 无法直接修改 targets)
        # 这里为了演示，我们提取数据并重新封装，或者直接修改内存中的targets如果结构允许
        # 简单做法：构造一个新的简单Dataset
        data_list = [trainset[idx][0] for idx in subset_indices]
        target_list = [trainset[idx][1] for idx in subset_indices]
        
        # 2. 模拟噪声率 (Noise Rate)
        # 设定：前 20% 的客户端噪声很大 (0.8)，其余正常 (0.0 - 0.1)
        if i < int(args.num_clients * 0.2):
            current_noise = 0.8 # 高噪声 -> 低 Reputation
        else:
            current_noise = 0.05 # 低噪声 -> 高 Reputation
            
        # 注入噪声
        n_noise = int(len(target_list) * current_noise)
        noise_indices = np.random.choice(len(target_list), n_noise, replace=False)
        for idx in noise_indices:
            target_list[idx] = np.random.randint(0, 10) # 简单随机噪声
            
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data, targets):
                self.data = data
                self.targets = targets
            def __len__(self): return len(self.data)
            def __getitem__(self, i): return self.data[i], self.targets[i]
            
        client_dataset = SimpleDataset(data_list, target_list)
        
        # 初始化模型和优化器
        model = CNNCifar(args)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        
        client = Client_CV(model, i, f"client{i}", len(client_dataset), client_dataset, testset, optimizer, args)
        clients.append(client)
        
    # 初始化服务器
    server_model = CNNCifar(args)
    server = Server(server_model, args.device)
    
    return clients, server