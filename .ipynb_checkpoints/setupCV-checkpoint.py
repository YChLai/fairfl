import torch
from torchvision import datasets, transforms
import numpy as np
import random
import os
import copy
from fl_client import Client_CV
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

# --- 移植自 FedCorr 的噪声添加函数 ---
def add_noise_fedcorr(args, y_train, dict_users):
    """
    args: 包含 level_n_system (噪声客户端比例) 和 level_n_lowerb (噪声率下限)
    y_train: 原始的所有训练标签 (numpy array)
    dict_users: 客户端数据索引列表 (list of lists)
    """
    np.random.seed(args.seed)

    # 1. 随机选择哪些客户端是噪声客户端 (Gamma_s)
    # binomial(n, p, size): 1 表示选中，0 表示没选中
    gamma_s = np.random.binomial(1, args.level_n_system, args.num_clients)
    
    # 2. 为每个客户端生成具体的噪声率 (Gamma_c)
    # 噪声率在 [level_n_lowerb, 1.0] 之间波动
    gamma_c_initial = np.random.rand(args.num_clients)
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial

    y_train_noisy = copy.deepcopy(y_train)
    real_noise_level = np.zeros(args.num_clients)

    print("\n--- FedCorr Noise Injection Summary ---")
    for i in np.where(gamma_c > 0)[0]: # 遍历所有噪声客户端
        sample_idx = np.array(dict_users[i])
        
        # 对该客户端的样本进行随机翻转
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        
        # 随机替换为 0-9 的任意类别 (Symmetric Noise)
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, 10, len(noisy_idx))
        
        # 统计真实噪声率
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print(f"  > Client {i}: Target Noise Level={gamma_c[i]:.2f}, Real Noise Ratio={noise_ratio:.2f}")
        real_noise_level[i] = noise_ratio

    return y_train_noisy, gamma_s, real_noise_level

def split_noniid(trainset, num_clients, alpha=0.5):
    n_classes = 10
    label_distribution = np.random.dirichlet([alpha]*num_clients, n_classes)
    class_indices = [np.where(np.array(trainset.targets) == i)[0] for i in range(n_classes)]
    client_indices = [[] for _ in range(num_clients)]
    
    for c, fracs in zip(range(n_classes), label_distribution):
        total_size = len(class_indices[c])
        split_points = (np.cumsum(fracs) * total_size).astype(int)[:-1]
        indices_split = np.split(class_indices[c], split_points)
        
        for i in range(num_clients):
            client_indices[i] += indices_split[i].tolist()
            
    return client_indices

def setup_devices_cv(datapath, args):
    trainset, testset = get_cifar10(datapath)
    
    # 1. Non-IID 数据划分
    print("Splitting data (Non-IID)...")
    client_indices = split_noniid(trainset, args.num_clients, alpha=args.alpha)
    
    # 2. FedCorr 噪声注入
    print("Injecting noise using FedCorr method...")
    # 注意：CIFAR10_Local 的 targets 可能是 list，转为 numpy 处理
    y_train_original = np.array(trainset.targets)
    y_train_noisy, gamma_s, real_noise_level = add_noise_fedcorr(args, y_train_original, client_indices)
    
    # 记录真实的噪声分布，供 Server 或分析使用 (可选)
    # args.real_noise_level = real_noise_level 

    clients = []
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data, targets):
            self.data = data
            self.targets = targets
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i], self.targets[i]

    # 3. 创建客户端对象
    for i in range(args.num_clients):
        subset_indices = client_indices[i]
        
        # 数据：从原始 trainset 拿
        data_list = [trainset[idx][0] for idx in subset_indices]
        
        # 标签：从刚才生成的 y_train_noisy 拿
        target_list = y_train_noisy[subset_indices].tolist()
        
        client_dataset = SimpleDataset(data_list, target_list)
        
        model = CNNCifar(args)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        
        client = Client_CV(model, i, f"client{i}", len(client_dataset), client_dataset, testset, optimizer, args)
        clients.append(client)
        
    server_model = CNNCifar(args)
    server = Server(server_model, args.device)
    
    return clients, server