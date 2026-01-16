import os
import argparse
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# 引入原有模块 (假设这些文件在同一目录下)
import setupCV
import fl_utils as utils
# from models import CNNCifar # 不再使用
from fl_client import Client_CV
from server import Server

# =============================================================================
#  0. ResNet18 模型定义 (针对 CIFAR-10 适配)
# =============================================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18_CIFAR(nn.Module):
    def __init__(self, args=None):
        super(ResNet18_CIFAR, self).__init__()
        self.in_planes = 64
        num_classes = 10

        # CIFAR-10 适配: 3x3 kernel, stride=1, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512*BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # 兼容旧接口，返回 output, feature1, feature2 (后两个设为None)
        return out, None, None

# =============================================================================
#  1. 辅助模块：损失函数与 Mixup
# =============================================================================

class GCELoss(nn.Module):
    """
    广义交叉熵损失 (Generalized Cross Entropy Loss)
    
    """
    def __init__(self, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q

    def forward(self, pred, labels):
        # pred 是 logits，需要 softmax 转为概率
        pred = F.softmax(pred, dim=1)
        pred = torch.gather(pred, dim=1, index=labels.unsqueeze(1))
        loss = (1 - pred**self.q) / self.q
        return loss.mean()

def mixup_data_mse(x, y_onehot, alpha=1.0, device='cuda'):
    """
    Mixup 数据增强
    
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index, :]
    return mixed_x, mixed_y

def calculate_gmm_posterior(losses, gmm_params):
    """
    计算 P(Clean|Loss)
    
    """
    losses = losses.reshape(-1, 1)
    means = gmm_params['means'].flatten()
    covs = gmm_params['covariances'].flatten()
    weights = gmm_params['weights'].flatten()
    
    # 确保 means[0] 是较小的 (Clean)
    if means[0] > means[1]:
        means = means[::-1]
        covs = covs[::-1]
        weights = weights[::-1]
    
    probs = np.zeros((len(losses), 2))
    for k in range(2):
        sigma = np.sqrt(covs[k]) + 1e-6
        probs[:, k] = weights[k] * norm.pdf(losses.flatten(), means[k], sigma)
    
    # 归一化
    prob_sum = probs.sum(axis=1, keepdims=True) + 1e-12
    posteriors = probs / prob_sum
    return posteriors[:, 0]

# =============================================================================
#  2. 客户端类 (Client_New) - 核心逻辑修正
# =============================================================================

class Client_New(Client_CV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_rate = 0.0
        self.local_gmm_params = None
        self.clean_indices = []
        self.noisy_indices = [] 
        self.complex_indices = []
        self.trusted_noisy_map = {} 

    def compute_all_losses(self, model):
        """计算所有训练样本的 Loss，用于 GMM"""
        model.eval()
        # [修正] 使用 CrossEntropyLoss 处理 Logits
        criterion = nn.CrossEntropyLoss(reduction='none') 
        _, losses = utils.get_output(self.proto_loader, model, self.args, criterion)
        return losses

    def fit_local_gmm(self):
        """本地 GMM 拟合"""
        losses = self.compute_all_losses(self.model)
        losses_np = losses.reshape(-1, 1)
        
        # 增加一点容错，如果loss极小(过拟合)或没变化，不拟合
        if np.var(losses_np) < 1e-4: return None

        try:
            gmm = GaussianMixture(n_components=2, random_state=self.args.seed, max_iter=100).fit(losses_np)
            self.local_gmm_params = {
                'means': gmm.means_,
                'covariances': gmm.covariances_,
                'weights': gmm.weights_,
                'count': len(losses_np)
            }
        except:
            self.local_gmm_params = None
        return self.local_gmm_params

    def classify_samples(self, global_gmm_params):
        """样本三分类"""
        if global_gmm_params is None or self.local_gmm_params is None:
            # 降级处理：全部视为 Clean 或跳过
            self.clean_indices = list(range(self.train_size))
            return

        losses = self.compute_all_losses(self.model)
        
        prob_global_clean = calculate_gmm_posterior(losses, global_gmm_params)
        prob_local_clean = calculate_gmm_posterior(losses, self.local_gmm_params)
        
        threshold = 0.5
        self.clean_indices = []
        self.noisy_indices = []
        self.complex_indices = []
        
        for i in range(self.train_size):
            is_global_clean = prob_global_clean[i] >= threshold
            is_local_clean = prob_local_clean[i] >= threshold
            
            if is_global_clean and is_local_clean:
                self.clean_indices.append(i) # Clean
            elif (not is_global_clean) and (not is_local_clean):
                self.noisy_indices.append(i) # Noisy
            else:
                self.complex_indices.append(i) # Complex
                
        self.noise_rate = len(self.noisy_indices) / self.train_size

    def preprocess_noisy_samples(self):
        """重标记 + PCS"""
        if not self.noisy_indices:
            self.trusted_noisy_map = {}
            return

        # 1. 获取全局模型预测
        self.model.eval()
        noisy_subset = Subset(self.dataset_train, self.noisy_indices)
        loader = DataLoader(noisy_subset, batch_size=self.args.batch_size, shuffle=False)
        
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(self.args.device)
                output, _, _ = self.model(data)
                # output 是 Logits，需要 Softmax
                probs = torch.softmax(output, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                all_probs.append(max_probs.cpu())
                all_preds.append(preds.cpu())
                
        all_probs = torch.cat(all_probs).numpy()
        all_preds = torch.cat(all_preds).numpy()
        
        # 2. 筛选 (Thresholding)
        candidates = [] 
        zeta = 0.75 # 阈值可调
        for idx_in_subset, prob in enumerate(all_probs):
            if prob >= zeta:
                original_idx = self.noisy_indices[idx_in_subset]
                new_label = all_preds[idx_in_subset]
                candidates.append((original_idx, new_label))
        
        if not candidates:
            self.trusted_noisy_map = {}
            return
            
        # 3. PCS: 本地模型一致性校验
        temp_model = copy.deepcopy(self.model)
        temp_model.train()
        # [修正] 增加 weight_decay
        temp_optimizer = torch.optim.SGD(temp_model.parameters(), lr=self.args.lr, weight_decay=5e-4)
        
        # [修正] 使用 CrossEntropyLoss
        temp_criterion = nn.CrossEntropyLoss()
        
        train_loader = DataLoader(self.dataset_train, batch_size=self.args.batch_size, shuffle=True)
        for data, target in train_loader:
            data, target = data.to(self.args.device), target.to(self.args.device)
            temp_optimizer.zero_grad()
            output, _, _ = temp_model(data)
            loss = temp_criterion(output, target)
            loss.backward()
            temp_optimizer.step()
            
        temp_model.eval()
        self.trusted_noisy_map = {}
        
        cand_indices = [x[0] for x in candidates]
        cand_subset = Subset(self.dataset_train, cand_indices)
        cand_loader = DataLoader(cand_subset, batch_size=self.args.batch_size, shuffle=False)
        
        local_preds = []
        with torch.no_grad():
            for data, _ in cand_loader:
                data = data.to(self.args.device)
                output, _, _ = temp_model(data)
                pred = output.argmax(dim=1)
                local_preds.append(pred.cpu())
        if local_preds:
            local_preds = torch.cat(local_preds).numpy()
        
        for i, (orig_idx, global_label) in enumerate(candidates):
            local_label = local_preds[i]
            if global_label == local_label:
                self.trusted_noisy_map[orig_idx] = int(global_label)

    def local_train_adaptive(self, local_epoch):
        """适应性本地训练"""
        self.model.train()
        
        # 准备 Loader
        def get_iter(indices):
            if len(indices) > 0:
                loader = DataLoader(Subset(self.dataset_train, indices), 
                                    batch_size=self.args.batch_size, shuffle=True)
                return loader, iter(loader)
            return None, None

        clean_loader, clean_iter = get_iter(self.clean_indices)
        complex_loader, complex_iter = get_iter(self.complex_indices)
        
        noisy_loader, noisy_iter = None, None
        if len(self.trusted_noisy_map) > 0:
            idxs = list(self.trusted_noisy_map.keys())
            lbls = list(self.trusted_noisy_map.values())
            
            # 提取数据
            if isinstance(self.dataset_train.data, list):
                all_data = torch.stack(self.dataset_train.data)
            else:
                all_data = self.dataset_train.data
            
            t_data = all_data[idxs]
            t_targets = torch.tensor(lbls).long()
            
            noisy_ds = TensorDataset(t_data, t_targets)
            noisy_loader = DataLoader(noisy_ds, batch_size=self.args.batch_size, shuffle=True)
            noisy_iter = iter(noisy_loader)

        # [修正] 损失函数定义
        criterion_ce = nn.CrossEntropyLoss()
        criterion_mse = nn.MSELoss()
        criterion_gce = GCELoss(q=0.7)
        
        lam_c, lam_n, lam_h = 0.4, 0.3, 0.3 

        for epoch in range(local_epoch):
            # 确定步数
            steps = 0
            if clean_loader: steps = max(steps, len(clean_loader))
            if complex_loader: steps = max(steps, len(complex_loader))
            if noisy_loader: steps = max(steps, len(noisy_loader))
            
            for _ in range(steps):
                self.optimizer.zero_grad()
                total_loss = 0.0
                has_loss = False
                
                # 1. Clean (CE)
                if clean_loader:
                    try:
                        data, target = next(clean_iter)
                    except StopIteration:
                        clean_iter = iter(clean_loader)
                        data, target = next(clean_iter)
                    data, target = data.to(self.args.device), target.to(self.args.device)
                    
                    output, _, _ = self.model(data)
                    loss_c = criterion_ce(output, target)
                    total_loss += lam_c * loss_c
                    has_loss = True

                # 2. Complex (GCE)
                if complex_loader:
                    try:
                        data, target = next(complex_iter)
                    except StopIteration:
                        complex_iter = iter(complex_loader)
                        data, target = next(complex_iter)
                    data, target = data.to(self.args.device), target.to(self.args.device)
                    
                    output, _, _ = self.model(data)
                    loss_h = criterion_gce(output, target)
                    total_loss += lam_h * loss_h
                    has_loss = True
                    
                # 3. Noisy (Mixup + MSE)
                if noisy_loader:
                    try:
                        data, target = next(noisy_iter)
                    except StopIteration:
                        noisy_iter = iter(noisy_loader)
                        data, target = next(noisy_iter)
                    
                    if len(data) > 1:
                        data, target = data.to(self.args.device), target.to(self.args.device)
                        target_oh = F.one_hot(target, num_classes=10).float()
                        
                        mixed_data, mixed_target = mixup_data_mse(data, target_oh, alpha=0.5, device=self.args.device)
                        output, _, _ = self.model(mixed_data)
                        
                        # [修正] MSE 需要概率分布，必须做 Softmax
                        probs = torch.softmax(output, dim=1)
                        loss_n = criterion_mse(probs, mixed_target)
                        
                        total_loss += lam_n * loss_n
                        has_loss = True

                if has_loss:
                    total_loss.backward()
                    self.optimizer.step()

        # [修正] 强制更新本地权重字典，以便 Server 聚合
        for k, v in self.model.named_parameters():
             if k in self.W:
                 self.W[k].data = v.data.clone()
        # 计算 dW (可选，取决于 Server 逻辑，但更新 W 是必须的)
        for k in self.W:
            self.dW[k] = self.W[k].data - self.W_old[k].data

# =============================================================================
#  3. Server 类 (Server_New)
# =============================================================================

class Server_New(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_gmm_params = None

    def aggregate_gmm(self, client_gmms_list):
        """FedDiv 聚合"""
        valid_gmms = [p for p in client_gmms_list if p is not None]
        if not valid_gmms: return None
            
        total_samples = sum([p['count'] for p in valid_gmms])
        
        agg_means = np.zeros_like(valid_gmms[0]['means'])
        agg_covs = np.zeros_like(valid_gmms[0]['covariances'])
        agg_weights = np.zeros_like(valid_gmms[0]['weights'])
        
        for p in valid_gmms:
            weight = p['count'] / total_samples
            means = p['means'].flatten()
            covs = p['covariances'].flatten()
            weights = p['weights'].flatten()
            
            if means[0] > means[1]:
                means = means[::-1]
                covs = covs[::-1]
                weights = weights[::-1]
            
            try:
                means = means.reshape(agg_means.shape)
                covs = covs.reshape(agg_covs.shape)
                weights = weights.reshape(agg_weights.shape)
            except: continue
            
            agg_means += weight * means
            agg_covs += weight * covs
            agg_weights += weight * weights
            
        self.global_gmm_params = {
            'means': agg_means, 'covariances': agg_covs, 'weights': agg_weights
        }
        return self.global_gmm_params

    def aggregate_weights_adaptive(self, selected_clients, noise_rates):
        """自适应聚合"""
        max_noise = max(noise_rates) if len(noise_rates) > 0 and max(noise_rates) > 1e-6 else 1.0
        normalized_noise = [nr / max_noise for nr in noise_rates]
        
        coeffs = []
        for i, client in enumerate(selected_clients):
            m_i = client.train_size
            eta = normalized_noise[i]
            coeffs.append(m_i * np.exp(-eta))
            
        total_coeff = sum(coeffs)
        
        for k in self.W.keys():
            self.W[k].data.zero_()
            
        for i, client in enumerate(selected_clients):
            w_i = coeffs[i] / (total_coeff + 1e-12)
            for k in self.W.keys():
                self.W[k].data += client.W[k].data * w_i

# =============================================================================
#  4. 主流程
# =============================================================================

def run_full_logic(clients, server, args):
    rs = torch.ones(len(clients)) / len(clients)
    frame = pd.DataFrame()
    print(f"Start Full Training on {args.device}")

    for round_idx in range(1, args.num_rounds + 1):
        print(f"\n--- Round {round_idx} ---")

        # 1. 广播
        for client in clients:
            client.download_from_server(server)

        # 2. 噪声滤波
        if round_idx > args.lid_start_round:
            client_gmms = []
            for client in clients:
                gmm_p = client.fit_local_gmm()
                client_gmms.append(gmm_p)
            
            global_gmm = server.aggregate_gmm(client_gmms)
            
            client_noise_rates = []
            if global_gmm:
                print(f"  > Global GMM Means: {global_gmm['means'].flatten()}")
                for client in clients:
                    client.classify_samples(global_gmm)
                    client.preprocess_noisy_samples()
                    client_noise_rates.append(client.noise_rate)
                    
                    print(f"    Client {client.id}: Clean {len(client.clean_indices)}, "
                          f"TrustedNoisy {len(client.trusted_noisy_map)}, "
                          f"Complex {len(client.complex_indices)}")
            else:
                client_noise_rates = [0.0] * len(clients)
        else:
            print("  > Warm-up: Skipping Noise Filter")
            client_noise_rates = [0.0] * len(clients)
            for client in clients:
                client.clean_indices = list(range(client.train_size))
                client.noisy_indices = []
                client.complex_indices = []
                client.trusted_noisy_map = {}

        # 3. 本地训练
        for client in clients:
            client.local_train_adaptive(args.local_epoch)

        # 4. 聚合
        server.aggregate_weights_adaptive(clients, client_noise_rates)

        # 5. 评估
        if round_idx % 1 == 0:
            server.model.eval()
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                # 使用测试集评估
                for data, target in clients[0].test_loader:
                    data, target = data.to(args.device), target.to(args.device)
                    output, _, _ = server.model(data)
                    pred = output.argmax(dim=1)
                    total_correct += pred.eq(target).sum().item()
                    total_samples += target.size(0)
            
            acc = 100. * total_correct / total_samples
            print(f"  > Global Model Acc: {acc:.2f}%")
            frame.loc[round_idx, 'acc'] = acc

    return frame

# =============================================================================
#  5. Setup & Entry
# =============================================================================

def setup_full_environment(args):
    trainset, testset = setupCV.get_cifar10(args.datapath)
    client_indices = setupCV.split_noniid(trainset, args.num_clients, alpha=args.alpha)
    
    y_train_original = np.array(trainset.targets)
    y_train_noisy, _, _ = setupCV.add_noise_fedcorr(args, y_train_original, client_indices)
    
    clients = []
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data, targets):
            if isinstance(data, list):
                self.data = torch.stack(data)
            else:
                self.data = data
            self.targets = targets
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i], self.targets[i]

    for i in range(args.num_clients):
        subset_indices = client_indices[i]
        data_list = [trainset[idx][0] for idx in subset_indices]
        target_list = y_train_noisy[subset_indices].tolist()
        
        client_dataset = SimpleDataset(data_list, target_list)
        
        # [修正] 使用 ResNet + Weight Decay
        model = ResNet18_CIFAR(args)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        
        client = Client_New(model, i, f"client{i}", len(client_dataset), client_dataset, testset, optimizer, args)
        clients.append(client)
        
    server_model = ResNet18_CIFAR(args)
    server = Server_New(server_model, args.device)
    
    return clients, server

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--outbase', type=str, default='./outputs')
    parser.add_argument('--num_rounds', type=int, default=50)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.05) # [建议] ResNet 0.05 or 0.1
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5) 
    parser.add_argument('--level_n_system', type=float, default=0.3)
    parser.add_argument('--level_n_lowerb', type=float, default=0.5)
    parser.add_argument('--lid_start_round', type=int, default=10) # [建议] 预热10-20轮
    
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    Path(args.outbase).mkdir(parents=True, exist_ok=True)
    
    clients, server = setup_full_environment(args)
    res = run_full_logic(clients, server, args)
    res.to_csv(os.path.join(args.outbase, 'acc_final.csv'))
    print("Done.")