# 已停用

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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# 引入原有模块
import setupCV
import fl_utils as utils
from models import CNNCifar
from fl_client import Client_CV
from server import Server

# =============================================================================
#  Auxiliary Functions for Mixup & Loss
# =============================================================================

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def probability_div(loss_array, gmm_params):
    """
    使用全局GMM参数计算样本属于Clean(低Loss)组件的概率 (Posterior Probability)
    gmm_params: {'means': [m1, m2], 'covariances': [c1, c2], 'weights': [w1, w2]}
    """
    means = gmm_params['means'].flatten()
    covs = gmm_params['covariances'].flatten() 
    weights = gmm_params['weights'].flatten()
    
    # 确保 means[0] 是较小的那个 (Clean Component)
    # 聚合时已经做过排序，这里再次double check
    if means[0] > means[1]:
        means = means[::-1]
        covs = covs[::-1]
        weights = weights[::-1]
        
    probs = np.zeros((len(loss_array), 2))
    
    # 计算 pdf
    for k in range(2):
        # 加上 epsilon 防止除零，covs 对应方差 sigma^2
        sigma = np.sqrt(covs[k]) + 1e-6
        probs[:, k] = weights[k] * norm.pdf(loss_array, means[k], sigma)
        
    # 归一化得到后验概率 P(Clean|Loss)
    prob_sum = probs.sum(axis=1, keepdims=True) + 1e-12
    posteriors = probs / prob_sum
    
    # 返回属于 Clean 组件 (index 0) 的概率
    return posteriors[:, 0] 

# =============================================================================
#  New Client Class (Full Implementation)
# =============================================================================

class Client_New(Client_CV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_rate = 0.0      # S_eta
        self.class_diversity = 0.0 # C_i
        
        # 存储样本的 Clean/Noisy 概率或标签
        # 初始化为全 Clean (1.0)
        self.sample_clean_prob = np.ones(self.train_size)
        
        # 本地 GMM 参数
        self.local_gmm_params = None 
        
        # 记录每轮从 Server 接收到的个性化模型参数 (Incentive mechanism 结果)
        self.personalized_W = None

    def calc_class_diversity(self, global_num_classes=10):
        """
        计算类别多样性 C_i = k_i / K
        """
        unique_labels = np.unique(self.dataset_train.targets)
        k_i = len(unique_labels)
        self.class_diversity = k_i / global_num_classes
        return self.class_diversity

    def compute_loss_values(self):
        """
        计算所有训练样本的 Loss 值，用于 GMM 拟合
        """
        criterion = nn.NLLLoss(reduction='none')
        # 使用 proto_loader (不 shuffle) 来对应索引
        _, loss_whole = utils.get_output(self.proto_loader, self.model, self.args, criterion)
        return loss_whole

    def fit_local_gmm(self):
        """
        拟合本地 GMM 模型。
        处理异常：如果 Loss 区分度太小（如刚开始训练），则返回 None 或默认值。
        """
        losses = self.compute_loss_values()
        losses = losses.reshape(-1, 1)
        
        # 简单的方差检查，如果 Loss 几乎一样，说明还没学到东西，不拟合
        if np.var(losses) < 1e-4:
            return None

        try:
            gmm = GaussianMixture(n_components=2, random_state=self.args.seed, max_iter=100).fit(losses)
            self.local_gmm_params = {
                'means': gmm.means_,
                'covariances': gmm.covariances_,
                'weights': gmm.weights_,
                'count': len(losses) 
            }
        except Exception as e:
            print(f"  Warning: Client {self.id} GMM fit failed: {e}")
            self.local_gmm_params = None
            
        return self.local_gmm_params

    def update_noise_info(self, global_gmm_params):
        """
        根据全局 GMM 滤波器更新噪声率和样本划分
        """
        losses = self.compute_loss_values()
        
        # 计算每个样本是 Clean 的概率 P(Clean|Loss)
        self.sample_clean_prob = probability_div(losses, global_gmm_params)
        
        # 定义噪声样本：概率 < 0.5 (或者使用 args.confidence_thres)
        # S_eta = N_i / T_i (噪声样本数 / 总数)
        # 也可以使用软标签求和： sum(1 - prob_clean) / T_i
        # 这里使用硬阈值计算噪声率，与文档定义一致
        threshold = 0.5
        noisy_count = np.sum(self.sample_clean_prob < threshold)
        self.noise_rate = noisy_count / len(losses)

    def local_train_adaptive(self, local_epoch):
        """
        完整版适应性训练：
        在一个 Batch 内区分 Clean/Noisy，分别计算 Loss 并加权。
        L = lambda_c * L_clean + lambda_n * L_noisy
        """
        self.model.train()
        
        # 获取 dataset 里的 data 和 targets
        # 注意：dataset_train 是 SimpleDataset, targets 是 list
        # 为了高效索引，我们需要 tensor 形式
        all_data = torch.stack(self.dataset_train.data).to(self.args.device) if isinstance(self.dataset_train.data, list) else self.dataset_train.data.to(self.args.device)
        all_targets = torch.tensor(self.dataset_train.targets).to(self.args.device)
        all_probs = torch.tensor(self.sample_clean_prob).float().to(self.args.device) # Clean Prob
        
        # 构建 TensorDataset 以利用 DataLoader 的 shuffle
        # 我们需要索引来找回对应的 probability
        indices = torch.arange(len(self.dataset_train))
        temp_dataset = TensorDataset(all_data, all_targets, indices)
        temp_loader = DataLoader(temp_dataset, batch_size=self.args.batch_size, shuffle=True)

        for epoch in range(local_epoch):
            for data_batch, target_batch, idx_batch in temp_loader:
                data_batch, target_batch = data_batch.to(self.args.device), target_batch.to(self.args.device)
                
                # 获取当前 Batch 每个样本的 Clean Probability
                probs_batch = all_probs[idx_batch]
                
                # 划分 Clean 和 Noisy 集合 (基于阈值 0.5)
                clean_mask = probs_batch >= 0.5
                noisy_mask = ~clean_mask
                
                self.optimizer.zero_grad()
                loss_final = 0.0
                
                # --- 1. Clean Samples (Cross Entropy) ---
                if clean_mask.sum() > 0:
                    clean_data = data_batch[clean_mask]
                    clean_target = target_batch[clean_mask]
                    
                    output_clean, _, _ = self.model(clean_data)
                    loss_c = F.nll_loss(output_clean, clean_target)
                    
                    # 按照 lambda_c 加权 (文档 source:17 lambda_c + lambda_n = 1)
                    # 简单设置 lambda_c = 0.5 或根据数量动态调整
                    loss_final += 0.5 * loss_c

                # --- 2. Noisy Samples (Mixup + MSE) ---
                if noisy_mask.sum() > 0:
                    noisy_data = data_batch[noisy_mask]
                    noisy_target = target_batch[noisy_mask]
                    
                    # 如果只有1个样本，mixup 会报错或无意义，做个判断
                    if len(noisy_data) > 1:
                        # Mixup
                        mixed_data, targets_a, targets_b, lam = mixup_data(noisy_data, noisy_target, alpha=0.5, device=self.args.device)
                        output_noisy, _, _ = self.model(mixed_data)
                        
                        # MSE Loss: 需要预测概率和 One-hot 标签
                        probs_pred = torch.exp(output_noisy)
                        
                        target_a_oh = F.one_hot(targets_a, num_classes=10).float()
                        target_b_oh = F.one_hot(targets_b, num_classes=10).float()
                        mixed_target = lam * target_a_oh + (1 - lam) * target_b_oh
                        
                        loss_n = F.mse_loss(probs_pred, mixed_target)
                        loss_final += 0.5 * loss_n
                    else:
                        # 只有1个样本时退化为普通 CE 或跳过
                        output_noisy, _, _ = self.model(noisy_data)
                        loss_n = F.nll_loss(output_noisy, noisy_target) # 备选
                        loss_final += 0.5 * loss_n

                if loss_final != 0.0:
                    loss_final.backward()
                    self.optimizer.step()

        # 计算 Update 量 (dW) 用于后续可能得逻辑（虽然本脚本用参数替换法）
        for k in self.W:
            self.dW[k] = self.W[k].data - self.W_old[k].data
            
    def apply_personalized_update(self, new_state_dict):
        """
        加载 Server 分发的 Masked 模型
        """
        self.model.load_state_dict(new_state_dict)
        # 更新 W 和 W_old，确保下一次训练从这个起点开始
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

# =============================================================================
#  New Server Class (Full Implementation)
# =============================================================================

class Server_New(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_gmm_params = None

    def aggregate_gmm(self, client_gmms_list):
        """
        FedDiv Eq(5): 聚合所有客户端的 GMM 参数
        严格处理组件排序对齐问题。
        """
        # 过滤掉 None (训练初期可能某些 Client 拟合失败)
        valid_gmms = [p for p in client_gmms_list if p is not None]
        if not valid_gmms:
            return None
            
        total_samples = sum([p['count'] for p in valid_gmms])
        
        # 取第一个有效的作为 shape 模板
        # gmm.covariances_ shape: (n_components, n_features, n_features) -> 通常是 (2, 1, 1)
        agg_means = np.zeros_like(valid_gmms[0]['means'])
        agg_covs = np.zeros_like(valid_gmms[0]['covariances'])
        agg_weights = np.zeros_like(valid_gmms[0]['weights'])
        
        for p in valid_gmms:
            weight = p['count'] / total_samples
            
            # Flatten 方便排序 (假设 n_features=1)
            means = p['means'].flatten()
            covs = p['covariances'].flatten()
            weights = p['weights'].flatten()
            
            # === 关键：对齐组件 ===
            # 假设 index 0 应该是 Clean (Mean 较小)，index 1 是 Noisy (Mean 较大)
            if means[0] > means[1]:
                means = means[::-1]
                covs = covs[::-1]
                weights = weights[::-1]
            
            # 还原形状以便累加
            # 【修复点】: 必须还原为与 agg_covs 完全相同的形状
            # 使用 .reshape(agg_covs.shape) 替代 .reshape(-1, 1)
            try:
                means = means.reshape(agg_means.shape)
                covs = covs.reshape(agg_covs.shape)
                weights = weights.reshape(agg_weights.shape)
            except ValueError as e:
                # 容错处理：如果形状确实无法 reshape (极少数情况)
                print(f"Reshape error: {e}. Skipping this client's GMM.")
                continue
            
            agg_means += weight * means
            agg_covs += weight * covs
            agg_weights += weight * weights
            
        self.global_gmm_params = {
            'means': agg_means,
            'covariances': agg_covs,
            'weights': agg_weights
        }
        return self.global_gmm_params

    def aggregate_weights_adaptive(self, selected_clients, noise_rates):
        """
        文档 Eq (4-10): 自适应聚合函数
        w_g = (m_i * e^{-eta_i}) / sum(...)
        """
        # 1. 归一化噪声率 eta_i = S_i / max(S)
        max_noise = max(noise_rates) if len(noise_rates) > 0 and max(noise_rates) > 1e-6 else 1.0
        # 防止除0
        if max_noise == 0: max_noise = 1.0
            
        normalized_noise = [nr / max_noise for nr in noise_rates]
        
        # 2. 计算聚合系数 coeff_i = m_i * exp(-eta_i)
        coeffs = []
        for i, client in enumerate(selected_clients):
            m_i = client.train_size
            eta = normalized_noise[i]
            coeffs.append(m_i * np.exp(-eta))
            
        total_coeff = sum(coeffs)
        
        # 3. 执行聚合
        # 先清空当前 global model 参数
        for k in self.W.keys():
            self.W[k].data.zero_()
            
        for i, client in enumerate(selected_clients):
            w_i = coeffs[i] / (total_coeff + 1e-12)
            for k in self.W.keys():
                self.W[k].data += client.W[k].data * w_i
                
    def distribute_masked_models(self, clients, reputations, global_update_list, server_params_old):
        """
        激励机制：根据 Reputation 计算 Mask，生成个性化模型分发给 Client
        """
        param_keys = list(self.model.state_dict().keys())
        
        # 计算 Mask 比例
        rs_min, rs_max = reputations.min(), reputations.max()
        rs_norm = (reputations - rs_min) / (rs_max - rs_min + 1e-9)
        # 贡献越低，Mask 比例越高 (被Mask掉的梯度越多) ? 
        # 文档: "鼓励客户...对客户进行奖励/惩罚"。 FairGraphFL 逻辑：Mask 掉越少越好。
        # q_ratio 是 mask percentile。reputation 高 -> q_ratio 小 (保留更多) 还是大？
        # 原代码：q_ratios = 0.3 + 0.7 * rs_norm. 如果 rs_norm 大，ratio 大。
        # mask_grad_update_by_order: keep top k (magnitude).
        # 如果 q_ratio 代表 "保留比例"：High Rep -> High Ratio -> Keep More. (合理)
        q_ratios = 0.3 + 0.7 * rs_norm 
        
        for i, client in enumerate(clients):
            mask_ratio = q_ratios[i].item() # 这是一个 [0.3, 1.0] 的值
            
            # 对 Global Update 进行 Mask (保留 Top-K magnitude)
            # copy deepcopy 很重要，否则会修改原始 list
            masked_update_list = utils.mask_grad_update_by_order(
                copy.deepcopy(global_update_list), mask_percentile=mask_ratio, mode='all'
            )
            
            # 重组为 state_dict: New_Client_Model = Old_Server_Param + Masked_Update
            client_new_dict = {}
            for idx, k in enumerate(param_keys):
                # 注意：server_params_old[k] 必须也是 float，如果原本是 long (如 num_batches_tracked) 需要小心
                if k in server_params_old:
                    update_val = masked_update_list[idx] if idx < len(masked_update_list) else 0
                    client_new_dict[k] = server_params_old[k] + update_val
                else:
                    # 某些 buffer 可能不在 update list 里
                    client_new_dict[k] = self.model.state_dict()[k]
            
            # 应用到客户端
            client.apply_personalized_update(client_new_dict)

# =============================================================================
#  Main Training Loop (Full Logic)
# =============================================================================

def run_full_logic(clients, server, args):
    rs = torch.ones(len(clients)) / len(clients) # 初始 Reputation
    
    # 记录上一轮 Server 参数，用于计算 Update
    server_params_old = copy.deepcopy(server.model.state_dict())
    
    frame = pd.DataFrame()
    print(f"Start Full Adaptive Noise FL Training with {len(clients)} clients on {args.device}")

    for round_idx in range(1, args.num_rounds + 1):
        print(f"\n--- Round {round_idx} ---")

        # ---------------------------------------------------------
        # 1. 模型分发 (Model Distribution)
        # ---------------------------------------------------------
        # 如果是第一轮，或者没有激励机制介入，则进行标准的广播
        # 如果是后续轮次，且启用了激励机制，Clients 已经在上一轮末尾接收了 Masked Model。
        # 这里我们需要做一个判断。
        # 为了严谨：第一轮显式下载。后续轮次依赖上一轮的 distribute。
        if round_idx == 1:
            for client in clients:
                client.download_from_server(server)
        
        # ---------------------------------------------------------
        # 2. 全局噪声滤波器构建 (Global Noise Filter)
        # ---------------------------------------------------------
        # 预热期 (Warm-up): 前几轮 Loss 不稳定，跳过 GMM 筛选，视为全 Clean
        if round_idx > args.lid_start_round: 
            client_gmms = []
            for client in clients:
                gmm_p = client.fit_local_gmm()
                client_gmms.append(gmm_p)
            
            global_gmm = server.aggregate_gmm(client_gmms)
            
            if global_gmm:
                print(f"  > Global GMM Means: {global_gmm['means'].flatten()}")
                # 更新客户端噪声信息
                client_noise_rates = []
                for client in clients:
                    client.update_noise_info(global_gmm)
                    client_noise_rates.append(client.noise_rate)
                
                avg_noise = np.mean(client_noise_rates)
                print(f"  > Avg Noise Rate: {avg_noise:.4f}")
            else:
                print("  > GMM Aggregation skipped (not enough valid local fits).")
                client_noise_rates = [0.0] * len(clients) # Default
        else:
            print("  > Warm-up phase: Skipping Noise Filter.")
            client_noise_rates = [0.0] * len(clients)

        # ---------------------------------------------------------
        # 3. 贡献评估与声誉更新 (Valuation & Reputation)
        # ---------------------------------------------------------
        # 必须先由 Client 计算指标 (需要一次 Forward pass 甚至 update prototype)
        # 文档公式需要: 梯度对齐 (Phi), 噪声率 (P -> 1-Noise), 类别多样性 (C)
        
        phis = torch.zeros(len(clients))
        divs = torch.zeros(len(clients))
        inv_noises = torch.zeros(len(clients))
        
        for i, client in enumerate(clients):
            # 更新原型 (用于 Cosine Sim)
            client.prototype_update()
            
            # 计算 Cosine Similarity
            sim_val = client.cosine_similar(server)
            phis[i] = float(sim_val)
            
            # 计算 Diversity
            divs[i] = client.calc_class_diversity()
            
            # 计算 1 - NoiseRate (噪声越低越好)
            inv_noises[i] = 1.0 - client.noise_rate
            
        # Server 聚合原型 (为下一轮准备)
        server.reput_aggregate_prototype(rs, clients)
        
        # 归一化各指标以便加权
        def normalize_tensor(t):
            if t.max() == t.min(): return torch.zeros_like(t)
            return (t - t.min()) / (t.max() - t.min())

        phis_norm = normalize_tensor(phis)
        divs_norm = normalize_tensor(divs)
        noises_norm = normalize_tensor(inv_noises)
        
        # 声誉更新公式: r_t = r_{t-1} + alpha * (phi + alpha2 * P + alpha3 * C)
        # 参数设定 (根据文档隐含逻辑或经验)
        alpha_main = 0.1
        w_phi, w_noise, w_div = 1.0, 1.0, 1.0
        
        score_update = w_phi * phis_norm + w_noise * noises_norm + w_div * divs_norm
        
        rs = rs + alpha_main * score_update
        # 归一化 Reputation
        rs = torch.clamp(rs, min=1e-3)
        rs = torch.div(rs, rs.sum())
        
        print(f"  > Reputations (Top): {[round(x, 3) for x in rs.tolist()[:5]]}...")

        # ---------------------------------------------------------
        # 4. 适应性本地训练 (Adaptive Local Training)
        # ---------------------------------------------------------
        for client in clients:
            client.local_train_adaptive(args.local_epoch)

        # ---------------------------------------------------------
        # 5. 适应性聚合 (Adaptive Aggregation)
        # ---------------------------------------------------------
        # 使用基于噪声率的权重 (w_g) 更新 Server 模型
        server.aggregate_weights_adaptive(clients, client_noise_rates)
        
        # ---------------------------------------------------------
        # 6. 激励机制与闭环分发 (Incentive & Distribution)
        # ---------------------------------------------------------
        # 计算全局更新量 (Global Update = New Server - Old Server)
        current_server_state = server.model.state_dict()
        global_update_dict = {}
        
        # 过滤掉 Long Tensor (如 num_batches_tracked)，只计算 Float 参数的更新
        for k, v in current_server_state.items():
            if v.dtype == torch.float:
                if k in server_params_old:
                     global_update_dict[k] = v - server_params_old[k]
                else:
                     global_update_dict[k] = torch.zeros_like(v)
        
        param_keys = list(global_update_dict.keys())
        global_update_list = [global_update_dict[k] for k in param_keys]
        
        # 生成 Masked 模型并应用到 Clients (为下一轮做准备)
        server.distribute_masked_models(clients, rs, global_update_list, server_params_old)
        
        # 更新 Server Old Params 为当前状态
        server_params_old = copy.deepcopy(current_server_state)
        
        # ---------------------------------------------------------
        # 7. 评估 (Evaluation)
        # ---------------------------------------------------------
        if round_idx % 1 == 0:
            server.model.eval()
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                # 使用测试集评估
                # 这里假设所有 Client 共享同一个 testset (setupCV.py 逻辑)
                # 为了速度，只测第一个 Client 的 loader 即可
                for data, target in clients[0].test_loader:
                    data, target = data.to(args.device), target.to(args.device)
                    output, _, _ = server.model(data)
                    pred = output.argmax(dim=1)
                    total_correct += pred.eq(target).sum().item()
                    total_samples += target.size(0)
            
            acc = 100. * total_correct / total_samples
            print(f"  > Global Model Acc: {acc:.2f}%")
            frame.loc[round_idx, 'acc'] = acc
            
            # (可选) 保存每一轮的结果
            # frame.to_csv(os.path.join(args.outbase, 'acc_log.csv'))

    return frame

# =============================================================================
#  Setup & Entry
# =============================================================================

def setup_full_environment(args):
    trainset, testset = setupCV.get_cifar10(args.datapath)
    client_indices = setupCV.split_noniid(trainset, args.num_clients, alpha=args.alpha)
    
    print("Injecting noise (FedCorr Style)...")
    y_train_original = np.array(trainset.targets)
    y_train_noisy, _, _ = setupCV.add_noise_fedcorr(args, y_train_original, client_indices)
    
    clients = []
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data, targets):
            self.data = data
            self.targets = targets
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i], self.targets[i]

    for i in range(args.num_clients):
        subset_indices = client_indices[i]
        # data list to tensor stack
        data_list = [trainset[idx][0] for idx in subset_indices]
        # target list
        target_list = y_train_noisy[subset_indices].tolist()
        
        client_dataset = SimpleDataset(data_list, target_list)
        
        model = CNNCifar(args)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        
        client = Client_New(model, i, f"client{i}", len(client_dataset), client_dataset, testset, optimizer, args)
        clients.append(client)
        
    server_model = CNNCifar(args)
    server = Server_New(server_model, args.device)
    
    return clients, server

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--outbase', type=str, default='./outputs')
    
    # Training Params
    parser.add_argument('--num_rounds', type=int, default=50)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5) # Non-IID degree
    
    # Noise Params
    parser.add_argument('--level_n_system', type=float, default=0.3) # 30% clients are noisy
    parser.add_argument('--level_n_lowerb', type=float, default=0.5) # Noise ratio lower bound
    
    # Logic Control
    parser.add_argument('--lid_start_round', type=int, default=5, help='Warmup rounds before GMM noise filter')
    
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    Path(args.outbase).mkdir(parents=True, exist_ok=True)
    
    print("Setting up Full Environment...")
    clients, server = setup_full_environment(args)
    
    res = run_full_logic(clients, server, args)
    
    res.to_csv(os.path.join(args.outbase, 'accuracy_full.csv'))
    print("Training Finished.")