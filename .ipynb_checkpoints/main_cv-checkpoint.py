import os
import argparse
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import math
from sklearn.mixture import GaussianMixture 
import setupCV
import fl_utils as utils
from gateway import Gateway

def run_fair_fedcorr(clients, server, args):
    print("Blockchain Gateway initialized.")

    """
    FairGraphFL + FedCorr 融合训练流程
    """
    rs = torch.ones(len(clients)) / len(clients)
    server_params_old = copy.deepcopy(server.model.state_dict())
    frame = pd.DataFrame()
    
    # 记录每个客户端累积的 LID 分数
    lid_accumulative = np.zeros(len(clients))
    
    print(f"Start FairGraphFL + FedCorr Training with {len(clients)} clients on {args.device}")

    for round_idx in range(1, args.num_rounds + 1):
        gateway = Gateway("http://localhost:8080")
        print(f"\n--- Round {round_idx} ---")

        # ---------------------------------------------------------
        # 1. 激励分发 (FairGraphFL)
        # ---------------------------------------------------------
        if round_idx == 1:
            for client in clients:
                client.download_from_server(server)
        
        # ---------------------------------------------------------
        # 2. 本地训练 & LID 收集 (FedCorr)
        # ---------------------------------------------------------
        selected_clients = clients
        for i, client in enumerate(selected_clients):
            client.local_train(args.local_epoch)
            client.prototype_update() # FairGraphFL 需要原型
            
            # === FedCorr: 收集 LID ===
            # 【这里控制 LID 开始收集的时间】
            if round_idx >= args.lid_start_round:
                avg_lid, _ = client.compute_lid_score()
                lid_accumulative[i] += avg_lid

            # <--- 【新增3】 缓存 LID 到 client 对象
                client.lid_score_cache = float(avg_lid)

        # ---------------------------------------------------------
        # 3. FedCorr 噪声检测与修正逻辑
        # ---------------------------------------------------------
        # 【这里控制执行清洗的时间】
        if round_idx == args.correction_round:
            print("\n*** Executing FedCorr Noise Correction ***")
            
            # (1) 使用 GMM 将客户端分为 Clean 和 Noisy
            lid_array = lid_accumulative.reshape(-1, 1)
            # 如果 LID 还没收集到数据（比如 lid_start_round 设得太晚），这里会报错，所以要确保有数据
            if np.all(lid_array == 0):
                print("Warning: No LID scores collected yet! Check lid_start_round.")
            else:
                gmm = GaussianMixture(n_components=2, random_state=args.seed).fit(lid_array)
                labels = gmm.predict(lid_array)
                
                # LID 较小的组是 Clean 的
                clean_label = np.argsort(gmm.means_[:, 0])[0]
                
                noisy_clients_idx = np.where(labels != clean_label)[0]
                clean_clients_idx = np.where(labels == clean_label)[0]
                
                print(f"  > Detected Noisy Clients: {noisy_clients_idx}")
                print(f"  > Detected Clean Clients: {clean_clients_idx}")
                # <--- 【新增4】 更新客户端的 is_noisy 状态
                for idx, client in enumerate(clients):
                    if idx in noisy_clients_idx:
                        client.is_noisy = True
                    else:
                        client.is_noisy = False
                
                # (2) 对噪声客户端进行标签修正
                for idx in noisy_clients_idx:
                    client = clients[idx]
                    _, loss_whole = client.compute_lid_score()
                    
                    # 在 Loss 上再跑一次 GMM 估算该客户端的噪声比例
                    gmm_loss = GaussianMixture(n_components=2, random_state=args.seed).fit(loss_whole.reshape(-1, 1))
                    labels_loss = gmm_loss.predict(loss_whole.reshape(-1, 1))
                    clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]
                    
                    # 预测为噪声样本的数量
                    pred_noisy_count = np.sum(labels_loss != clean_label_loss)
                    estimated_noise_level = pred_noisy_count / len(loss_whole)
                    
                    print(f"  > Client {idx} estimated noise level: {estimated_noise_level:.2f}")
                    
                    # 执行修正
                    client.correct_labels(server, estimated_noise_level)

        # ---------------------------------------------------------
        # 4. FairGraphFL 聚合与激励
        # ---------------------------------------------------------
        server.reput_aggregate_prototype(rs, selected_clients)
        
        # 计算相似度更新声誉
        phis = torch.zeros(len(selected_clients))
        for i, client in enumerate(selected_clients):
            # 计算并缓存 Cosine Sim (虽然当前合约未上链此字段，但保留是个好习惯)
            sim_val = client.cosine_similar(server)
            phis[i] = sim_val
            # 【新增】缓存到 client 对象，以备 Gateway 或后续扩展使用
            if isinstance(sim_val, torch.Tensor):
                client.cosine_sim_cache = sim_val.item()
            else:
                client.cosine_sim_cache = float(sim_val)
            
        rs = 0.95 * rs + 0.05 * phis
        rs = torch.clamp(rs, min=1e-3)
        rs = torch.div(rs, rs.sum())
        
        # 打印 Reputation
        print(f"  > Reputations: {[round(x, 4) for x in rs.tolist()]}")
        
        # 全局模型加权聚合
        # 注意：这里加了强制 float 转换，防止 ResNet 的 Long 类型参数报错
        global_dict = {}
        for k in server_params_old.keys():
            global_dict[k] = torch.zeros_like(server_params_old[k], dtype=torch.float)
        
        for i, client in enumerate(selected_clients):
            client_params = client.model.state_dict()
            weight = rs[i].item()
            for k in global_dict.keys():
                global_dict[k] += client_params[k] * weight
        
        # 转回 Long 类型 (如果原始参数是 Long)
        for k in global_dict.keys():
            if server_params_old[k].dtype == torch.long:
                global_dict[k] = global_dict[k].long()
                
        # 计算全局更新量
        global_update_dict = {}
        for k in global_dict.keys():
            global_update_dict[k] = global_dict[k] - server_params_old[k]
        param_keys = list(global_update_dict.keys())
        global_update_list = [global_update_dict[k] for k in param_keys]
        
        # 更新 Server 模型
        server.model.load_state_dict(global_dict)
        
        # 计算激励比例 (Incentive Ratios)
        # 加入 Warm-up：前2轮不开启激进惩罚，让模型先学好特征
        rs_min = rs.min()
        rs_max = rs.max()
        rs_norm = (rs - rs_min) / (rs_max - rs_min + 1e-9)
        
        if round_idx <= 2:
            q_ratios = torch.ones_like(rs)
            print(f"  > [Warm-up Phase] Incentive mechanism disabled")
        else:
            # 激进惩罚策略
            q_ratios = 0.3 + 0.7 * rs_norm 
            print(f"  > Incentive Ratios: {[round(x, 4) for x in q_ratios.tolist()]}")
        
        # 分发掩码后的梯度
        for i, client in enumerate(selected_clients):
            mask_ratio = q_ratios[i].item()
            masked_update_list = utils.mask_grad_update_by_order(
                copy.deepcopy(global_update_list), mask_percentile=mask_ratio, mode='all'
            )
            client_new_dict = {}
            for idx, k in enumerate(param_keys):
                client_new_dict[k] = server_params_old[k] + masked_update_list[idx]
            client.model.load_state_dict(client_new_dict)
            client.W = {key: value for key, value in client.model.named_parameters()}
            
        server_params_old = copy.deepcopy(server.model.state_dict())

        try:
            gateway.upload_round_data(
                round_idx=round_idx,
                server=server,
                clients=clients,
                reputations=rs.tolist(),
                incentive_ratios=q_ratios.tolist()
            )
        except Exception as e:
            print(f"[Warning] Blockchain upload failed: {e}")

        # ---------------------------------------------------------
        # 5. 评估
        # ---------------------------------------------------------
        if round_idx % 1 == 0:
            test_accs = []
            print(f"  > Round {round_idx} Client Accuracies:")
            for i, client in enumerate(clients):
                client.model.eval()
                correct = 0; total = 0
                with torch.no_grad():
                    for data, target in client.test_loader:
                        data, target = data.to(args.device), target.to(args.device)
                        output, _, _ = client.model(data)
                        pred = output.argmax(dim=1)
                        correct += pred.eq(target).sum().item()
                        total += target.size(0)
                acc = 100. * correct / total
                test_accs.append(acc)
                print(f"    Client {i}: {acc:.2f}%")
            print(f"  > Avg Acc: {np.mean(test_accs):.2f}%")


    return frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 基础设置
    parser.add_argument('--device', type=str, default='cuda', help='cpu/cuda')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--outbase', type=str, default='./outputs')
    
    # 训练参数
    parser.add_argument('--num_rounds', type=int, default=100)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)

    # FedCorr 噪声参数
    parser.add_argument('--level_n_system', type=float, default=0.3)
    parser.add_argument('--level_n_lowerb', type=float, default=0.5)
    
    # === 关键：FedCorr 控制参数 ===
    parser.add_argument('--lid_start_round', type=int, default=10, 
                        help='Start collecting LID scores from this round (warm-up end)')
    parser.add_argument('--correction_round', type=int, default=20, 
                        help='Round to perform label correction')
    
    parser.add_argument('--relabel_ratio', type=float, default=0.5)
    parser.add_argument('--confidence_thres', type=float, default=0.8)

    args = parser.parse_args()

    # 设备检查
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed(args.seed)

    print(f"Running on {args.device}")
    
    outpath = os.path.join(args.outbase, f'cifar10_clients{args.num_clients}_alpha{args.alpha}_fedcorr')
    Path(outpath).mkdir(parents=True, exist_ok=True)

    print("Setting up devices and data (CIFAR-10)...")
    clients, server = setupCV.setup_devices_cv(args.datapath, args)
    
    # 注意这里调用的是 run_fair_fedcorr
    print("Starting FairGraphFL + FedCorr training...")
    result_frame = run_fair_fedcorr(clients, server, args)
    
    outfile = os.path.join(outpath, 'accuracy.csv')
    result_frame.to_csv(outfile)
    print(f"Training finished. Results saved to {outfile}")