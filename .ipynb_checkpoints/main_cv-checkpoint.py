import os
import argparse
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import math

import setupCV
import utils  # 确保导入了我们精简后的 utils.py

def run_fair_cv(clients, server, args):
    """
    FairGraphFL-CV (With Incentive Mechanism)
    """
    # 初始化 Reputation
    rs = torch.ones(len(clients)) / len(clients)
    
    # 记录上一轮的全局模型参数 (用于计算梯度更新量)
    # W_global_old
    server_params_old = copy.deepcopy(server.model.state_dict())
    
    frame = pd.DataFrame()
    print(f"Start Training with {len(clients)} clients on {args.device}")

    for round_idx in range(1, args.num_rounds + 1):
        print(f"\n--- Round {round_idx} ---")
        
        # ---------------------------------------------------------
        # 1. 客户端下载模型 (Incentive 分发后的模型)
        # ---------------------------------------------------------
        # 注意：这里不再是简单的 client.download_from_server(server)
        # 因为每个 Client 获得的模型可能不一样（被掩码过）
        # 我们在上一轮结尾已经把处理好的参数赋值给了 client.model，所以这里不需要额外操作
        # 只有第一轮需要初始化
        if round_idx == 1:
            for client in clients:
                client.download_from_server(server)

        # ---------------------------------------------------------
        # 2. 本地训练 & 计算原型
        # ---------------------------------------------------------
        selected_clients = clients
        for i, client in enumerate(selected_clients):
            client.local_train(args.local_epoch)
            client.prototype_update()
            
        # ---------------------------------------------------------
        # 3. 服务器聚合原型 & 更新 Reputation
        # ---------------------------------------------------------
        # 聚合原型
        if round_idx == 1:
            server.reput_aggregate_prototype(rs, selected_clients)
        else:
            server.reput_aggregate_prototype(rs, selected_clients)
            
        # 计算相似度并更新 Reputation
        phis = torch.zeros(len(selected_clients))
        for i, client in enumerate(selected_clients):
            phis[i] = client.cosine_similar(server)
        
        # 动量更新
        rs = 0.95 * rs + 0.05 * phis
        rs = torch.clamp(rs, min=1e-3)
        rs = torch.div(rs, rs.sum())
        
        print(f"  > Current Reputations (First 5): {rs[:5].tolist()}")

        # ---------------------------------------------------------
        # 4. 计算全局模型更新 (Global Update)
        # ---------------------------------------------------------
        # 先计算这一轮理想的全局模型 (W_global_new)
        global_dict = {}
        # 初始化
        for k in server_params_old.keys():
            global_dict[k] = torch.zeros_like(server_params_old[k])
            
        # 加权聚合 (Aggregation)
        for i, client in enumerate(selected_clients):
            client_params = client.model.state_dict()
            weight = rs[i].item()
            for k in global_dict.keys():
                global_dict[k] += client_params[k] * weight
        
        # 计算全局更新量 (Gradient): Delta = W_new - W_old
        # 这里我们将参数变化量视为"梯度"
        global_update_dict = {}
        for k in global_dict.keys():
            global_update_dict[k] = global_dict[k] - server_params_old[k]
            
        # 将 Dict 转换为 list 格式，方便 utils 处理
        # 顺序必须固定，使用 keys() 排序或固定顺序
        param_keys = list(global_update_dict.keys())
        global_update_list = [global_update_dict[k] for k in param_keys]

        # 更新服务器的物理模型 (用于评估和下一轮基准)
        server.model.load_state_dict(global_dict)
        
        # ---------------------------------------------------------
        # 5. 激励机制 (Incentive): 梯度掩码分发
        # ---------------------------------------------------------
        # 这里的逻辑是：Client_Next = Server_Old + Mask(Delta)
        # 声誉越高的客户端，得到的 Delta 越完整；声誉低的，得到的 Delta 越稀疏。
        
        # 计算掩码比例 q_ratios
        # 使用 tanh 函数将声誉映射到 (0, 1] 区间，声誉越高 q 越大
        # 系数 5.0 是一个超参数，控制激励的敏感度，可以调整
        # rs 是归一化的，数值较小 (e.g. 0.1)，所以需要乘一个系数让它在 tanh 上有区分度
        q_ratios = torch.tanh(5.0 * rs) 
        q_ratios = q_ratios / torch.max(q_ratios) # 归一化，让最好的客户端获得 1.0 (100%更新)
        
        print(f"  > Incentive Ratios (First 5): {q_ratios[:5].tolist()}")

        for i, client in enumerate(selected_clients):
            # 获取该客户端的保留比例
            mask_ratio = q_ratios[i].item()
            
            # 对全局更新量进行掩码 (只保留幅值最大的前 mask_ratio 部分)
            # 注意：mask_grad_update_by_order 会修改传入的 list，所以要深拷贝
            masked_update_list = utils.mask_grad_update_by_order(
                copy.deepcopy(global_update_list), 
                mask_percentile=mask_ratio, 
                mode='all' # 全局掩码模式，也可以选 'layer' 层级掩码
            )
            
            # 构造客户端的新参数: W_client_new = W_server_old + Masked_Delta
            client_new_dict = {}
            for idx, k in enumerate(param_keys):
                # 恢复参数形状并相加
                update_tensor = masked_update_list[idx]
                client_new_dict[k] = server_params_old[k] + update_tensor
            
            # 将新参数加载给客户端，供下一轮训练使用
            client.model.load_state_dict(client_new_dict)
            # 同步 W (用于优化器等)
            client.W = {key: value for key, value in client.model.named_parameters()}

        # ---------------------------------------------------------
        # 6. 更新基准模型 (Old Server Model)
        # ---------------------------------------------------------
        server_params_old = copy.deepcopy(server.model.state_dict())

        # ---------------------------------------------------------
        # 7. 测试评估
        # ---------------------------------------------------------
        if round_idx % 1 == 0:
            test_accs = []
            # 评估服务器模型（最完美的模型）
            # 或者评估各个客户端手中的模型（被掩码过的模型）
            # 原论文通常评估服务器模型，但为了看激励效果，我们可以看客户端模型的精度
            
            # 这里演示评估每个客户端自己的模型性能
            for client in clients:
                client.model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in client.test_loader:
                        data, target = data.to(args.device), target.to(args.device)
                        output, _, _ = client.model(data)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                acc = 100. * correct / total
                test_accs.append(acc)
                frame.loc[client.name, f'round_{round_idx}'] = acc
            
            avg_acc = np.mean(test_accs)

    return frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 基础设置
    parser.add_argument('--device', type=str, default='cuda', help='cpu/cuda')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--datapath', type=str, default='./data', help='Data directory')
    parser.add_argument('--outbase', type=str, default='./outputs')
    
    # 训练参数
    parser.add_argument('--num_rounds', type=int, default=100, help='Total communication rounds')
    parser.add_argument('--local_epoch', type=int, default=5, help='Local training epochs per round')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    
    # 联邦学习设置
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    
    # FairGraphFL / CV 特定参数
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='Dirichlet distribution alpha for Non-IID setting (smaller = more non-iid)')
    # 注意：noise_rate 的逻辑目前硬编码在 setupCV 中（前20%客户端高噪声），
    # 如果需要动态调整，可以修改 setupCV 使其接受参数。
    
    parser.add_argument('--level_n_system', type=float, default=0.3, 
                        help='Fraction of noisy clients (e.g. 0.3 means 30% clients have noise)')
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, 
                        help='Lower bound of noise level (e.g. 0.5 means noise ratio is between 0.5 and 1.0)')

    args = parser.parse_args()

    # 自动检测设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed(args.seed)

    print(f"Running on {args.device}")
    
    # 输出路径
    outpath = os.path.join(args.outbase, f'cifar10_clients{args.num_clients}_alpha{args.alpha}')
    Path(outpath).mkdir(parents=True, exist_ok=True)

    # 1. 设置数据和设备 (使用新的 setupCV)
    print("Setting up devices and data (CIFAR-10)...")
    # setup_devices_cv 返回 clients, server
    clients, server = setupCV.setup_devices_cv(args.datapath, args)
    
    # 2. 开始 FairGraphFL 流程 (CV版)
    print("Starting FairGraphFL-CV training...")
    result_frame = run_fair_cv(clients, server, args)
    
    # 3. 保存结果
    outfile = os.path.join(outpath, 'accuracy.csv')
    result_frame.to_csv(outfile)
    print(f"Training finished. Results saved to {outfile}")