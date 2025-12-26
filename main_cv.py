import os
import argparse
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import copy

# 引入之前的 setupCV
import setupCV 

def run_fair_cv(clients, server, args):
    """
    适配 CIFAR-10 的 FairGraphFL 训练主循环
    逻辑：
    1. 本地训练
    2. 计算类原型 (Class Prototypes)
    3. 服务器基于 Reputation 聚合全局原型
    4. 计算相似度并更新 Reputation
    5. 基于 Reputation 聚合模型权重 (Weighted Aggregation)
    """
    
    # 初始化 Reputation，初始时均匀分布
    rs = torch.ones(len(clients)) / len(clients)
    
    # 记录结果
    frame = pd.DataFrame()
    
    print(f"Start Training with {len(clients)} clients on {args.device}")

    for round_idx in range(1, args.num_rounds + 1):
        print(f"\n--- Round {round_idx} ---")
        
        # 1. 客户端下载最新的全局模型
        # 注意：每一轮开始前，根据上一轮计算出的 rs 进行加权聚合后的模型被分发
        for client in clients:
            client.download_from_server(server)
        
        # 2. 本地训练 & 计算原型
        # 为了提高效率，并非所有客户端每轮都必须被选中，这里演示全量参与(Full participation)
        # 如果客户端较多，可以采样 selected_clients
        selected_clients = clients 
        
        for i, client in enumerate(selected_clients):
            # 本地训练
            client.local_train(args.local_epoch)
            
            # 更新本地类原型 (计算特征均值)
            client.prototype_update()
            
            # 简单打印一下进度
            if i % 5 == 0:
                print(f"  > Client {client.id} training done.")

        # 3. 服务器聚合全局原型 (Reputation-based)
        # 第一轮 rs 是均匀的，后续轮次 rs 会根据相似度变化
        if round_idx == 1:
            # 第一轮也可以直接简单平均
            server.reput_aggregate_prototype(rs, selected_clients)
        else:
            server.reput_aggregate_prototype(rs, selected_clients)
            
        # 4. 计算相似度并更新 Reputation
        phis = torch.zeros(len(selected_clients))
        for i, client in enumerate(selected_clients):
            # 计算本地原型与全局原型的余弦相似度
            phis[i] = client.cosine_similar(server)
        
        # 平滑更新 Reputation
        # alpha=0.05 是原项目的默认动量参数
        rs = 0.95 * rs + 0.05 * phis
        
        # 确保 rs 为正数且归一化
        rs = torch.clamp(rs, min=1e-3)
        rs = torch.div(rs, rs.sum())
        
        print(f"  > Current Reputations (First 5): {rs[:5].tolist()}")
        
        # 5. 更新全局模型权重 (核心：基于 Reputation 加权聚合)
        # 严格遵循 FairGraphFL 思想：Reputation 高的模型权重占比更大
        
        global_dict = {}
        server_params = server.model.state_dict()
        
        # 初始化
        for k in server_params.keys():
            global_dict[k] = torch.zeros_like(server_params[k])
            
        # 加权累加
        for i, client in enumerate(selected_clients):
            client_params = client.model.state_dict()
            weight = rs[i].item() # 使用 Reputation 作为聚合权重
            
            for k in global_dict.keys():
                global_dict[k] += client_params[k] * weight
                
        # 更新服务器模型
        server.model.load_state_dict(global_dict)
        # 同步 server.W，以便下一轮分发
        server.W = {key: value for key, value in server.model.named_parameters()}
        
        # 6. 清理缓存的原型 (下一轮重新计算)
        # for client in selected_clients:
        #    client.clear_prototype() # 如果 Client 类里有这个方法可以调用，没有也不影响覆盖
        
        # 7. 测试评估 (使用测试集评估每个客户端的准确率)
        # 也可以只评估 Global Model 在 Test Set 上的表现，这里按原项目习惯评估每个客户端
        if round_idx % 1 == 0:
            test_accs = []
            for client in clients:
                # 评估前先确保客户端模型是最新的(聚合后的)，或者评估聚合前的？
                # 通常评估的是聚合后的全局模型在各个客户端数据分布上的表现
                # 这里我们让客户端加载刚刚聚合好的全局模型进行评估
                client.download_from_server(server)
                _, acc = client.model.eval(), 0
                
                # 简单的 evaluate 实现
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
            print(f"  > Round {round_idx} Average Test Acc: {avg_acc:.2f}%")

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