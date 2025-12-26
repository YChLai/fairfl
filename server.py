import torch
import numpy as np
import random

class Server():
    def __init__(self, model, device):
        self.model = model.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.device = device
        
        # 这里的 key 从 motif 变为 class label (int)
        self.global_prototype = {} 
        self.weight = {} # 用于记录每个类别的聚合权重（基于客户端的 reputation）

    def aggregate_weights(self, selected_clients):
        # 经典的 FedAvg / 加权聚合
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        
        for k in self.W.keys():
            weighted_sum = torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients])
            self.W[k].data = torch.div(torch.sum(weighted_sum, dim=0), total_size).clone()

    def reput_aggregate_prototype(self, rs_list, clients):
        """
        基于 Reputation 聚合全局原型。
        rs_list: 对应 clients 的 reputation 分数列表 (虽然原代码这里的参数逻辑有点混，这里简化处理)
        实际上原代码里 clients 自带 self.rs (每个motif/class的reputation)
        """
        self.global_prototype = {}
        self.weight = {}

        for client in clients:
            for key in client.prototype.keys():
                # key 是 class label
                # 获取该客户端在该类别上的 reputation (权重)
                # 如果没有记录，默认为一个较小值
                w = client.rs.get(key, 1e-3)

                if key not in self.global_prototype:
                    self.global_prototype[key] = w * client.prototype[key]
                    self.weight[key] = w
                else:
                    self.global_prototype[key] += w * client.prototype[key]
                    self.weight[key] += w
        
        # 平均化
        for key in self.global_prototype.keys():
            self.global_prototype[key] /= self.weight[key]
            self.global_prototype[key] = self.global_prototype[key].data
            
    def update_reput(self, clients):
        # 原项目中的归一化操作
        for key in self.global_prototype.keys():
            weight_sum = 0
            for client in clients:
                if key in client.rs:
                    weight_sum += client.rs[key]
            
            if weight_sum > 0:
                for client in clients:
                    if key in client.rs:
                        client.rs[key] /= weight_sum