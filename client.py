import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
from torch import nn

class Client_CV():
    def __init__(self, model, client_id, client_name, train_size, dataset_train, dataset_test, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.optimizer = optimizer
        self.args = args
        
        # Dataloaders
        self.train_loader = DataLoader(self.dataset_train, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.dataset_test, batch_size=args.batch_size, shuffle=False)
        # 用于计算原型的 loader (不 shuffle)
        self.proto_loader = DataLoader(self.dataset_train, batch_size=args.batch_size, shuffle=False)

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

        # 核心变量替换：Motif -> Class
        self.prototype = {} # key: class_label (0-9), value: embedding tensor
        self.simi = {}
        self.rs = {} # Reputation score
        
    def prototype_update(self):
        """
        计算本地数据的类原型 (Class Prototypes)
        对应原项目的 prototype_update，但这里是计算每个类别的特征均值
        """
        self.model.eval()
        features_sum = {}
        counts = {}
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.proto_loader):
                data, target = data.to(self.args.device), target.to(self.args.device)
                _, feature, _ = self.model(data)
                
                # 按类别聚合特征
                for i in range(len(target)):
                    label = target[i].item()
                    emb = feature[i]
                    if label not in features_sum:
                        features_sum[label] = emb
                        counts[label] = 1
                    else:
                        features_sum[label] += emb
                        counts[label] += 1
        
        # 计算均值并归一化
        self.prototype = {}
        for label in features_sum.keys():
            avg_proto = features_sum[label] / counts[label]
            # 归一化，方便计算余弦相似度
            self.prototype[label] = avg_proto / max(torch.norm(avg_proto), 1e-12)

        print(f'Client {self.id}: Finish constructing class prototypes! Classes found: {list(self.prototype.keys())}')

    def cosine_similar(self, server):
        """
        贡献评估函数：计算本地类原型与全局类原型的余弦相似度。
        这反映了：
        1. 噪声率：如果本地有很多噪声，类原型会偏离全局方向，相似度低。
        2. 类别多样性：如果缺少某些类，这里就不会计算该类的相似度（或者在Server端聚合时权重低）。
        """
        similarity = []
        valid_keys = []
        
        for key in self.prototype.keys():
            if key in server.global_prototype:
                # key 是类别索引 (0-9)
                sim = F.cosine_similarity(self.prototype[key].unsqueeze(0), 
                                          server.global_prototype[key].unsqueeze(0), 
                                          dim=1, eps=1e-10)
                self.simi[key] = sim.item()
                similarity.append(sim.item())
                valid_keys.append(key)
        
        if len(similarity) == 0:
            reput = 0.0
        else:
            reput = np.mean(similarity)

        # 更新 Reputation Score (平滑更新)
        if len(self.rs) == 0:
            for key in valid_keys:
                self.rs[key] = 0.05 * self.simi[key]
        else:
            for key in valid_keys:
                if key in self.rs:
                    self.rs[key] = 0.95 * self.rs[key] + 0.05 * self.simi[key]
                else:
                    self.rs[key] = 0.05 * self.simi[key]

        # 确保 rs 不为负或过小
        for key in self.rs.keys():
            if isinstance(self.rs[key], torch.Tensor):
                 self.rs[key] = torch.clamp(self.rs[key], min=1e-3).item()
            else:
                 self.rs[key] = max(self.rs[key], 1e-3)
                 
        return reput

    def download_from_server(self, server):
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()

    def local_train(self, local_epoch):
        """ 标准的本地训练 """
        self.model.train()
        for epoch in range(local_epoch):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.args.device), target.to(self.args.device)
                self.optimizer.zero_grad()
                pred, _, _ = self.model(data)
                loss = self.model.loss(pred, target)
                loss.backward()
                self.optimizer.step()
        
        # 简单的权重差分更新计算
        for k in self.W:
            self.dW[k] = self.W[k].data - self.W_old[k].data