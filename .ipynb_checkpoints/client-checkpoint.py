import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
from torch import nn
import utils # 确保导入修改后的 utils

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
        # 用于计算原型和LID的 loader (不 shuffle)
        self.proto_loader = DataLoader(self.dataset_train, batch_size=args.batch_size, shuffle=False)

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

        # 核心变量
        self.prototype = {} 
        self.simi = {}
        self.rs = {} 
        
    # --- FedCorr 新增功能 ---
    def compute_lid_score(self):
        """
        计算本地数据的 LID 分数和 Loss
        """
        criterion = nn.NLLLoss(reduction='none') # 因为模型输出是 log_softmax
        # 使用本地模型在本地数据上的输出计算 LID
        output_whole, loss_whole = utils.get_output(self.proto_loader, self.model, self.args, criterion)
        
        # 计算 LID
        lid_local = utils.lid_term(output_whole, output_whole)
        
        # 返回平均 LID 和所有样本的 Loss (用于后续 GMM 估算噪声率)
        return np.mean(lid_local), loss_whole

    def correct_labels(self, server, estimated_noise_level):
        """
        使用全局模型修正标签 (Label Correction)
        """
        if estimated_noise_level < 0.01:
            return 0

        # 下载最新的全局模型用于预测
        temp_model = deepcopy(server.model).to(self.args.device)
        temp_model.eval()
        
        criterion = nn.NLLLoss(reduction='none')
        # 获取全局模型在本地数据上的输出
        glob_output, loss_glob = utils.get_output(self.proto_loader, temp_model, self.args, criterion)
        
        # 筛选出 Loss 最大的那部分样本 (认为是噪声)
        num_samples = len(glob_output)
        num_noise = int(num_samples * estimated_noise_level * self.args.relabel_ratio)
        
        if num_noise == 0:
            return 0
            
        # 找出 Loss 最高的 top-K 索引
        high_loss_indices = np.argsort(loss_glob)[-num_noise:]
        
        # 找出全局模型预测置信度高的样本
        # glob_output 是概率分布 (exp后)
        confidence_indices = np.where(np.max(glob_output, axis=1) > self.args.confidence_thres)[0]
        
        # 取交集：既是高 Loss (疑似噪声)，又是全局模型确信能纠正的
        correction_indices = list(set(high_loss_indices) & set(confidence_indices))
        
        # 执行修正
        count = 0
        # dataset_train 是 SimpleDataset，直接修改 targets
        current_targets = np.array(self.dataset_train.targets)
        new_labels = np.argmax(glob_output, axis=1)
        
        for idx in correction_indices:
            current_targets[idx] = new_labels[idx]
            count += 1
            
        # 更新数据集
        self.dataset_train.targets = current_targets.tolist()
        print(f"  > Client {self.id}: Corrected {count} samples based on global model.")
        return count

    # --- 原有方法保持不变 ---
    def prototype_update(self):
        self.model.eval()
        features_sum = {}
        counts = {}
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.proto_loader):
                data, target = data.to(self.args.device), target.to(self.args.device)
                _, feature, _ = self.model(data)
                for i in range(len(target)):
                    label = target[i].item()
                    emb = feature[i]
                    if label not in features_sum:
                        features_sum[label] = emb
                        counts[label] = 1
                    else:
                        features_sum[label] += emb
                        counts[label] += 1
        self.prototype = {}
        for label in features_sum.keys():
            avg_proto = features_sum[label] / counts[label]
            self.prototype[label] = avg_proto / max(torch.norm(avg_proto), 1e-12)

    def cosine_similar(self, server):
        similarity = []
        valid_keys = []
        for key in self.prototype.keys():
            if key in server.global_prototype:
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

        # === 新增：在这里打印余弦相似度详情 ===
        # 格式化一下，保留4位小数，方便查看
        sim_str = [f"{s:.4f}" for s in similarity]
        print(f"Client {self.id} Cosine Sim (Avg: {reput:.4f}): {sim_str}")
        
        if len(self.rs) == 0:
            for key in valid_keys:
                self.rs[key] = 0.05 * self.simi[key]
        else:
            for key in valid_keys:
                if key in self.rs:
                    self.rs[key] = 0.95 * self.rs[key] + 0.05 * self.simi[key]
                else:
                    self.rs[key] = 0.05 * self.simi[key]

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
        self.model.train()
        for epoch in range(local_epoch):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.args.device), target.to(self.args.device)
                self.optimizer.zero_grad()
                pred, _, _ = self.model(data)
                loss = self.model.loss(pred, target)
                loss.backward()
                self.optimizer.step()
        for k in self.W:
            self.dW[k] = self.W[k].data - self.W_old[k].data