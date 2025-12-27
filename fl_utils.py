import torch
import copy
import math
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import cdist # 需要 pip install scipy

# --- 原 FairGraphFL 工具函数 (保持不变) ---

def flatten(grad_update):
    return torch.cat([update.data.view(-1) for update in grad_update])

def unflatten(flattened, normal_shape):
    grad_update = []
    for param in normal_shape:
        n_params = len(param.view(-1))
        grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size()))
        flattened = flattened[n_params:]
    return grad_update

def mask_grad_update_by_magnitude(grad_update, mask_constant):
    grad_update = copy.deepcopy(grad_update)
    for i, update in enumerate(grad_update):
        grad_update[i].data[update.data.abs() < mask_constant] = 0
    return grad_update

def mask_grad_update_by_order(grad_update, mask_order=None, mask_percentile=None, mode='all'):
    if mode == 'layer':
        grad_update = copy.deepcopy(grad_update)
        mask_percentile = max(0, mask_percentile)
        for i, layer in enumerate(grad_update):
            layer_mod = layer.data.view(-1).abs()
            if mask_percentile is not None:
                mask_order = math.ceil(len(layer_mod) * mask_percentile)
            if mask_order == 0:
                grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
            else:
                topk, indices = torch.topk(layer_mod, min(mask_order, len(layer_mod)-1))
                grad_update[i].data[layer.data.abs() < topk[-1]] = 0
        return grad_update

    elif mode == 'all':
        all_update_mod = torch.cat([update.data.view(-1).abs() for update in grad_update])
        if not mask_order and mask_percentile is not None:
            mask_order = int(len(all_update_mod) * mask_percentile)
        if mask_order == 0:
            return mask_grad_update_by_magnitude(grad_update, float('inf'))
        else:
            topk, indices = torch.topk(all_update_mod, mask_order)
            return mask_grad_update_by_magnitude(grad_update, topk[-1])

def add_gradient_updates(grad_update_1, grad_update_2, weight=1.0):
    assert len(grad_update_1) == len(grad_update_2), "Lengths of the two grad_updates not equal"
    for param_1, param_2 in zip(grad_update_1, grad_update_2):
        param_1.data += param_2.data * weight

# --- 新增: FedCorr 核心函数 ---

def lid_term(X, batch, k=20):
    """
    计算局部内在维度 (LID) - 修复版
    """
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)
    
    f = lambda v: - k / (np.sum(np.log(v / (v[-1]+eps)))+eps)
    distances = cdist(X, batch)

    # 获取最近的 k 个邻居
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    m, n = sort_indices.shape
    
    # === 修复点：强制转换为 list，防止 np.ogrid 返回 tuple 导致报错 ===
    idx = list(np.ogrid[:m, :n])
    
    # 现在 idx 肯定是 list，可以修改了
    idx[1] = sort_indices
    
    # sorted matrix
    distances_ = distances[tuple(idx)]
    lids = np.apply_along_axis(f, axis=1, arr=distances_)
    return lids

def get_output(loader, net, args, criterion=None):
    """
    获取模型在数据集上的输出概率和损失
    """
    net.eval()
    output_whole = []
    loss_whole = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            # 注意：这里的 model 输出格式需适配你的 models.py
            # 假设 CNNCifar 返回 (log_softmax, feature1, feature2)
            outputs, _, _ = net(images)
            
            # 如果输出已经是 log_softmax，这里可能不需要再 F.softmax
            # 但为了计算 LID，通常需要概率分布
            probs = torch.exp(outputs) 
            
            if criterion is not None:
                # 计算每个样本的 loss (reduction='none')
                loss = criterion(outputs, labels) 
                loss_whole.append(loss.cpu().numpy())
            
            output_whole.append(probs.cpu().numpy())
            
    output_whole = np.concatenate(output_whole, axis=0)
    if criterion is not None:
        loss_whole = np.concatenate(loss_whole, axis=0)
        return output_whole, loss_whole
    else:
        return output_whole