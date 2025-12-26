import torch
import copy
import math

# --- 移除了所有图相关的函数 (get_maxDegree, convert_to_nodeDegreeFeatures 等) ---

def flatten(grad_update):
    """
    将参数列表展平为一维向量，用于计算范数或相似度
    """
    return torch.cat([update.data.view(-1) for update in grad_update])


def unflatten(flattened, normal_shape):
    """
    将展平的向量恢复为原始参数形状
    """
    grad_update = []
    for param in normal_shape:
        n_params = len(param.view(-1))
        grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size()))
        flattened = flattened[n_params:]
    return grad_update


def mask_grad_update_by_magnitude(grad_update, mask_constant):
    """
    基于幅度掩码梯度：绝对值小于 mask_constant 的置为 0
    """
    grad_update = copy.deepcopy(grad_update)
    for i, update in enumerate(grad_update):
        grad_update[i].data[update.data.abs() < mask_constant] = 0
    return grad_update


def mask_grad_update_by_order(grad_update, mask_order=None, mask_percentile=None, mode='all'):
    """
    基于排名掩码梯度：保留最大的前 k 个值，其余置为 0。
    这是 FairGraphFL 进行贡献度分配（Reputation-based Gradient Masking）的核心工具。
    """
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
    """
    累加梯度更新
    """
    assert len(grad_update_1) == len(grad_update_2), "Lengths of the two grad_updates not equal"
    
    for param_1, param_2 in zip(grad_update_1, grad_update_2):
        param_1.data += param_2.data * weight