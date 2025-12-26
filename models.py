import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10) # CIFAR-10 has 10 classes

    def forward(self, x):
        # x shape: [batch, 3, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        feature = F.relu(self.fc2(x)) # This is the embedding (size 192)
        out = self.fc3(feature)
        
        # 保持与原项目一致的返回值格式: (log_softmax_output, feature_embedding, feature_embedding)
        # 原项目: return x, x1, x2 (pred, pool, emb)
        return F.log_softmax(out, dim=1), feature, feature

    def loss(self, pred, label):
        return F.nll_loss(pred, label)