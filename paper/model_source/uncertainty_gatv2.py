"""
UncertaintyGATv2 - 不确定性感知图注意力网络V2
针对生产安全风险评估中的不确定性建模:
  - 使用MC Dropout进行贝叶斯近似推理
  - 输出预测分布而非点估计
  - 量化风险评估的置信度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class UncertaintyGATv2(nn.Module):

    def __init__(self, in_dim=5, hidden_dim=128, num_layers=3, num_heads=4,
                 dropout=0.3, num_classes=2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(
            GATv2Conv(in_dim, hidden_dim // num_heads, heads=num_heads,
                      dropout=dropout, concat=True)
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                          dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.convs.append(
            GATv2Conv(hidden_dim, hidden_dim, heads=1,
                      dropout=dropout, concat=False)
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        # 分类头: 标准logits + 不确定性logits
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        # 不确定性估计头
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # 标量不确定性
        )

        self.attention_weights = None

    def forward(self, data, mc_samples=1):
        x, edge_index = data.x, data.edge_index

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if i == len(self.convs) - 1:
                x_new, attn = conv(x, edge_index, return_attention_weights=True)
                self.attention_weights = attn
            else:
                x_new = conv(x, edge_index)
            x = norm(x_new)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # 标准分类logits
        logits = self.classifier(x)

        # 不确定性估计
        uncertainty = self.uncertainty_head(x)  # [N, 1]

        return logits, uncertainty

    def predict_with_uncertainty(self, data, mc_samples=30):
        """MC Dropout推理: 多次前向传播估计不确定性"""
        self.train()  # 保持dropout开启
        all_logits = []
        all_uncertainties = []
        for _ in range(mc_samples):
            logits, uncertainty = self.forward(data, mc_samples=1)
            all_logits.append(F.softmax(logits, dim=-1))
            all_uncertainties.append(uncertainty)
        mean_prob = torch.stack(all_logits).mean(0)
        epistemic_uncertainty = torch.stack(all_logits).std(0).mean(dim=-1)
        aleatoric_uncertainty = torch.stack(all_uncertainties).mean(0).squeeze(-1)
        return mean_prob, epistemic_uncertainty + aleatoric_uncertainty

    def get_attention_weights(self):
        return self.attention_weights


class UncertaintyLoss(nn.Module):
    """不确定性感知损失: CE + 不确定性正则化"""

    def __init__(self, lambda_reg=0.01):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        logits, uncertainty = outputs
        ce_loss = self.ce(logits, targets)
        # 正则化: 惩罚过大的不确定性
        reg_loss = self.lambda_reg * uncertainty.mean()
        return ce_loss + reg_loss
