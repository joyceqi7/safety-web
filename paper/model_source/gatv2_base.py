"""
GATv2 (原始版) - 标准图注意力网络V2
Original GATv2: 使用动态注意力机制，解决GAT的静态注意力问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATv2Base(nn.Module):
    """原始GATv2模型用于节点级风险分类"""

    def __init__(self, in_dim=5, hidden_dim=128, num_layers=3, num_heads=4,
                 dropout=0.3, num_classes=2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 第一层: in_dim -> hidden_dim (multi-head)
        self.convs.append(
            GATv2Conv(in_dim, hidden_dim // num_heads, heads=num_heads,
                      dropout=dropout, concat=True)
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        # 中间层: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                          dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # 最后一层: hidden_dim -> hidden_dim (单头，用于分类)
        self.convs.append(
            GATv2Conv(hidden_dim, hidden_dim, heads=1,
                      dropout=dropout, concat=False)
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.attention_weights = None

    def forward(self, data):
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

        logits = self.classifier(x)
        return logits

    def get_attention_weights(self):
        """返回注意力权重用于可解释性分析"""
        return self.attention_weights
