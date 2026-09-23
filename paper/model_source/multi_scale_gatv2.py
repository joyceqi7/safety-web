"""
MultiScaleGATv2 - 多尺度图注意力网络V2
针对生产安全风险的多尺度特性设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops


class MultiScaleGATv2(nn.Module):

    def __init__(self, in_dim=5, hidden_dim=128, num_layers=3, num_heads=4,
                 dropout=0.3, num_classes=2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

        # 三个并行分支，每个输出 hidden_dim
        # 局部尺度 (1-hop)
        self.local_conv1 = GATv2Conv(in_dim, hidden_dim // num_heads,
                                     heads=num_heads, dropout=dropout, concat=True)
        self.local_norm1 = nn.LayerNorm(hidden_dim)
        self.local_conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=1,
                                     dropout=dropout, concat=False)
        self.local_norm2 = nn.LayerNorm(hidden_dim)

        # 中观尺度 (2-hop)
        self.meso_conv1 = GATv2Conv(in_dim, hidden_dim // num_heads,
                                    heads=num_heads, dropout=dropout, concat=True)
        self.meso_norm1 = nn.LayerNorm(hidden_dim)
        self.meso_conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=1,
                                    dropout=dropout, concat=False)
        self.meso_norm2 = nn.LayerNorm(hidden_dim)

        # 全局尺度
        self.global_conv1 = GATv2Conv(in_dim, hidden_dim // num_heads,
                                      heads=num_heads, dropout=dropout, concat=True)
        self.global_norm1 = nn.LayerNorm(hidden_dim)
        self.global_conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=1,
                                      dropout=dropout, concat=False)
        self.global_norm2 = nn.LayerNorm(hidden_dim)

        # 多尺度注意力融合 (输入: 3 * hidden_dim，输出: 3个权重 + hidden_dim特征)
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1),
        )

        # 融合投影
        self.fusion_proj = nn.Linear(hidden_dim * 3, hidden_dim)

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

        # 局部尺度
        x_local = self.local_conv1(x, edge_index)
        x_local = self.local_norm1(x_local)
        x_local = F.elu(x_local)
        x_local = F.dropout(x_local, p=self.dropout, training=self.training)
        x_local, local_attn = self.local_conv2(x_local, edge_index,
                                                return_attention_weights=True)
        x_local = self.local_norm2(x_local)

        # 中观尺度
        edge_index_2hop = self._get_2hop_edges(edge_index, x.size(0))
        x_meso = self.meso_conv1(x, edge_index_2hop)
        x_meso = self.meso_norm1(x_meso)
        x_meso = F.elu(x_meso)
        x_meso = F.dropout(x_meso, p=self.dropout, training=self.training)
        x_meso = self.meso_conv2(x_meso, edge_index_2hop)
        x_meso = self.meso_norm2(x_meso)

        # 全局尺度
        edge_index_global, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x_global = self.global_conv1(x, edge_index_global)
        x_global = self.global_norm1(x_global)
        x_global = F.elu(x_global)
        x_global = F.dropout(x_global, p=self.dropout, training=self.training)
        x_global = self.global_conv2(x_global, edge_index_global)
        x_global = self.global_norm2(x_global)

        # 多尺度特征拼接
        concat_feat = torch.cat([x_local, x_meso, x_global], dim=-1)  # [N, 3H]

        # 尺度权重
        scale_weights = self.scale_attention(concat_feat)  # [N, 3]
        self.attention_weights = scale_weights

        # 融合: 先投影再按权重融合
        fused = self.fusion_proj(concat_feat)  # [N, H]

        logits = self.classifier(fused)
        return logits

    def _get_2hop_edges(self, edge_index, num_nodes):
        if edge_index.size(1) == 0:
            return edge_index

        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1.0

        adj_2hop = adj @ adj
        adj_2hop = adj_2hop - adj
        adj_2hop = adj_2hop.clamp(min=0)
        adj_2hop = (adj_2hop > 0).float()

        src, dst = torch.where(adj_2hop > 0)
        if src.size(0) == 0:
            return edge_index
        edge_index_2hop = torch.stack([src, dst], dim=0)
        return torch.cat([edge_index, edge_index_2hop], dim=1)

    def get_attention_weights(self):
        return self.attention_weights
