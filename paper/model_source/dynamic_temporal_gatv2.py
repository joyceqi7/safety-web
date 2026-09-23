"""
DynamicTemporalGATv2 - 动态时序图注意力网络V2
针对生产安全风险的时序因果特性设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class TemporalAttention(nn.Module):

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class DynamicTemporalGATv2(nn.Module):

    def __init__(self, in_dim=5, hidden_dim=128, num_layers=3, num_heads=4,
                 dropout=0.3, num_classes=2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.spatial_convs = nn.ModuleList()
        self.spatial_norms = nn.ModuleList()

        self.spatial_convs.append(
            GATv2Conv(in_dim, hidden_dim // num_heads, heads=num_heads,
                      dropout=dropout, concat=True)
        )
        self.spatial_norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 2):
            self.spatial_convs.append(
                GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                          dropout=dropout, concat=True)
            )
            self.spatial_norms.append(nn.LayerNorm(hidden_dim))

        self.spatial_convs.append(
            GATv2Conv(hidden_dim, hidden_dim, heads=1,
                      dropout=dropout, concat=False)
        )
        self.spatial_norms.append(nn.LayerNorm(hidden_dim))

        # 时序建模: 使用1D卷积 + 注意力
        self.temporal_attn = TemporalAttention(hidden_dim, num_heads=4,
                                               dropout=dropout)

        # 时序卷积
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

        # 融合
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.attention_weights = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 空间编码
        for i, (conv, norm) in enumerate(zip(self.spatial_convs,
                                              self.spatial_norms)):
            if i == len(self.spatial_convs) - 1:
                x_new, attn = conv(x, edge_index, return_attention_weights=True)
                self.attention_weights = attn
            else:
                x_new = conv(x, edge_index)
            x = norm(x_new)
            if i < len(self.spatial_convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        spatial_feat = x  # [N, H]

        # 时序建模: 构建路径序列
        x_seq = self._build_causal_sequence(x, edge_index)  # [1, N, H]

        # 时序注意力
        temporal_feat = self.temporal_attn(x_seq)  # [1, N, H]
        temporal_feat = temporal_feat.squeeze(0)  # [N, H]

        # 时序卷积 (转置为 [1, H, N] -> [1, H, N])
        x_conv_in = x.transpose(0, 1).unsqueeze(0)  # [1, H, N]
        conv_out = self.temporal_conv(x_conv_in)     # [1, H, N]
        conv_out = conv_out.squeeze(0).transpose(0, 1)  # [N, H]

        # 融合空间和时序特征
        combined = torch.cat([spatial_feat, temporal_feat + conv_out], dim=-1)
        logits = self.classifier(combined)
        return logits

    def _build_causal_sequence(self, x, edge_index):
        """基于因果拓扑排序构建节点序列"""
        num_nodes = x.size(0)

        # 计算入度（原因深度）
        in_degree = torch.zeros(num_nodes, device=x.device)
        if edge_index.size(1) > 0:
            dst = edge_index[1]
            in_degree.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device))

        # 根节点（无入边）= 深度0
        roots = (in_degree == 0).nonzero(as_tuple=True)[0]
        depth = torch.full((num_nodes,), -1, device=x.device, dtype=torch.float)
        if len(roots) > 0:
            depth[roots] = 0

        edge_list = edge_index.t().tolist() if edge_index.size(1) > 0 else []
        adj = {i: [] for i in range(num_nodes)}
        for src, dst in edge_list:
            adj[src].append(dst)

        # BFS计算深度
        for _ in range(min(num_nodes, 50)):
            for u in range(num_nodes):
                if depth[u] >= 0:
                    for v in adj[u]:
                        if depth[v] < 0 or depth[v] < depth[u] + 1:
                            depth[v] = depth[u] + 1

        depth = depth.clamp(min=0)
        sorted_indices = torch.argsort(depth)
        return x[sorted_indices].unsqueeze(0)  # [1, N, H]

    def get_attention_weights(self):
        return self.attention_weights
