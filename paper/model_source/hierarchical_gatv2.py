"""
HierarchicalGATv2 - 层级图注意力网络V2
针对生产安全风险的层级传导特性设计:
  直接原因层 -> 间接原因层 -> 根本原因层
通过多尺度GATv2 + 层级聚合实现风险传导建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class HierarchicalGATv2(nn.Module):

    def __init__(self, in_dim=5, hidden_dim=128, num_layers=3, num_heads=4,
                 dropout=0.3, num_classes=2, pool_ratio=0.7):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        h = hidden_dim  # shorthand

        # 底层编码: 直接原因
        self.bottom_conv = GATv2Conv(in_dim, h, heads=num_heads,
                                     dropout=dropout, concat=True)
        self.bottom_norm = nn.LayerNorm(h * num_heads)
        self.bottom_proj = nn.Linear(h * num_heads, h)  # project to H

        # 中层编码: 间接传导
        self.mid_conv = GATv2Conv(h, h, heads=num_heads,
                                  dropout=dropout, concat=True)
        self.mid_norm = nn.LayerNorm(h * num_heads)
        self.mid_proj = nn.Linear(h * num_heads, h)

        # 顶层编码: 根本原因 + 注意力
        self.top_conv = GATv2Conv(h, h, heads=1,
                                  dropout=dropout, concat=False)
        self.top_norm = nn.LayerNorm(h)

        # 层级特征投影 (3*H -> H): 全局上下文
        self.global_to_node = nn.Sequential(
            nn.Linear(h * 3, h * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h * 2, h),
        )

        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, num_classes),
        )

        self.attention_weights = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 底层: 直接原因
        x1 = self.bottom_conv(x, edge_index)
        x1 = self.bottom_norm(x1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        g1 = global_mean_pool(x1, batch)
        g1 = self.bottom_proj(g1)  # [G, H]
        x1_proj = self.bottom_proj(x1)  # [N, H]

        # 中层: 间接传导 (接底层投影后的特征)
        x2 = self.mid_conv(x1_proj, edge_index)
        x2 = self.mid_norm(x2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        g2 = global_max_pool(x2, batch)
        g2 = self.mid_proj(g2)  # [G, H]
        x2_proj = self.mid_proj(x2)  # [N, H]

        # 顶层: 根本原因 + 注意力
        x3, attn = self.top_conv(x2_proj, edge_index, return_attention_weights=True)
        x3 = self.top_norm(x3)
        self.attention_weights = attn
        g3 = global_mean_pool(x3, batch)  # [G, H]

        # 层级全局特征融合
        hier_global = torch.cat([g1, g2, g3], dim=-1)  # [G, 3H]
        node_ctx = self.global_to_node(hier_global)      # [G, H]
        node_ctx_expanded = node_ctx[batch]               # [N, H]

        # 融合节点特征与全局层级上下文
        combined = torch.cat([x3, node_ctx_expanded], dim=-1)
        logits = self.classifier(combined)
        return logits

    def get_attention_weights(self):
        return self.attention_weights
