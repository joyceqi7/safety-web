"""
GATv2Transformer - GATv2与Transformer融合模型
结合图注意力网络和自注意力机制的互补优势:
  - GATv2: 建模图结构中的局部风险传导关系
  - Transformer: 建模全局风险要素间的长程依赖
  - 双分支并行后融合，兼顾局部结构和全局语义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATv2Transformer(nn.Module):

    def __init__(self, in_dim=5, hidden_dim=128, num_layers=3, num_heads=4,
                 dropout=0.3, num_classes=2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # GATv2分支
        self.gat_convs = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        self.gat_convs.append(
            GATv2Conv(in_dim, hidden_dim // num_heads, heads=num_heads,
                      dropout=dropout, concat=True)
        )
        self.gat_norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 2):
            self.gat_convs.append(
                GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                          dropout=dropout, concat=True)
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))

        self.gat_convs.append(
            GATv2Conv(hidden_dim, hidden_dim // 2, heads=1,
                      dropout=dropout, concat=False)
        )
        self.gat_norms.append(nn.LayerNorm(hidden_dim // 2))

        # Transformer分支
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 融合
        fusion_dim = (hidden_dim // 2) + hidden_dim
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            fusion_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.attention_weights = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GATv2分支
        x_gat = x
        for i, (conv, norm) in enumerate(zip(self.gat_convs, self.gat_norms)):
            if i == len(self.gat_convs) - 1:
                x_gat_new, attn = conv(x_gat, edge_index, return_attention_weights=True)
                self.attention_weights = attn
            else:
                x_gat_new = conv(x_gat, edge_index)
            x_gat = norm(x_gat_new)
            if i < len(self.gat_convs) - 1:
                x_gat = F.elu(x_gat)
                x_gat = F.dropout(x_gat, p=self.dropout, training=self.training)

        # Transformer分支
        x_tf = self.input_proj(x)
        x_tf = x_tf.unsqueeze(0)
        attn_mask = self._build_structure_mask(edge_index, x.size(0), x.device)
        x_tf = self.transformer(x_tf, mask=attn_mask)
        x_tf = x_tf.squeeze(0)

        # 融合
        x_fused = torch.cat([x_gat, x_tf], dim=-1)
        x_fused = x_fused.unsqueeze(0)
        x_fused, _ = self.cross_attn(x_fused, x_fused, x_fused)
        x_fused = self.fusion_norm(x_fused.squeeze(0))

        logits = self.classifier(x_fused)
        return logits

    def _build_structure_mask(self, edge_index, num_nodes, device):
        mask = torch.zeros(num_nodes, num_nodes, device=device)
        if edge_index.size(1) > 0:
            src, dst = edge_index
            mask[src, dst] = 1.0
            mask[dst, src] = 1.0
        mask = (1.0 - mask) * -1e9
        mask.fill_diagonal_(0)
        return mask

    def get_attention_weights(self):
        return self.attention_weights
