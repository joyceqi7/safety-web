"""
步骤4: 将LLM提取的风险数据转换为PyG (PyTorch Geometric) 格式
构建图数据集，包含:
  - 节点特征矩阵 x: [num_nodes, 5]
  - 边索引 edge_index: [2, num_edges]
  - 边权重 edge_weight: [num_edges]
  - 节点标签 y: [num_nodes]
  - 图批次标识 batch
"""

import os
import json
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import StandardScaler
from config import (EXTRACTED_DATA_DIR, GRAPH_DIR, RANDOM_SEED,
                    RISK_ELEMENT_TYPES)


def normalize_features(features: list) -> list:
    """将1-5评分归一化到[0,1]"""
    return [(f - 1.0) / 4.0 for f in features]


def build_single_graph(extraction: dict) -> Data | None:
    """
    将单份报告的LLM提取结果转换为PyG Data对象

    Args:
        extraction: LLM提取的JSON数据

    Returns:
        torch_geometric.data.Data 或 None
    """
    nodes = extraction.get("nodes", [])
    edges = extraction.get("edges", [])

    if len(nodes) < 2:
        return None

    # 构建节点ID到索引的映射
    id_to_idx = {node["id"]: i for i, node in enumerate(nodes)}

    # 节点特征: [num_nodes, 5]
    x = torch.tensor(
        [normalize_features(node["features"]) for node in nodes],
        dtype=torch.float
    )

    # 节点标签: 0=低风险, 1=高风险
    y = torch.tensor(
        [1 if node["is_high_risk"] else 0 for node in nodes],
        dtype=torch.long
    )

    # 节点类型 (用于分析)
    node_types = [node["type"] for node in nodes]
    node_names = [node["name"] for node in nodes]

    # 边索引: [2, num_edges]
    if edges:
        edge_list = []
        edge_weights = []
        for edge in edges:
            src = id_to_idx.get(edge["source"])
            tgt = id_to_idx.get(edge["target"])
            if src is not None and tgt is not None:
                edge_list.append([src, tgt])
                edge_weights.append(edge.get("confidence", 0.5))
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weight = torch.zeros(0, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)

    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=len(nodes),
    )
    # 附加元数据
    data.node_types = node_types
    data.node_names = node_names
    data.filename = extraction.get("_metadata", {}).get("filename", "unknown")

    return data


def build_all_graphs(data_dir: str = EXTRACTED_DATA_DIR) -> list[Data]:
    """批量构建所有报告的图数据"""
    summary_path = os.path.join(data_dir, "_all_extractions.json")
    if not os.path.exists(summary_path):
        # 尝试逐个加载
        graph_list = []
        for f in sorted(os.listdir(data_dir)):
            if f.endswith(".json") and not f.startswith("_"):
                with open(os.path.join(data_dir, f), "r", encoding="utf-8") as fp:
                    extraction = json.load(fp)
                data = build_single_graph(extraction)
                if data is not None:
                    graph_list.append(data)
        return graph_list

    with open(summary_path, "r", encoding="utf-8") as f:
        all_extractions = json.load(f)

    graph_list = []
    for extraction in all_extractions:
        data = build_single_graph(extraction)
        if data is not None:
            graph_list.append(data)

    print(f"构建了 {len(graph_list)} 个图 (共 {len(all_extractions)} 份提取结果)")
    return graph_list


class AccidentGraphDataset(InMemoryDataset):
    """生产安全事故图数据集"""

    def __init__(self, root: str = GRAPH_DIR, transform=None, pre_transform=None):
        self.graph_list = None
        super().__init__(root, transform, pre_transform)
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0],
                                                 weights_only=False)
        else:
            self.process()

    @property
    def raw_file_names(self):
        return ["_all_extractions.json"]

    @property
    def processed_file_names(self):
        return ["accident_graph_dataset.pt"]

    def download(self):
        pass

    def process(self):
        # 从提取数据构建图
        graph_list = build_all_graphs()

        if not graph_list:
            raise RuntimeError("没有成功构建任何图数据")

        self.graph_list = graph_list
        data, slices = self.collate(graph_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_graph_list(self):
        if self.graph_list is None:
            self.graph_list = [self.get(i) for i in range(len(self))]
        return self.graph_list


def get_dataset_split(dataset):
    """划分训练/验证/测试集"""
    from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED

    np.random.seed(RANDOM_SEED)
    n = len(dataset)
    indices = np.random.permutation(n)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return train_idx, val_idx, test_idx


def get_node_statistics(graph_list: list) -> dict:
    """统计图数据集的节点信息"""
    total_nodes = 0
    total_edges = 0
    type_counts = {t: 0 for t in RISK_ELEMENT_TYPES}
    high_risk_count = 0

    for data in graph_list:
        n = data.num_nodes
        total_nodes += n
        total_edges += data.num_edges
        for t in data.node_types:
            if t in type_counts:
                type_counts[t] += 1
        high_risk_count += data.y.sum().item()

    return {
        "num_graphs": len(graph_list),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "avg_nodes_per_graph": total_nodes / max(len(graph_list), 1),
        "avg_edges_per_graph": total_edges / max(len(graph_list), 1),
        "type_distribution": type_counts,
        "high_risk_ratio": high_risk_count / max(total_nodes, 1),
    }


if __name__ == "__main__":
    dataset = AccidentGraphDataset()
    print(f"\n数据集统计:")
    stats = get_node_statistics(dataset.get_graph_list())
    print(json.dumps(stats, ensure_ascii=False, indent=2))
