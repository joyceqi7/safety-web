# Safety 论文代码包

论文《面向职业事故分析的大语言模型辅助风险图构建与节点筛查》（LLM 辅助风险图构建 + GATv2 系列模型节点筛查）的完整代码与数据包。


## 目录结构

```
Safety_Paper_Code/
├── README.md                 ← 本文件
├── requirements.txt          ← Python 依赖
├── Supplementary_Prompt.md   ← LLM 提取的完整提示词与 API 参数记录
├── data_processing/          ← 步骤1–3：PDF→文本→LLM结构化提取→PyG图
│   ├── config.py             ← 路径/种子/划分比例/API参数（需填 API Key）
│   ├── extract_pdf.py        ← 步骤1：批量提取 PDF 文本（PyMuPDF）
│   ├── llm_extract.py        ← 步骤2：LLM 提取节点/边/五维特征（DeepSeek API）
│   └── build_graph.py        ← 步骤3：构建 PyG 图数据集 + 划分统计
├── model_source/             ← 六种对比模型实现
│   ├── gatv2_base.py                （GATv2 基线）
│   ├── hierarchical_gatv2.py        （层次编码 + 图级上下文，主模型）
│   ├── uncertainty_gatv2.py         （不确定性估计）
│   ├── multi_scale_gatv2.py         （多尺度邻域）
│   ├── dynamic_temporal_gatv2.py    （因果顺序）
│   └── gatv2_transformer.py         （Transformer 长程依赖）
├── experiments/              ← 补充实验（本轮论文修改的依据，可复现）
│   ├── run_experiments.py    ← E0–E5 总控：规则基线/多种子/三特征/消融/鲁棒性/五折CV
│   ├── fix_e3.py             ← E3 修正脚本（HierNoContext 重训）
│   ├── run_e6_fewshot.py     ← E6：小样本（10%–70% 训练比例）× 3 种子
│   └── results/              ← E0–E6 全部结果 JSON（与论文各表一一对应）
├── original_results/         ← 原始实验结果（2026-05-04 首次运行）
│   ├── all_experiments.json        （噪声/删边/小样本/五折汇总）
│   └── comparison_results.json     （六模型固定划分指标，对应论文表5）
├── processed_data/           ← 处理好的数据（可直接用于训练）
│   ├── extracted_data/       ← 103 份有效图的 LLM 提取 JSON + _all_extractions.json 汇总
│   │                             （1,131 节点 / 1,122 边；104 份报告中 103 份有效）
│   ├── extracted_text/       ← PDF 提取的纯文本
│   └── graph_data/           ← PyG 序列化数据集（accident_graph_dataset.pt 等）
└── model_weights/            ← 六个 .pt 检查点（含 state_dict/测试指标/训练历史/配置）
```

## 环境

- Python ≥ 3.10（补充实验实测：Python 3.13 + PyTorch 2.14.0+cpu + PyG 2.8.0，纯 CPU 可跑）
- 安装依赖：`pip install -r requirements.txt`

## 快速开始

### 路线一：复现补充实验（无需 API Key，约 40+30 分钟 CPU）

```bash
cd experiments
python run_experiments.py     # E0–E5，结果写入 results/（覆盖同名 JSON）
python run_e6_fewshot.py      # E6 小样本多种子，约 30 分钟
```

`run_experiments.py` 的 E1 就是六模型的完整训练+评估入口（原始训练入口脚本未随作者材料提供，E1 以相同超参重训了全部六模型）。

各实验组与论文表格的对应：

| 实验 | 内容 | 对应论文 |
|------|------|----------|
| E0 | 规则基线 + 数据统计 | 表 5 规则基线行、§4.1 数据统计 |
| E1 | 六模型 × 种子 42/43/44 固定划分 | 表 5（重训版）、表 11 完整模型行 |
| E2 | 去频率/严重程度三特征 | 表 11 三特征行、循环依赖分析 |
| E3 | 消融（−图级上下文 / −因果边） | 表 11 消融行 |
| E4 | 噪声/删边 × 3 实现（用 E1 seed42 模型） | 表 7、表 8 |
| E5 | 报告级分组五折交叉验证 | 表 6 |
| E6 | 小样本 10%–70% × 3 种子 | 表 9 |

注意：重新运行会覆盖 `experiments/results/` 下的同名 JSON，并在其中生成 `m3_*.pt`/`m5_*.pt` 中间模型文件；如需保留当前结果请先备份。

### 路线二：从头跑数据处理流程（需要 API Key 与原始 PDF）

1. `data_processing/config.py`：填入 `LLM_API_KEY`（及实际使用的端点/模型 ID）；
2. 将原始事故调查报告 PDF 放入包根的 `raw_reports/`（本包不含原始 PDF，共 104 份，2019–2025 年取自上海/北京/天津/重庆应急管理部门官网，详见论文数据可用性声明）；
3. 依序运行：

```bash
cd data_processing
python extract_pdf.py     # 步骤1：PDF → 文本
python llm_extract.py     # 步骤2：文本 → 节点/边/特征 JSON（带缓存与格式校验）
python build_graph.py     # 步骤3：JSON → PyG 数据集，打印数据集统计
```

提取提示词全文与 API 参数（temperature=0.1、max_tokens=4096、JSON 输出等）见根目录 `Supplementary_Prompt.md`。

## 数据与结果说明

- `processed_data/extracted_data/_all_extractions.json`：103 个图的汇总，节点特征为 1–5 整数 Likert 评分，`build_graph.py` 与补充实验脚本均按 `(f-1)/4` 归一化到 [0,1]。
- 高风险标签规则：severity≥4，或（severity≥3 且 frequency≥4），或要素被明确列为事故直接原因（数据中该条款从未触发）。
- `model_weights/` 中的检查点为原始实验产物，不含优化器状态与种子元数据。
- `original_results/` 是原始单次运行结果；论文修改稿中表 7/8/9 已改用 `experiments/results/` 中的多种子版本。

## 已知边界

- 原始 PDF 报告（101MB）不在包内；缺少它们时路线二无法运行，路线一不受影响。
- `config.py` 为重建模板：原始配置（真实 API 端点/密钥）未包含在作者提供的材料中。
- 原始实验（产生 `original_results/` 的那次运行）的训练入口脚本未包含在作者材料中；如需该环境的精确复现，以 E1 重训结果为准（论文修改稿已按此口径表述）。
