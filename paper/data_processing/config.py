# -*- coding: utf-8 -*-
"""
统一配置文件：路径、随机种子、数据划分比例与 LLM 访问参数。

说明：
- 所有路径默认相对本代码包根目录，移动整个包后无需修改。
- LLM_API_KEY 不随代码包分发，运行 llm_extract.py 前请自行填入。
- 原始配置文件（含真实 API 端点与密钥）未包含在作者提供的材料中，
  本文件为按论文与 Supplementary_Prompt.md 记录重建的模板。
"""
import os

# 代码包根目录（config.py 位于 data_processing/ 下，故取上一级）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------- 路径 ----------------
# 原始 PDF 事故报告目录（本包不含原始 PDF，需自行放置后再运行 extract_pdf.py）
DATA_DIR = os.path.join(_ROOT, 'raw_reports')
# PDF 提取出的纯文本（步骤1输出 / 步骤2输入）
EXTRACTED_TEXT_DIR = os.path.join(_ROOT, 'processed_data', 'extracted_text')
# LLM 结构化提取结果（步骤2输出 / 步骤3输入，含 _all_extractions.json 汇总）
EXTRACTED_DATA_DIR = os.path.join(_ROOT, 'processed_data', 'extracted_data')
# PyG 图数据集（步骤3输出）
GRAPH_DIR = os.path.join(_ROOT, 'processed_data', 'graph_data')

# ---------------- 随机种子与数据划分 ----------------
RANDOM_SEED = 42            # 固定划分种子（对应论文表5/7/8的 72/15/16 份报告划分）
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------- 风险要素类型 ----------------
RISK_ELEMENT_TYPES = [
    "人的不安全行为",
    "物的不安全状态",
    "环境不安全因素",
    "管理缺陷",
]

# ---------------- LLM 访问参数 ----------------
# 论文使用 DeepSeek-V4-Pro（参数记录见根目录 Supplementary_Prompt.md）：
# temperature=0.1, max_tokens=4096, response_format=json_object，
# 超 12000 字符截断，单报告最多重试 3 次。
# 端点与模型 ID 请按实际使用的服务填写。
LLM_API_KEY = ""              # 在此填入你的 API Key（不要提交到公开仓库）
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"   # 按实际使用的模型 ID 修改（论文记录为 DeepSeek-V4-Pro）
