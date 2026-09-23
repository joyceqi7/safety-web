"""
步骤2-3: 使用LLM (DeepSeek API) 从事故报告文本中提取风险数据
提取三类核心数据:
  1. 风险要素节点 (四类: 人/物/环境/管理)
  2. 因果关系边 (邻接矩阵 + 边置信度)
  3. 节点5维特征 [frequency, severity, detectability, controllability, propagation_speed]

采用多轮Prompt工程设计
"""

import os
import sys
import json
import time
import hashlib
from openai import OpenAI
from config import (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
                    EXTRACTED_TEXT_DIR, EXTRACTED_DATA_DIR)

# 修复Windows GBK编码问题
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


SYSTEM_PROMPT = """你是一个生产安全事故风险分析专家。你需要从事故调查报告中精确提取风险要素信息。

## 风险要素分类标准
1. **人的不安全行为**: 操作失误、违章作业、监护不到位、培训不足等人因
2. **物的不安全状态**: 设备故障、防护缺失、材料缺陷、设计缺陷等物因
3. **环境不安全因素**: 天气恶劣、场地狭窄、照明不足、通风不良等环境因素
4. **管理缺陷**: 制度缺失、监管不力、隐患排查不到位、应急预案缺失等管理因素

## 评分标准 (1-5分)
- **frequency (发生频率)**: 1=极罕见, 2=罕见, 3=偶尔, 4=频繁, 5=极频繁
- **severity (影响程度)**: 1=可忽略, 2=轻微, 3=中等, 4=严重, 5=灾难性
- **detectability (可检测性)**: 1=极易检测, 2=易检测, 3=中等, 4=难检测, 5=极难检测
- **controllability (可控制性)**: 1=极易控制, 2=易控制, 3=中等, 4=难控制, 5=极难控制
- **propagation_speed (传导速度)**: 1=极慢, 2=慢, 3=中等, 4=快, 5=极快

## 输出要求
仅输出JSON，不要包含Markdown代码块标记。JSON格式如下:
{
  "nodes": [
    {
      "id": "N0",
      "name": "风险要素名称",
      "type": "人的不安全行为",
      "features": [frequency, severity, detectability, controllability, propagation_speed],
      "is_high_risk": true
    }
  ],
  "edges": [
    {
      "source": "N0",
      "target": "N1",
      "relation": "导致",
      "confidence": 0.85
    }
  ]
}"""


def build_extraction_prompt(report_text: str) -> str:
    """构建提取Prompt"""
    # 对过长文本进行截断
    max_chars = 12000
    if len(report_text) > max_chars:
        report_text = report_text[:max_chars] + "\n...(文本已截断)"

    return f"""请分析以下生产安全事故调查报告，提取所有风险要素及其因果关系。

## 事故报告内容
{report_text}

## 提取要求
1. 提取所有相关风险要素节点，确保覆盖四类风险要素（人的不安全行为、物的不安全状态、环境不安全因素、管理缺陷）
2. 识别风险要素之间的因果关系（谁导致了谁）
3. 对每个节点从五个维度评分(1-5)
4. 判断每个节点是高风险(true)还是低风险(false)：
   - 高风险标准：severity>=4 或 (severity>=3 且 frequency>=4) 或该要素被明确列为事故直接原因
5. 至少提取3个节点，最多20个节点"""


def call_llm(prompt: str, max_retries: int = 3) -> dict | None:
    """调用DeepSeek API"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            # 清理可能的Markdown标记
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  JSON解析失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
        except Exception as e:
            print(f"  API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(5)
    return None


def validate_extraction(data: dict) -> bool:
    """验证提取数据的格式"""
    if not data or "nodes" not in data or "edges" not in data:
        return False
    nodes = data["nodes"]
    edges = data["edges"]
    if len(nodes) < 2:
        return False
    node_ids = set()
    for node in nodes:
        if not all(k in node for k in ["id", "name", "type", "features", "is_high_risk"]):
            return False
        if len(node["features"]) != 5:
            return False
        if node["type"] not in ["人的不安全行为", "物的不安全状态", "环境不安全因素", "管理缺陷"]:
            return False
        node_ids.add(node["id"])
    for edge in edges:
        if not all(k in edge for k in ["source", "target", "relation", "confidence"]):
            return False
        if edge["source"] not in node_ids or edge["target"] not in node_ids:
            return False
    return True


def extract_single_report(text: str, filename: str,
                          cache_dir: str = EXTRACTED_DATA_DIR) -> dict | None:
    """提取单个报告的风险数据 (带缓存)"""
    # 检查缓存
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{text_hash}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    prompt = build_extraction_prompt(text)
    result = call_llm(prompt)

    if result and validate_extraction(result):
        result["_metadata"] = {"filename": filename, "text_hash": text_hash}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    safe_name = filename.encode('ascii', errors='replace').decode('ascii')
    print(f"  WARNING: {safe_name} extraction failed or validation failed")
    return None


def extract_all_reports(text_dir: str = EXTRACTED_TEXT_DIR,
                        output_dir: str = EXTRACTED_DATA_DIR) -> list[dict]:
    """批量提取所有报告的风险数据"""
    txt_files = sorted([
        f for f in os.listdir(text_dir)
        if f.endswith(".txt") and not f.startswith("_")
    ])
    print(f"处理 {len(txt_files)} 份文本文件")

    all_results = []
    failed = []
    for txt_file in txt_files:
        txt_path = os.path.join(text_dir, txt_file)
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        if len(text.strip()) < 50:
            failed.append(txt_file)
            safe_name = txt_file.encode('ascii', errors='replace').decode('ascii')
            print(f"  SKIP {safe_name}: text too short ({len(text)} chars)")
            continue

        safe_name = txt_file.encode('ascii', errors='replace').decode('ascii')
        print(f"  Processing: {safe_name} ({len(text)} chars)")
        result = extract_single_report(text, txt_file.replace(".txt", ".pdf"))
        if result:
            all_results.append(result)
        else:
            failed.append(txt_file)

    # 保存汇总
    summary_path = os.path.join(output_dir, "_all_extractions.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    total_nodes = sum(len(r.get("nodes", [])) for r in all_results)
    total_edges = sum(len(r.get("edges", [])) for r in all_results)
    print(f"\nExtraction complete: {len(all_results)}/{len(txt_files)} successful")
    print(f"  Total nodes: {total_nodes}, Total edges: {total_edges}")
    if failed:
        safe_failed = [f.encode('ascii', errors='replace').decode('ascii') for f in failed[:5]]
        print(f"  Failed: {len(failed)} files ({', '.join(safe_failed)}...)")
    return all_results


if __name__ == "__main__":
    extract_all_reports()
