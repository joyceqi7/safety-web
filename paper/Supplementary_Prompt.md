# Supplementary Method: DeepSeekV4Pro Extraction Prompt

This file records the prompt template used for structured extraction from the public accident investigation reports. DeepSeekV4Pro was accessed through an API. No API key, authentication token, or private endpoint is included.

## System prompt

你是一个生产安全事故风险分析专家。你需要从事故调查报告中精确提取风险要素信息。

### 风险要素分类标准

1. 人的不安全行为：操作失误、违章作业、监护不到位、培训不足等人因。
2. 物的不安全状态：设备故障、防护缺失、材料缺陷、设计缺陷等物因。
3. 环境不安全因素：天气恶劣、场地狭窄、照明不足、通风不良等环境因素。
4. 管理缺陷：制度缺失、监管不力、隐患排查不到位、应急预案缺失等管理因素。

### 评分标准（1–5 分）

- frequency（发生频率）：1=极罕见，2=罕见，3=偶尔，4=频繁，5=极频繁。
- severity（影响程度）：1=可忽略，2=轻微，3=中等，4=严重，5=灾难性。
- detectability（可检测性）：1=极易检测，2=易检测，3=中等，4=难检测，5=极难检测。
- controllability（可控制性）：1=极易控制，2=易控制，3=中等，4=难控制，5=极难控制。
- propagation_speed（传导速度）：1=极慢，2=慢，3=中等，4=快，5=极快。

### 输出要求

仅输出 JSON，不要包含 Markdown 代码块标记。JSON 格式如下：

```json
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
}
```

## User prompt template

请分析以下生产安全事故调查报告，提取所有风险要素及其因果关系。

### 事故报告内容

`{report_text}`

### 提取要求

1. 提取所有相关风险要素节点，确保覆盖四类风险要素（人的不安全行为、物的不安全状态、环境不安全因素、管理缺陷）。
2. 识别风险要素之间的因果关系（谁导致了谁）。
3. 对每个节点从五个维度评分（1–5）。
4. 判断每个节点是高风险（true）还是低风险（false）：severity≥4，或 severity≥3 且 frequency≥4，或该要素被明确列为事故直接原因。
5. 至少提取 3 个节点，最多 20 个节点。

## Recorded processing parameters

- Model: DeepSeekV4Pro.
- Access: API.
- Temperature: 0.1.
- Maximum output tokens: 4096.
- Response format: JSON object.
- Input truncation: reports longer than 12,000 characters were truncated and marked with `...(文本已截断)`.
- Retry limit: up to 3 API attempts per report.
- Post-processing: JSON parsing, schema validation, manual cleaning, duplicate removal, and format correction.
- Output schema validation required node fields `id`, `name`, `type`, `features`, and `is_high_risk`, and edge fields `source`, `target`, `relation`, and `confidence`.

## Source implementation

The prompt and parameters above are transcribed from `数据处理方法/llm_extract.py`. The accompanying supplementary archive includes the available processing scripts, processed data, and six model checkpoints. The original configuration file containing the API endpoint and key was not included.
