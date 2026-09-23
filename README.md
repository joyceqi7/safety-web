# Safety 风险图分析平台

将事故调查报告转化为风险节点与候选关系图，结合六种 GATv2 模型进行节点筛查。前端为中文分析工作台，后端使用 FastAPI、PyTorch 和 PyTorch Geometric。

[在线网站](https://47.99.242.201/) · [Windows 部署记录](deploy/WINDOWS-ALIYUN.md) · [Linux / Docker 部署](deploy/ALIYUN.md)

## 功能

- 浏览 103 份有效事故图，包含 1,131 个节点和 1,122 条关系。
- 使用六个原始模型权重进行 CPU 推理，查看节点五维特征与筛查分数。
- 上传文本型 PDF / UTF-8 TXT，通过 DeepSeek 提取结构化风险图。
- 导入、导出单报告 JSON，查看原文及归档实验指标。
- 默认免登录访问，支持通过环境变量开启 HTTP Basic 登录。

## 快速启动

需要 Python 3.13。Windows 可双击 `启动网站.bat`；第一次启动会安装依赖。

也可在项目根目录执行：

```powershell
git clone https://github.com/joyceqi7/safety-web.git
cd safety-web
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

打开 [本地网站](http://127.0.0.1:8765/)。Linux 可用 `python3.13 -m venv .venv`，并将上述 Python 路径替换为 `.venv/bin/python`。

Windows 如提示 `c10.dll` 或 Visual C++ 运行库缺失，请安装 [Microsoft Visual C++ Redistributable x64](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)。

## 配置

真实配置保存在 `.env`，仓库只包含空密钥的 `.env.example`。

| 变量 | 用途 |
| --- | --- |
| `APP_AUTH_ENABLED` | 默认 `false`，无需登录；设为 `true` 可开启验证 |
| `APP_USERNAME` / `APP_PASSWORD` | 开启登录时使用；生产环境启用验证后必须设置密码 |
| `LLM_BASE_URL` | DeepSeek API 地址 |
| `LLM_MODEL` | 默认 `deepseek-v4-pro` |
| `LLM_API_KEY` | 服务端 API 密钥，新报告提取时需要 |
| `TORCH_THREADS` | CPU 推理线程数，默认 2 |

样本浏览和模型推理无需 API 密钥。点击“提取风险图”会将报告正文发送到配置的大模型服务并产生 API 用量。免登录模式下，访问者也可使用此功能。

PDF 最多 100 页、10 MB；扫描件需先 OCR。提取正文限制为 50–12,000 字符，超长报告需手动节选。上传内容和新结果不持久化保存，刷新前请导出 JSON。

## 项目结构

```text
app/                     FastAPI 接口、数据校验、推理与静态前端
paper/                   原论文代码、处理后数据、模型权重及实验结果
  model_source/          六种模型实现
  model_weights/         原始训练权重
  processed_data/        提取文本、JSON 与图数据
  experiments/           补充实验脚本与归档结果
tests/                   接口测试和手动 API 检查
deploy/                  Docker、Windows 后台运行与部署说明
.env.example             配置模板（不含密钥）
requirements.lock.txt    已验证依赖版本，CPU Torch 单独安装
requirements-dev.txt     开发测试依赖
```

`paper/` 保留原代码包结构，复现实验见 [论文代码说明](paper/README.md)。实验重跑会覆盖其结果目录；网站推理不会重训练或修改权重。原始 PDF 和 Word 论文未收录。

## 验证

安装运行依赖后执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

18 项测试覆盖六模型真实推理、标签不参与模型输入、非法图校验、文件读取、可选登录和免登录访问。

手动 API 检查（第二条会使用付费 API，发送脚本中的虚构报告）：

```powershell
.\.venv\Scripts\python.exe -m tests.check_provider
.\.venv\Scripts\python.exe -m tests.check_live_extraction
```

Windows 阿里云原生部署已验证公网 HTTPS、六模型推理和 DeepSeek 提取。Docker / Linux 方案提供配置，尚未执行容器验收。早期验收快照保留当时的登录状态；当前线上已切换为免登录。

## 打包与部署

```powershell
.\.venv\Scripts\python.exe deploy/package.py
```

在项目上一级生成 `SafetyWeb-deploy.zip`。虚拟环境、真实 `.env`、临时脚本、浏览器会话、日志和登录信息均不进入 Git 或部署包。服务器更新时保留 `.env` 和 `runtime/` 中的配置与证书。

## 研究边界

模型分数是高风险类别输出，不是事故发生概率。原始标签是规则引导的代理标签，特征与标签存在依赖；图中关系是待复核线索，不能视为已验证的因果机制。归档实验指标不等于当前网页的新评估结果。本项目用于知识组织、候选风险筛查与专家复核，不替代独立安全评估。
