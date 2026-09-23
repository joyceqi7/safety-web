# 阿里云部署说明

当前网站已按要求开放访问：无需用户名或密码。`APP_AUTH_ENABLED=false` 时页面、模型筛查和报告提取均可直接使用；如需恢复登录，可设为 true 并配置 APP_PASSWORD。


当前实例已通过 Windows 原生方案部署到 https://47.99.242.201 ，见 [Windows 运维记录](WINDOWS-ALIYUN.md)。以下保留为 Linux / Docker 的替代部署方案。

## 部署前检查

现有服务器是 ECS 还是轻量应用服务器、Linux 发行版及版本、CPU 架构、公网 IP、SSH 用户及端口、已配置的 SSH 密钥/别名、是否有域名，以及现有 80/443 端口是否被其他网站占用。不要把私钥、服务器密码或 API Key 提交到代码仓库。

本方案使用 Linux x86_64 + Docker Engine + Compose v2；可先按 2 核 / 4 GB 内存规划单团队 CPU 演示，实际资源需求需在现有服务器上确认。它不会自动购买或修改阿里云资源。已有其他网站时，先确认端口与反向代理接入方式，不要停止已有站点。

## 1. 上传

将 SafetyWeb-deploy.zip 上传到服务器新目录（例如 `/opt/safety-web`）并解压。压缩包不包含本机虚拟环境、日志、密钥。服务器需要能访问配置的镜像仓库、Python 包源和 DeepSeek API；若镜像下载受限，可在可联网构建机打包镜像后导入，不能把镜像下载失败误判为应用故障。

## 2. 配置

在项目根目录：

```bash
cp .env.example .env
chmod 600 .env
# 用服务器编辑器填写 .env，开启登录时需设置 APP_PASSWORD。
# LLM_API_KEY 填写后才能对新报告调用 API；样本推理不需要它。
```

默认 APP_AUTH_ENABLED=false，页面和 API 无需登录。设为 true 后，生产环境必须配置 APP_PASSWORD，用户名默认 admin。

### 暂时没有域名：通过 SSH 隧道验收

保持默认 `SITE_ADDRESS=:80` 和 `BIND_ADDRESS=127.0.0.1`，可把 `HTTP_PORT=8080` 避开服务器已有站点。在本机建立隧道：

```bash
ssh -L 8766:127.0.0.1:8080 SSH用户名@服务器IP
```

然后打开 http://127.0.0.1:8766 ；只有开启验证时才需要登录。

### 使用域名公开访问

将已具备使用条件的域名 DNS A 记录指向服务器，设置：

```dotenv
SITE_ADDRESS=safety.example.com
BIND_ADDRESS=0.0.0.0
HTTP_PORT=80
HTTPS_PORT=443
```

将示例域名替换为实际域名。Caddy 申请和续期证书需公网可达的域名及可用的 80/443 端口；上线时需确认域名在当前服务器地域的接入条件。TLS 成功后使用 `https://实际域名`。不要将 API 密钥放进浏览器或前端代码。

阿里云安全组（轻量服务器则检查对应防火墙）和系统防火墙需允许所需的 80/443 入站流量；SSH 仅向管理来源开放。8000 是容器内部端口，不需要对公网开放。官方说明：
https://www.alibabacloud.com/help/zh/ecs/user-guide/start-using-security-groups

## 3. 启动与验收

确认 Docker Engine 和 Compose v2 已安装。首次部署：

```bash
bash deploy/deploy.sh
```

也可手动运行：

```bash
docker compose config --quiet
docker compose up -d --build --wait --wait-timeout 300
docker compose ps
docker compose logs --tail 100 app proxy
```

避免运行会展开并打印 `.env` 密钥的 `docker compose config`（不带 `--quiet`）。

验收包含：容器 healthy、HTTPS 证书、免登录模式的样本查询（开启验证时再检查 401）、六模型推理、导入与导出，以及使用虚构报告确认 API 提取。`/healthz` 仅报告基本存活和样本数，不加载全部模型；必须另测推理。

## 4. 更新和恢复

更新前保留旧项目目录和正在运行的镜像 ID。将新包解压到独立目录，保留服务端 `.env`，先构建成功再切换。不要覆盖正在使用的环境配置。

```bash
docker compose build app
docker compose up -d --wait --wait-timeout 300
```

停止本站：`docker compose down`。不要加 `-v`，否则会删除 Caddy 的证书卷。回滚需使用之前保留的镜像/目录和同一服务端配置。

## 实际验证范围

本地 Windows Python 3.13 的服务、六个原始权重、18 项测试、浏览器桌面/手机流程及真实 DeepSeek 提取已经验证。本机未安装 Docker；Linux 容器方案尚未验收；当前 Windows 阿里云实例的公网 HTTPS 和实际推理已验收，详见 Windows 部署记录。
