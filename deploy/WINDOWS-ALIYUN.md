# Windows 阿里云部署记录

当前网站已按要求开放访问：无需用户名或密码。`APP_AUTH_ENABLED=false` 时页面、模型筛查和报告提取均可直接使用；如需恢复登录，可设为 true 并配置 APP_PASSWORD。


目标实例：i-bp145hgmtim756kd8yak，华东 1（杭州）。公网地址：47.99.242.201。

部署采用 Windows Server 2022 原生进程，不更换操作系统，不使用 Linux Docker。

- 项目目录：`C:\SafetyWeb`
- Python 3.13.15：`C:\SafetyRuntime\Python313`
- 独立环境：`C:\SafetyWeb\.venv`
- Caddy 2.11.4：`C:\SafetyRuntime\Caddy`
- 配置：`C:\SafetyWeb\.env` 和 `C:\SafetyWeb\Caddyfile`
- 后台入口：`C:\SafetyWeb\run_service.py`
- Windows 启动任务：`SafetyWeb`，以 LocalService 运行
- 日志目录：`C:\SafetyWeb\runtime\logs`

公网仅提供 80 / 443，应用监听 `127.0.0.1:8000`。通过 Caddy 的 ACME shortlived 配置申请 IP 地址证书并自动续期；每次部署验收必须验证实际 TLS 证书，而不能只看配置文件。

维护命令（在服务器管理员 PowerShell 中执行）：

```powershell
Get-ScheduledTask -TaskName SafetyWeb
Get-Content C:\SafetyWeb\runtime\logs\app.log -Tail 50
Get-Content C:\SafetyWeb\runtime\logs\proxy.log -Tail 50
Invoke-RestMethod http://127.0.0.1:8000/healthz
```

上传新版本前备份应用目录。配置文件和 runtime 中的证书不能覆盖。API Key 和网站登录密码仅保存在服务端 `.env`；本地登录信息另存于桌面 `SafetyWeb-access.txt`，不放入部署包。

获得域名后，将域名解析至公网 IP，并修改 Caddyfile 的站点地址，再使用 `caddy reload --config C:\SafetyWeb\Caddyfile --adapter caddyfile` 应用配置。

原始代码包和本机 `.venv` 均保留。临时传输监听、临时证书绑定和传输专用安全组规则会在启动正式服务前撤销。

## 2026-09-22 公网验收

已验证 HTTP 308 跳转、可信 HTTPS、未登录返回 401、登录后网页显示、103 份样本查询、六模型推理、DeepSeek 虚构报告提取及推理、配置文件不对公网提供。IP 证书由 Let's Encrypt 签发，Caddy 管理自动续期。后台任务以 LocalService 运行并设置开机触发；未为了测试而重启整台服务器。

运行依赖另包含 Microsoft Visual C++ Redistributable，已从微软官方下载并验证签名后安装。
