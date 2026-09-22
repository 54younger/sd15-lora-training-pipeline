# LoRA Training Studio 浏览器工作台

英文网站提供 **Prepare Dataset → Train Model → Evaluate → Publish & Download** 四步向导，每一步完成后等待用户确认。关闭页面不会终止已开始的后台工作；重新输入同一 owner 的访问令牌后，可从 Training Runs 找回任务。

## Docker 部署

在仓库根目录执行；已有 `.env` 时保留它：

```bash
test -f .env || cp .env.example .env
openssl rand -hex 32
```

将生成的随机字符串填入 `.env` 的 `LORA_API_KEYS`，例如 `{"你的随机令牌":"owner-a"}`。浏览器登录使用其中的令牌，不是整段 JSON。网站不需要 `LORA_ADMIN_KEYS` 管理员权限。请勿提交 `.env`。

如果 Docker socket 需要 `sudo`（例如构建镜像时使用了 `sudo`），本指南中的所有 `docker compose` 命令都必须加上 `sudo`，包括 `up`、`ps`、`logs` 和 `down`。请使用 `sudo docker compose ...`；不要使用 `sudo dc`，因为 `dc` 仅是详细 CLI 指南中的 shell 函数。

真实 SD 1.5 训练需要 NVIDIA GPU、驱动和 NVIDIA Container Toolkit：

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

无 GPU 的 CPU 技术演示：

```bash
docker compose up --build -d
```

打开 **http://localhost:8080**，输入访问令牌。CPU 模式明确标为 Demo/Test Only，只验证流水线，不产生真实 SD 1.5 评估图片或质量结论。

部署包含 `web`（页面及同源 `/api` 代理）、`api`、`worker`。原有 API 入口 `http://localhost:8000` 继续可用。`pipeline-data` 保存数据库、上传和产物，`model-cache` 保存模型权重。

GPU 首次执行可能下载 SD 1.5、CLIP，以及启用 BLIP 时的相应权重。确保 worker 能访问 Hugging Face；需要认证时在 `.env` 设置可选 `HF_TOKEN`，仅 worker 接收它。下载时间不代表训练吞吐。基础模型、硬件、精度、caption 和质量策略由服务端 profile 决定。

网站默认只绑定本机，可用 `LORA_WEB_PORT` 修改端口。小团队远程访问应配置 HTTPS 反向代理，再按需调整 `LORA_WEB_BIND`。图片 SHA-256 使用 Web Crypto，需要 HTTPS 或 localhost 安全上下文。

## 四步操作

### 1. Prepare Dataset

在 New Training 中选择 profile，输入 trigger token，批量选择或拖入图片。支持 JPEG、PNG 和静态 WebP；默认 100–1,000 张、单张至多 20 MiB、合计至多 2 GiB，实际限制以页面读取的 profile 为准。

可导入文件名到 caption 的 `captions.json`：

```json
{
  "photo-001.jpg": "a ceramic cup on a wooden table",
  "photo-002.png": "a blue ceramic bowl"
}
```

文件名必须唯一，开始处理前可逐张修改 caption；缺失值使用服务端配置的生成方式。处理包括哈希计算、上传、验证、去重、训练/验证集划分及 caption 生成。移除重复或无效图片后仍需达到有效图片和分组要求。

完成后查看统计、预览和 manifest，再进入训练。数据提交后冻结，修改需新建数据集。上传失败可重传；刷新后可找回已创建数据集，重新选择原文件并核对哈希，仅补传缺失文件。尚未提交的本地文件不会永久保存在浏览器中。

### 2. Train Model

设置训练步数和 learning rate；高级设置提供 rank、alpha、batch size、gradient accumulation、checkpoint interval 和 seed。启动后本阶段参数冻结。

页面展示实际阶段、步数、最新 loss、吞吐、内存和 checkpoint 信息（以实际后端字段为准）。排队/模型加载不会伪造百分比。任务可取消；自动重试及 checkpoint 恢复沿用 worker 的持久化机制。

### 3. Evaluate

训练完成后配置 prompts、seeds、inference steps、guidance scale，再启动评估。真实 GPU 模式以相同条件生成基础模型/LoRA 配对图片，展示指标与技术/质量检查。

未校准质量策略时，技术成功仍为 UNVERIFIED；技术失败或质量拒绝不能绕过。CPU tiny 的语义指标和配对图片不可用。

### 4. Publish & Download

检查评估结果并显式发布。发布会校验 adapter、manifest、报告及校验和，再注册模型。下载未经质量验证的模型需显式确认；可下载 adapter、JSON 报告和相关 manifest。

失败/取消任务保留诊断信息，可使用已验证数据集创建新任务；网页不回退已冻结阶段，不接受任意服务器 checkpoint 路径。

## 运维与兼容性

- `WAITING_FOR_USER` 不占 GPU，不消耗执行超时或 GPU 秒预算，但仍计入活动任务额度；不用的任务可取消。
- System Status 显示 API、数据库、存储、worker 状态；页面可访问不代表 GPU 就绪。
- 401：重新输入有效令牌；令牌保存在当前浏览器会话，退出清除缓存，历史由服务端按 owner 保存。
- 409：另一页面可能已经推进/取消任务，刷新查看最新状态。
- 422：按数据/字段错误修正输入；界面兼容业务错误和 FastAPI 校验格式。
- OOM 或模型下载失败会明确报告，不会自动降低参数或切换到 CPU 演示后端。

```bash
docker compose ps
docker compose logs --tail=200 web api worker
docker compose down
```

GPU 生命周期命令也需带 `-f compose.yaml -f compose.gpu.yaml`。`down` 保留卷，`down -v` 删除数据库、产物和缓存，不能当作重启命令。

旧客户端省略 `execution_mode` 时仍自动执行；网站使用 `manual` 模式。`POST /v1/training-jobs/{id}/advance` 使用 Bearer 和 `Idempotency-Key`，提交 `stage`，以及 TRAIN 的 `training_overrides` 或 EVALUATE 的 `evaluation_overrides`。参数只在对应阶段首次入队时冻结，重放不重复调度，越级与冲突请求被拒绝。

新增 owner 隔离的 datasets/jobs 分页列表、文件上传状态、轻量任务摘要和登记产物接口。产物通过服务端 ID 解析，不接受任意路径；精确字段见运行时 `/docs`。

升级采用 SQLite 增量迁移，已有任务默认自动模式。备份 `pipeline-data` 后同时重建 API/worker，避免新旧版本混用。验证记录见 [VALIDATION.md](../VALIDATION.md)。

## 前端开发与浏览器回归

本地开发需要 Node 22.12+，在 `frontend` 中运行 `npm ci`、`npm run dev`。默认代理本机 8000 API；`npm run build` 包含 TypeScript 检查，`npm test` 执行前端单元测试。

浏览器回归使用真实 API 与独立 CPU worker，而非模拟接口：

```bash
python scripts/web_e2e_fixture.py --metadata-path /tmp/studio-fixture.json
```

该命令保持运行，生成 100 张合成图片并输出测试路径/令牌。另一个终端用 `VITE_API_TARGET=http://127.0.0.1:8765 npm run dev` 启动前端，按 [frontend/README.md](../frontend/README.md) 设置 `STUDIO_E2E_*` 后执行 Playwright。fixture 使用新建临时目录，停止后保留图片、数据库与日志供检查。
