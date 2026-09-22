# Automatic LoRA Training Pipeline

[English](README.md) · [Docker CLI 详细指南](docs/07-docker-cli-training.md) · [浏览器工作台指南](docs/06-web-studio.md)

本项目将 100–1,000 张图片训练为 Stable Diffusion 1.5 attention LoRA，并提供可持久化的四步流程：**准备数据 → 训练 → 评估 → 发布与下载**。推荐从英文浏览器工作台 `http://localhost:8080` 开始使用。

## 使用 Docker 快速启动

需要 Docker Engine 和 Docker Compose v2。真实 SD 1.5 训练还需要 Linux/WSL2 的 NVIDIA GPU 主机、兼容驱动和 NVIDIA Container Toolkit。下方 CPU 方式仅用于技术演示。

创建本地配置，然后将 `LORA_API_KEYS` 中的示例令牌替换为长随机值：

```bash
cp .env.example .env
openssl rand -hex 32
```

例如在 `.env` 中设置 `LORA_API_KEYS={"your-random-token":"owner-a"}`。网站登录时填写 `your-random-token`：它是 JSON 映射的**键**，不是整段 JSON。不要提交 `.env`。

如果 Docker socket 需要 `sudo`（例如 build 时使用了 `sudo`），本页后续所有 `docker compose` 命令都必须加上 `sudo`，包括 `up`、`ps`、`logs` 和 `down`。请使用 `sudo docker compose ...`；不要使用 `sudo dc`，因为 `dc` 只是详细 CLI 指南中的 shell 函数。

启动真实 GPU 服务（所有 GPU 生命周期命令都必须同时使用两个 Compose 文件）：

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

如果已经执行过 build，则不重新构建，直接启动：

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

CPU 技术演示：

```bash
docker compose up --build -d
```

打开 `http://localhost:8080`，输入令牌后按页面完成四步：

1. **Prepare Dataset**：上传图片，导入/编辑 caption，验证并冻结数据集。
2. **Train Model**：设置受限训练参数，查看 worker 的真实进度。
3. **Evaluate**：执行固定条件检查，并在支持时查看生成对比。
4. **Publish & Download**：发布通过检查的产物，下载 adapter、报告和 manifest。

部署会一起启动网页、API 和 worker。API 仍可通过 `http://localhost:8000` 访问；网页通常通过同源 `/api` 代理它。

## 日常运维

GPU 部署时请始终保留两个 `-f` 参数：

```bash
docker compose -f compose.yaml -f compose.gpu.yaml ps
docker compose -f compose.yaml -f compose.gpu.yaml logs --tail=200 web api worker
docker compose -f compose.yaml -f compose.gpu.yaml down
```

`down` 会保留 named volumes。普通重启不要使用 `down -v`：它会删除数据库、上传数据、checkpoint、adapter、报告和模型缓存。

## Docker CLI 训练

浏览器是最简便的操作入口；原有命令行流程仍保留，适合可复现的 GPU 验收，包括 IBean 准备/caption、固定模型版本、训练/恢复、评估、API 检查和 benchmark。请查看 [Docker CLI 详细指南](docs/07-docker-cli-training.md)：从 GPU preflight 开始，先执行 `prepare`/`caption`，再使用一次性 worker 容器训练。

## 文档与验证边界

- [浏览器工作台指南](docs/06-web-studio.md)：页面部署、四步操作、恢复与排障。
- [Docker CLI 详细指南](docs/07-docker-cli-training.md)：完整真实 GPU runbook、API 和 benchmark 流程。
- [运行与 API 指南](docs/04-running-and-api.md)：运维和合约参考。
- [文档索引](docs/00-assignment-guide.md)：架构、技术规格、取舍和证据导航。
- [验证记录](VALIDATION.md)：本仓库环境中实际执行的检查及其边界。Python/CPU 检查不能证明真实 GPU、SD 1.5 质量、Docker 运行时或 benchmark 结果。
