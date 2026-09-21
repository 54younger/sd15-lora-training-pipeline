# 交付验证记录 / Validation record

验证日期：2026-09-21。所有测试使用临时生成的图像，不依赖用户数据集。

## 已验证 / Verified

- 最终全量测试：`63 passed, 1 warning in 25.83s`。警告来自验证环境已有的 AutoAWQ 弃用提示，不是测试失败；该耗时不作为训练性能基准。
- 离线 CPU tiny 后端执行真实 Diffusers/PEFT LoRA 训练，不下载预训练权重。
- CPU CLI smoke：生成 120 张图像，完成清洗、caption、分组划分；第 2 步保存并中断，第 4 步恢复完成；adapter 加载和技术评估通过。
- HTTP/ASGI + 独立 worker 子进程端到端测试：上传、验证、训练、评估、发布、租户隔离与未验证模型下载门禁。
- 数据边界、checkpoint 恢复、质量门禁、幂等事务、调度容量、租约 fencing、取消竞争和 GPU 时间预算回归测试。
- Ruff 覆盖 `src`、`tests` 和 `scripts/api_demo.py`；Mypy 仅覆盖 `common.py` 与 `config.py`，并非全项目严格类型检查。
- Python wheel 构建；CPU/GPU 两套 Docker Compose 配置解析。
- 三张 PlantUML 图通过语法检查并重新渲染为 PNG 和 SVG。

CPU smoke 的实际结果为 `technical_pass=true`、`quality_status=UNCALIBRATED`、`test_only=true`。这不是 SD 1.5 视觉质量认证，也不是 GPU 性能基准。

## 环境与复现 / Environment and reproduction

验证环境为 Linux/WSL2、Python 3.12、PyTorch 2.8.0、Diffusers 0.35.1、PEFT 0.17.1。使用隔离虚拟环境，未修改用户现有 Conda 环境。

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q
python -m ruff check src tests scripts/api_demo.py
python -m mypy
HF_HUB_OFFLINE=1 lora-pipeline cpu-smoke --output /tmp/lora-new-smoke
```

当前执行沙箱会阻塞包括空 FastAPI 应用在内的 TestClient 请求，因此服务测试在批准的沙箱外环境中运行，使用本地 ASGI 和临时文件，不访问外部服务。

## 尚未验证 / Not verified

- Docker daemon 的操作系统 socket 权限不可用：未执行镜像构建或容器端到端运行。Compose 配置检查不能代替容器运行测试。
- 未运行真实 SD 1.5 GPU smoke，也未验证 RTX 4060 Ti 16GB 的实际显存峰值或吞吐率。请按运行指南在本地 GPU 上执行。
- 未校准质量阈值，未声称任何真实生成质量合格；默认成功模型是 `UNVERIFIED`，必须显式允许下载。
- 性能交付提供指标定义和采集协议；未测量的 GPU 性能字段保持 `not_measured` 或 `null`，不以 CPU 测试耗时代替。

See [English run guide](docs_en/04-running-and-api.md) and [benchmark protocol](docs_en/05-performance-benchmarks.md). Real GPU and container execution remain separate target-environment acceptance checks.
