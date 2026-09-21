# Automatic LoRA Training Pipeline

这是 Automatic LoRA Training Pipeline 作业的单机实现。流水线接收 100–1,000 张用户图片，完成验证、caption、分组划分、Stable Diffusion 1.5 attention LoRA 训练、checkpoint 恢复、评估、发布和下载。FastAPI 与独立 worker 用 SQLite 和本地制品目录协作；目标主机上的 worker 按物理 GPU UUID 分配 slot，让 2–4 张 GPU 可以安全承接并发任务。

本文按两个 assignments 编排。Part 2 以真实 GPU Docker 验证为主线：每个模块都有职责、命令、制品与成功判据，最后用三次真实训练得到 benchmarks。tiny CPU 后端仅用于离线回归，不能作为 SD 1.5 质量或 GPU 性能结论。

当前开发环境已完成 Python 测试、CPU tiny smoke、Compose 配置解析；Docker daemon 和真实 GPU 未在此环境运行。GPU 命令是目标主机验收 runbook，未测字段必须保持 null/not_measured。完整边界见 [VALIDATION.md](VALIDATION.md)。

## 交付物映射


| 作业交付                           | 仓库证据                                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| Part 1 架构图、数据流、容错        | [系统架构](docs/01-system-architecture.md)、[图源和图像](diagrams/)                              |
| Part 1 组件、API、schema、错误策略 | [技术规格](docs/02-technical-specification.md)、[设计取舍](docs/03-implementation-tradeoffs.md)  |
| Part 2 数据、训练、评估、REST API  | src/lora_pipeline 中 data、captions、training、evaluation、api、worker、store 与下方 Docker 验证 |
| Docker、API、测试                  | Dockerfile、compose.yaml、compose.gpu.yaml、[运行/API 指南](docs/04-running-and-api.md)、tests/  |
| 性能指标与基准                     | 下方 Benchmarks、[指标定义](docs/05-performance-benchmarks.md)                                   |

## Part 1：System Design

~~~mermaid
flowchart LR
  U[用户图片/可选 caption] --> API[FastAPI API]
  API -->|认证、owner、幂等、配额| DB[(SQLite durable queue)]
  API --> OBJ[/data objects/]
  DB --> W[独立 worker]
  W --> S{UUID GPU / CPU slots}
  S --> D[VERIFY → PREPARE]
  D --> C[CAPTION]
  C --> T[TRAIN]
  T --> E[EVALUATE]
  E --> P[PUBLISH]
  D --> ART[/data manifests + artifacts/]
  T --> ART
  E --> ART
  P --> R[model registry + LoRA adapter]
  M[/models cache/] --> C
  M --> T
  M --> E
~~~

上传 metadata 与字节先写入持久目录；完成上传后 worker 执行 VERIFY → PREPARE → CAPTION → TRAIN → EVALUATE → PUBLISH。每阶段用 SHA-256、冻结 manifest 与原子制品发布衔接。attempt、lease、heartbeat、fencing token 保存在 SQLite，worker 重启可以重新领取工作，旧 attempt 不能覆盖新结果。


| 组件                 | 设计与生产约束                                                                                                                                              |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| API / Store          | Bearer token 映射 owner；变更请求带 Idempotency-Key；流式上传校验声明的大小/SHA-256；SQLite 保存队列、任务和调度状态。                                      |
| Data / Captions      | 支持 JPEG、PNG、静态 WebP；默认单图 20 MiB/40 MP、最短边 256，数据集 100–1,000 图/2 GiB。EXIF 归一化、去重、group-wise 约 90:10 split；用户 caption 优先。 |
| Training             | sd15 冻结 VAE、text encoder、UNet 基座，只训练 attention LoRA。checkpoint 含 adapter、optimizer、scheduler、scaler、RNG、cursor 和兼容性身份。              |
| Evaluation / Publish | 重载 adapter 做技术检查；真实 SD 1.5 在固定 prompt/seed 下产生 base/adapter 配对与 CLIP 诊断。未校准 policy 的成功是 COMPLETED_UNVERIFIED/UNVERIFIED。      |
| 调度/容错            | 每张物理 GPU 一个 OS 锁；owner 轮转、FIFO、时限、GPU 秒预算、取消与重试处理竞争。SQLite/本地卷是单机共同故障域；多机须换共享 DB、对象存储和远程执行器。     |

REST API 提供 dataset 上传、training profile/job、evaluation/model 下载、health 和 metrics。错误 JSON 固定含 code、message、retryable、details、request_id；精确 schema 见运行时 /docs 和 [技术规格](docs/02-technical-specification.md)。

## Part 2：GPU Docker 前提

目标主机需要 Linux/WSL2、Docker Engine、Compose v2、NVIDIA driver、NVIDIA Container Toolkit。GPU overlay 安装 CUDA PyTorch、关闭 fake/test backend，只给 worker 请求 GPU；API 不使用 CUDA。SD 1.5、BLIP、CLIP 权重缓存在 model-cache volume，首次下载不计入训练基准。

在仓库根目录定义 helper。每个新 shell 都要重新定义，避免用 CPU Compose 配置重建 GPU 服务。`dc` 只是当前 Bash 会话里的快捷函数，不是 Docker 的子命令；如果使用了新终端，需要重新执行定义。PowerShell 请使用：`function dc { docker compose -f compose.yaml -f compose.gpu.yaml @args }`。

如果当前用户没有访问 Docker socket 的权限，把下面函数中的命令写成 `sudo docker compose ...`。不要执行 `sudo dc build`，因为 `sudo` 不会查找当前 shell 函数。长期使用可以在宿主机执行 `sudo usermod -aG docker "$USER"`，重新登录（或执行 `newgrp docker`）后直接使用 Docker；docker 组通常具有近似 root 的主机权限，只应加入可信用户。

~~~bash
# 如果当前用户已能访问 Docker socket，使用这一行：
dc() { docker compose -f compose.yaml -f compose.gpu.yaml "$@"; }
# 如果必须使用 sudo，改为：
# dc() { sudo docker compose -f compose.yaml -f compose.gpu.yaml "$@"; }
if [ ! -e .env ]; then
  export DEMO_TOKEN="$(openssl rand -hex 32)"
  cp .env.example .env
  sed -i "s/replace-with-long-random-token/$DEMO_TOKEN/g" .env
else
  # Replace with the existing .env API/admin token in each new shell.
  export DEMO_TOKEN='paste-the-token-from-.env'
fi
dc config --quiet
# BuildKit is required for the persistent pip cache mount. This must print a version.
docker buildx version
dc build --progress=plain
dc run --rm --no-deps worker lora-pipeline preflight
~~~

preflight 必须报告 cuda_available 为 true、预期 GPU 和显存；否则停止。pipeline-data 保存 SQLite、上传、manifest、checkpoint、adapter、报告，model-cache 保存权重。可用 .env 的 LORA_GPU_UUIDS 限制可调度卡，每个实际 UUID 只有一个 slot。

模块 1–3 只使用一次性容器，按顺序独占 GPU；不要先启动常驻 worker。模块 4 才启动 API/worker。

### 推荐的本地验证数据集：IBean

为了让模块验证可以复现，仓库已下载 [IBean bean disease dataset](https://github.com/AI-Lab-Makerere/ibean)。它采用 MIT 许可，原始训练集有 1,034 张 500×500 JPEG；本地副本整理为 999 张，每个类别 333 张（`healthy`、`angular_leaf_spot`、`bean_rust`），符合本项目 100–1,000 张和最短边至少 256 的输入限制。类别名被转换为简短 caption，保存在 `datasets/ibean/images/captions.json`。这组图片适合验证解码、分组切分、caption、LoRA checkpoint、评估和 GPU 资源使用；它不是经过校准的风格质量基准。

当前本地文件如下：

```text
datasets/ibean/raw/train.zip       143,812,152 bytes
datasets/ibean/images/*.jpg       999 images, 333 per class
datasets/ibean/images/captions.json
```

训练集压缩包 SHA-256 为 `284fe8456ce20687f4367ae7ad94a64577e7f9fde2c2c6b1c74340ab5dc82715`。如果需要在另一台机器重新下载，官方原始 Google Storage 地址目前可能返回 403，使用 Hugging Face 镜像并校验哈希：

```bash
mkdir -p datasets/ibean/raw datasets/ibean/images
curl -fL --retry 2 \
  -o datasets/ibean/raw/train.zip \
  https://huggingface.co/datasets/beans/resolve/main/data/train.zip
echo '284fe8456ce20687f4367ae7ad94a64577e7f9fde2c2c6b1c74340ab5dc82715  datasets/ibean/raw/train.zip' | sha256sum -c -
unzip -q -o datasets/ibean/raw/train.zip -d datasets/ibean/raw
```

上面的压缩包包含 1,034 张训练图片，而本项目的上限是 1,000 张。下面的脚本从三个类别各取 333 张，并重新生成本地 caption sidecar；它只复制文件，不修改原始压缩包：

```bash
python - <<'PY'
import json, shutil
from pathlib import Path

raw = Path("datasets/ibean/raw/train")
out = Path("datasets/ibean/images")
out.mkdir(parents=True, exist_ok=True)
phrases = {
    "healthy": "healthy bean leaf",
    "angular_leaf_spot": "bean leaf with angular leaf spot",
    "bean_rust": "bean leaf with bean rust",
}
captions = {}
for label, phrase in phrases.items():
    for source in sorted((raw / label).glob("*.jpg"))[:333]:
        target = out / f"{label}__{source.name}"
        shutil.copy2(source, target)
        captions[target.name] = f"a close-up photograph of a {phrase}"
(out / "captions.json").write_text(json.dumps(captions, indent=2) + "\n")
print(f"prepared {len(captions)} images")
PY
export DATASET_DIR="$PWD/datasets/ibean/images"
```

数据集来源、许可证和原始类别说明见 [IBean 官方仓库](https://github.com/AI-Lab-Makerere/ibean)；Hugging Face 镜像仅作为可复现下载入口。使用这些图片进行公开发布或商业训练前，应同时检查上游数据集条款和图片来源。

### 模块 1：Data Processing

**对应要求：** 预处理/验证、caption generation、split/augmentation、质量筛选。

prepare 解码和规范化图片，拒绝坏文件、动画、超限和精确重复；近重复 group 不跨 train/validation。它把规范化 PNG 写进 /data/real-prepared/images，之后训练不再需要宿主机图片挂载。默认需要 100 有效唯一图、80 train、10 validation、每侧至少两个 group；模糊/亮度/对比度只产生 warning。

将真实图片只读挂载。可选 captions.json 为 filename 到 caption 的对象；没有 sidecar 时使用 template。下面只生成输入清单，原图不会被修改或转换。

~~~bash
# 默认使用本仓库前面准备好的 IBean 图片；替换 DATASET_DIR 可验证自己的数据集。
# Docker bind mount 的源路径必须是绝对路径，因此这里使用 "$PWD/..." 展开绝对路径。
export DATASET_DIR="${DATASET_DIR:-$PWD/datasets/ibean/images}"
test -d "$DATASET_DIR" || { echo "DATASET_DIR does not exist: $DATASET_DIR" >&2; exit 1; }
prepare_ibean_data() {
# heredoc 通过 stdin 传入脚本；-T 禁用 Compose 默认分配的伪终端，兼容 SSH/CI/重定向执行。
dc run --rm --no-deps -i -T -v "$DATASET_DIR:/input:ro" worker python - <<'PY'
import json
from pathlib import Path
root = Path("/input")
captions = json.loads((root/"captions.json").read_text()) if (root/"captions.json").exists() else {}
paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".webp"})
if not 100 <= len(paths) <= 1000: raise SystemExit(f"expected 100..1000 images, found {len(paths)}")
items = [{"id":f"input-{i:04d}","name":p.name,"path":str(p),"caption":captions.get(p.name)} for i,p in enumerate(paths)]
Path("/data/real-files.json").write_text(json.dumps(items))
PY
# 只有上一步成功后才继续。若这里不存在，先处理上一步输出的首个错误。
dc run --rm --no-deps -T worker test -f /data/real-files.json || return 1
export DATA_PREPARATION_STARTED="$(date +%s)"
dc run --rm --no-deps -v "$DATASET_DIR:/input:ro" worker lora-pipeline prepare \
  --files /data/real-files.json --output /data/real-prepared || return 1
export DATA_PREPARATION_SECONDS="$(( $(date +%s) - DATA_PREPARATION_STARTED ))"
printf 'DATA_PREPARATION_SECONDS=%s\n' "$DATA_PREPARATION_SECONDS"
dc run --rm --no-deps worker lora-pipeline caption \
  --prepared /data/real-prepared/prepared.json --output /data/real-input \
  --mode template --trigger-token mystyle
}
prepare_ibean_data
unset -f prepare_ibean_data
~~~

这里的 `/data` 是**容器内路径**，由 Compose 挂载到 `pipeline-data` named volume，并不对应宿主机的 `/data` 目录。因此三个一次性容器会共享生成的 manifest，容器被 `--rm` 删除后文件仍保留在 volume 中，但不能直接在宿主机执行 `ls /data/real-input` 查找。`DATASET_DIR` 只需挂载到读取原图的前两个容器。

确认最终文件存在并只显示摘要：

~~~bash
dc run --rm --no-deps -T worker python -c '
import json
from pathlib import Path
p = Path("/data/real-input/training-input.json")
x = json.loads(p.read_text())
print({"path": str(p), "train": len(x["train"]), "validation": len(x["validation"]), "captioning": x["captioning"]})
'
~~~

如需把 manifest 导出到当前宿主机目录，使用标准输出重定向；`-T` 可以保证文件中不会混入 TTY 控制字符：

~~~bash
dc run --rm --no-deps -T worker \
  sh -c 'cat /data/real-input/training-input.json' > training-input.json
test -s training-input.json && echo "exported to $PWD/training-input.json"
~~~

不要删除 heredoc 命令上的 `-T`：若生成清单时出现 `the input device is not a TTY`，说明容器仍在尝试分配 TTY；该步骤失败后 `/data/real-files.json` 不会存在，继续运行就会连带出现 `/data/real-files.json` 和 `/data/real-prepared/prepared.json` 不存在。应从生成清单的命令重新执行，而不是手工创建这两个文件。

制品是 prepared.json、规范化图片与 training-input.json。最后一个文件冻结 split、checksum、处理版本、caption 来源和父 manifest；训练、恢复、评估/A-B 均引用它。用户 caption 优先，缺失项才用 template，并追加 trigger token。真实 GPU 可用 BLIP；缺模型时它明确失败，不会退回 template：

~~~bash
dc run --rm --no-deps worker lora-pipeline caption \
  --prepared /data/real-prepared/prepared.json --output /data/real-input-blip \
  --mode blip --device cuda --trigger-token mystyle --local-files-only
~~~

**成功判据：** prepare 报告 accepted unique、split 和 warnings，且 /data/real-input/training-input.json 存在；DATA_PREPARATION_SECONDS 是该 prepare 调用的墙钟秒数，后续 benchmark 会原样写入汇总。不合格数据应以 DATASET_TOO_SMALL 或 DATASET_UNSUITABLE 失败。训练默认等比 resize + center crop；random_crop/horizontal_flip 必须在训练 JSON 显式启用，validation 无随机增强。

规范化 PNG 使用最终编码字节的 SHA-256 内容寻址，新一次 prepare 不会覆盖旧 frozen manifest 引用的图片。升级到该实现前若训练报 `Frozen train image checksum does not match`，不要修改 manifest 或跳过校验：先重新 `dc build`，再完整重跑本模块的 `prepare_ibean_data`（包含 prepare 和 caption），最后重新开始训练。错误 details 中的 path、expected_sha256 和 actual_sha256 可用于确认损坏文件。

### 模块 2：Training

**对应要求：** 配置化 LoRA、GPU memory management、checkpoint/resume、progress monitoring。

sd15 是真实 Diffusers/PEFT 训练，基础权重冻结。configs/sd15-smoke.json 是 512px、batch 1、accumulation 4、rank/alpha 4、FP16、gradient checkpointing 的 10-step GPU 验证配置，不是质量或吞吐 benchmark。

训练是前台命令，只需要当前一个 terminal；命令返回前 shell 看起来被占用是正常现象。第二个 terminal 仅用于可选的 `watch -n 1 nvidia-smi` 监控。模块 1–3 不要提前启动常驻 worker，模块 4 才执行 `dc up -d api worker`。

SD 1.5 权重保存在 Compose 的 `model-cache` volume。首次训练前先在联网环境预热一次缓存，并把实际解析出的 immutable commit 写入本次训练配置；后续训练和恢复使用该 pinned、offline 配置，避免上游默认分支更新导致基座漂移：

~~~bash
dc run --rm --no-deps -i -T -v "$PWD/configs:/workspace/configs:ro" worker python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
repo = "stable-diffusion-v1-5/stable-diffusion-v1-5"
root = Path(snapshot_download(repo_id=repo))
revision = root.name
if len(revision) < 7:
    raise RuntimeError(f"snapshot did not resolve to an immutable commit: {root}")
config = json.loads(Path("/workspace/configs/sd15-smoke.json").read_text())
config.update({"model_name": repo, "revision": revision, "local_files_only": True})
Path("/data/sd15-smoke-pinned.json").write_text(json.dumps(config, indent=2) + "\n")
print(json.dumps({"snapshot": str(root), "revision": revision, "config": "/data/sd15-smoke-pinned.json"}))
PY

# 验证 pinned revision 在断网模式下也能从同一个 named volume 解析：
dc run --rm --no-deps -T worker python -c \
  'import json; from pathlib import Path; from huggingface_hub import snapshot_download; c=json.loads(Path("/data/sd15-smoke-pinned.json").read_text()); print(snapshot_download(repo_id=c["model_name"], revision=c["revision"], local_files_only=True))'
~~~

若这里下载失败，先修复 Docker 容器的 Hugging Face 网络、代理/DNS或磁盘空间。模型仓库需要认证时，在宿主机 `export HF_TOKEN=...`，并仅在预热命令的 `worker` 前增加 `-e HF_TOKEN`；不要把 token 写进镜像、配置或 README。完成预热后 pinned 配置只读本地缓存，训练命令无需继续传 token。`BASE_MODEL_UNAVAILABLE` 的 details 会报告已脱敏的模型名、revision、缓存目录、离线模式和底层原因。不要通过反复启动训练代替缓存预热。

~~~bash
dc run --rm --no-deps -v "$PWD/configs:/workspace/configs:ro" worker \
  lora-pipeline train --input /data/real-input/training-input.json \
  --config /data/sd15-smoke-pinned.json \
  --output /data/sd15-smoke-training --stop-after-step 5
dc run --rm --no-deps -v "$PWD/configs:/workspace/configs:ro" worker \
  lora-pipeline train --input /data/real-input/training-input.json \
  --config /data/sd15-smoke-pinned.json --output /data/sd15-smoke-training \
  --resume /data/sd15-smoke-training/checkpoints/step-00000005/checkpoint.json
~~~

CLI 在 stderr 逐行显示当前阶段、文本进度条、current/total 和百分比，包括 input validation、model loading、training、checkpoint、adapter saving；不需要另开 terminal 才能看到进度。每个训练 step 的后端 progress 仍记录 global_step、loss、samples_processed、samples_processed_this_run、samples_per_second、checkpoint_seconds、cpu_rss_bytes，以及 CUDA 的 gpu_memory_allocated/reserved，供 worker/API 查询。stdout 只保留最终结果 JSON，便于脚本解析；不要用会吞掉失败码的 pipe。training-result.json 记录 adapter checksum、配置、基座 revision/fingerprint、input hash、参数量与 elapsed_seconds。

**成功判据：** result 为 COMPLETED，global_step 为 10，并有 adapter/adapter.safetensors。输入/基座/数值配置不匹配为 CHECKPOINT_INCOMPATIBLE，损坏 checkpoint 为 CHECKPOINT_CORRUPT，OOM 为 GPU_OUT_OF_MEMORY；不会自动换参数重跑。

### 模块 3：Evaluation 与 A/B

**对应要求：** 质量评估、性能指标收集、A/B framework。

evaluate 先重新加载 adapter 做技术检查，再以固定 prompt、seed、steps、guidance 为 base/LoRA 生成配对输出。真实 SD 1.5 报告 CLIP prompt score、held-out similarity、diversity、max train similarity、base 对照、evaluation.html/json。项目没有 FID；这些诊断也不能单独证明风格质量。先建立与真实数据集的主体/风格匹配的 prompt 配置；下面的 portrait 文本只保证命令可运行，必须替换为该数据集的真实评估 prompts。

Evaluation 同样在 stderr 显示当前阶段和进度条，包括 technical smoke、base/adapter generation、CLIP model loading、generated/held-out/train reference scoring 和 report writing；stdout 只输出最终 evaluation JSON。模型加载阶段百分比保持在 0% 并不表示卡死，应等待进入逐图 generation/scoring；若失败则以非零退出并在 stderr 返回结构化 error。

~~~bash
dc run --rm --no-deps -i -T -v "$PWD/configs:/workspace/configs:ro" worker python - <<'PY'
import json
from pathlib import Path
config=json.loads(Path("/workspace/configs/evaluation-smoke.json").read_text())
config["prompts"]=["a portrait photograph in mystyle style", "a full-body portrait in mystyle style"]
Path("/data/real-evaluation.json").write_text(json.dumps(config))
PY
dc run --rm --no-deps -v "$PWD/configs:/workspace/configs:ro" worker \
  lora-pipeline evaluate \
  --training-result /data/sd15-smoke-training/training-result.json \
  --input /data/real-input/training-input.json --output /data/sd15-smoke-evaluation \
  --config /data/real-evaluation.json

dc run --rm --no-deps -i -T worker python - <<'PY'
import json
from pathlib import Path
x=json.loads(Path("/data/sd15-smoke-pinned.json").read_text())
x["learning_rate"]=0.0002
Path("/data/sd15-smoke-b.json").write_text(json.dumps(x))
PY
dc run --rm --no-deps worker lora-pipeline train \
  --input /data/real-input/training-input.json --config /data/sd15-smoke-b.json \
  --output /data/sd15-smoke-training-b
dc run --rm --no-deps -v "$PWD/configs:/workspace/configs:ro" worker \
  lora-pipeline compare \
  --left /data/sd15-smoke-training/training-result.json \
  --right /data/sd15-smoke-training-b/training-result.json \
  --input /data/real-input/training-input.json --output /data/sd15-smoke-ab \
  --config /data/real-evaluation.json
~~~

**成功判据：** evaluation.json 的 technical_pass 为 true，且有 paired_outputs/生成目录；comparison.json 的左右候选使用相同 paired conditions。默认 quality_policy 为 null 时 UNCALIBRATED 是预期，不能声称质量通过；只有校准 policy PASS 才能 READY。

### 模块 4：REST API、Health、Logging

**对应要求：** REST training management、health endpoints、错误处理和 logging。

CLI 验证结束才启动常驻服务。live 仅证明 API 可响应，ready 还要求 SQLite、数据目录和 worker heartbeat，初启时可能短暂 503。

~~~bash
dc up -d api worker
dc ps
curl --fail --silent http://127.0.0.1:8000/health/live
curl --silent --show-error http://127.0.0.1:8000/health/ready
curl --fail --silent -H "Authorization: Bearer $DEMO_TOKEN" http://127.0.0.1:8000/metrics
dc logs --tail=200 api worker
~~~

scripts/api_demo.py 只针对生成 PNG/tiny 回归：它硬编码 image/png，优先 tiny profile，不能用于混合格式真实 GPU 数据或 benchmark。真实 HTTP 客户端应保留源 MIME type、选择 local-sd15-v1、给每个 mutation 设置 Idempotency-Key，并对 dataset/job 分别设置 timeout。可先用 API_MAX_STEPS=10 做端到端 GPU 验收，正式运行删除该覆盖后使用 profile 默认 500 steps。

~~~bash
export API_MAX_STEPS=10
dc run --rm --no-deps -i -T -e API_MAX_STEPS -e LORA_DEMO_TOKEN="$DEMO_TOKEN" \
  -v "$DATASET_DIR:/input:ro" api python - <<'PY'
import hashlib,json,mimetypes,os,time,uuid
from pathlib import Path
import httpx
root=Path("/input"); token=os.environ["LORA_DEMO_TOKEN"]
caps=json.loads((root/"captions.json").read_text()) if (root/"captions.json").exists() else {}
items=[]
for p in sorted(root.iterdir()):
 if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}:
  raw=p.read_bytes(); mime=mimetypes.guess_type(p.name)[0]
  if mime not in {"image/jpeg","image/png","image/webp"}: raise RuntimeError(f"bad MIME: {p}")
  items.append((p,{"name":p.name,"size_bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"mime_type":mime,"caption":caps.get(p.name)}))
if not 100<=len(items)<=1000: raise RuntimeError(f"expected 100..1000 files, got {len(items)}")
h={"Authorization":f"Bearer {token}"}
def post(c,path,body):
 r=c.post(path,json=body,headers={**h,"Idempotency-Key":str(uuid.uuid4())});r.raise_for_status();return r.json()
with httpx.Client(base_url="http://api:8000",timeout=120) as c:
 ds=post(c,"/v1/datasets",{"files":[m for _,m in items]}); dsid=ds.get("dataset_id",ds["id"]); by={p.name:p for p,_ in items}
 for f in ds["files"]:
  p=by[f["name"]];fid=f.get("file_id",f["id"]);r=c.put(f"/v1/datasets/{dsid}/files/{fid}",content=p.read_bytes(),headers={**h,"Content-Type":mimetypes.guess_type(p.name)[0]});r.raise_for_status()
 post(c,f"/v1/datasets/{dsid}/complete",{})
 end=time.monotonic()+900
 while time.monotonic()<end:
  ds=c.get(f"/v1/datasets/{dsid}",headers=h).json()
  if ds["state"]=="COMPLETED":break
  if ds["state"]=="INVALID":raise RuntimeError(json.dumps(ds))
  time.sleep(1)
 else:raise TimeoutError("dataset verification")
 ps=c.get("/v1/training-profiles",headers=h).json()["profiles"]; p=next(x for x in ps if x["profile_revision_id"]=="local-sd15-v1")
 job=post(c,"/v1/training-jobs",{"dataset_id":dsid,"profile_revision_id":p["profile_revision_id"],"trigger_token":"mystyle","training_overrides":{"max_steps":int(os.environ.get("API_MAX_STEPS","500"))}});jid=job.get("job_id",job["id"])
 end=time.monotonic()+7500
 while time.monotonic()<end:
  job=c.get(f"/v1/training-jobs/{jid}",headers=h).json()
  if job["state"] in {"READY","COMPLETED_UNVERIFIED"}:break
  if job["state"] in {"FAILED","QUALITY_REJECTED","CANCELLED"}:raise RuntimeError(json.dumps(job))
  time.sleep(5)
 else:raise TimeoutError("training job")
 ev=c.get(f"/v1/training-jobs/{jid}/evaluation",headers=h);ev.raise_for_status()
 a=c.get(f"/v1/models/{job['model_id']}/download",headers=h,params={"allow_unverified":"true"});a.raise_for_status()
 out=Path("/data/api-gpu-result");out.mkdir(exist_ok=True)
 (out/"adapter.safetensors").write_bytes(a.content)
 (out/"job.json").write_text(json.dumps(job))
 (out/"evaluation.json").write_text(json.dumps(ev.json()))
 print(json.dumps({"dataset":dsid,"job":job,"evaluation":ev.json(),"output":str(out)},indent=2))
PY
dc cp api:/data/api-gpu-result ./api-gpu-result
dc cp api:/data/sd15-smoke-evaluation ./sd15-smoke-evaluation
dc cp api:/data/sd15-smoke-ab ./sd15-smoke-ab
~~~

**成功判据：** PUT 返回无 body 的 204，dataset 为 COMPLETED，job 是 READY 或默认 COMPLETED_UNVERIFIED，evaluation 可读，adapter 可在显式 allow_unverified=true 时下载。metrics 要 admin key；日志为 JSON 事件且不记录 token、caption、图片。跨 owner 返回 404，非法请求返回标准错误 JSON。

## Docker 测试与 CPU 可选路径

运行镜像不带 pytest/Ruff/Mypy。下列命令只读挂载源码，在 /tmp 建临时 venv；它会访问 PyPI，不能验证 GPU。

~~~bash
dc run --rm --no-deps -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$PWD:/workspace:ro" -w /workspace api sh -lc '
    python -m venv --system-site-packages /tmp/lora-dev &&
    /tmp/lora-dev/bin/pip install --no-cache-dir "pytest==8.4.2" "ruff==0.13.2" "mypy==1.18.2" "types-Pillow" "types-psutil" &&
    /tmp/lora-dev/bin/python -m pytest -p no:cacheprovider -q &&
    /tmp/lora-dev/bin/python -m ruff check --no-cache src tests scripts/api_demo.py &&
    /tmp/lora-dev/bin/python -m mypy --cache-dir=/tmp/mypy-cache
  '
~~~

无 GPU 时使用默认 docker compose（不使用 dc）和 .env.example 的 test backend/fake slot 执行 lora-pipeline cpu-smoke --output /data/cpu-smoke。它生成合成 PNG，在 step 2 保存、恢复至 step 4、重载 adapter，结果标记 test_only=true、quality_status=UNCALIBRATED；这只是回归检查。

## 维护与排障

dc down 保留 named volumes，随后 dc up -d 可恢复服务和制品；docker compose down -v 会删除数据库、checkpoint、adapter、模型缓存，先用 dc cp 导出需要内容。GPU 生命周期命令都使用 dc。

若 docker compose build 在拉取 python:3.12-slim-bookworm metadata 时超时，Dockerfile 尚未开始执行。检查 Docker daemon 的出网、代理和 DNS，确认 daemon 能访问 Docker Hub 后重试。GPU_UNAVAILABLE 时重跑 preflight，检查 driver、NVIDIA Container Toolkit、overlay、LORA_GPU_UUIDS。GPU_OUT_OF_MEMORY 时降低 resolution/rank/batch 或提高 accumulation，记录新配置后重新测量。

若错误出现在 Dockerfile 的 `pip install` 阶段，并包含 `files.pythonhosted.org`、`ReadTimeoutError` 或 `pip subprocess`，说明基础镜像已经成功，失败发生在 PyPI 依赖下载。镜像对 PyPI 使用独立于 `TORCH_INDEX_URL` 的 `PIP_INDEX_URL`，默认超时为 300 秒、重试 10 次，并以 BuildKit cache mount 保存已成功下载的 wheel/HTTP 缓存。因此同一个 builder 上再次执行 build 会复用已下载内容；不要在普通重试时加 `--no-cache`。BuildKit 是前提，先运行 `docker buildx version`（需 sudo 时用 `sudo docker buildx version`）确认其可用，再确认 Docker daemon 能访问 PyPI，然后用与权限匹配的命令重新构建：

```bash
# 当前用户能访问 Docker socket：
dc build --progress=plain

# 当前用户必须通过 sudo 使用 Docker：
sudo docker run --rm python:3.12-slim-bookworm \
  python -c 'import urllib.request; print(urllib.request.urlopen("https://pypi.org/simple/", timeout=30).status)'
sudo docker compose -f compose.yaml -f compose.gpu.yaml build --progress=plain
```

若默认 PyPI 在当前网络不稳定，可由用户选择一个可信镜像，并只改变应用依赖下载源；CPU/GPU PyTorch wheel 仍分别使用 compose 的 `TORCH_INDEX_URL`/`TORCH_INDEX_URL_GPU`。`PIP_INDEX_URL` 是 build arg，可能出现在构建元数据中，不能包含用户名、密码或 token；需要认证的镜像应使用 BuildKit secrets，超出下面命令的范围。例如：

```bash
# Replace the URL only with a mirror approved for this environment.
export PIP_INDEX_URL='https://your-approved-pypi-mirror/simple'
dc build --progress=plain

# sudo does not inherit the exported variable reliably, so pass it to root explicitly.
sudo env PIP_INDEX_URL='https://your-approved-pypi-mirror/simple' \
  docker compose -f compose.yaml -f compose.gpu.yaml build --progress=plain
```

如需调大等待时间，也可在上述任一命令前设置 `PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=15`；sudo 方式同样使用 `sudo env PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=15 ...`。只有确认缓存内容需要丢弃时才重置 BuildKit 的 cache mount；这会清除该 Docker builder 中所有项目的执行缓存，下一次构建将重新下载：

```bash
sudo docker builder prune --filter type=exec.cachemount --force
```

如果 PyPI 网络检查也超时，需要修复 Docker daemon 的代理/DNS，或改用获准的镜像；单纯重复启动 worker 不会修复下载失败。`pull access denied: local-lora-pipeline` 是同一问题的后果：镜像构建失败，所以本地没有 `local-lora-pipeline:0.1.0`，Compose 才尝试从远程仓库拉取这个本地标签。确认下面命令能看到镜像后再启动服务：

```bash
sudo docker image inspect local-lora-pipeline:0.1.0 >/dev/null
dc up -d api worker
```

## Benchmarks：三次真实 GPU 完整训练

10-step smoke 只验证模块。正式 benchmark 固定同一真实 training-input.json、已缓存 SD 1.5 revision、同一 GPU 和同一配置，先运行一次不计统计的 warm-up，再运行至少三次**完整**训练。总 wall time 包含模型加载；steady-state 从 optimizer step 11 开始，明确排除加载和前 10 个 warm-up steps。

下方命令在 /data/benchmarks 写每次 progress/result 以及机器可读 summary.json/summary.csv。它直接调用同一 training.train()，避免把 CLI 多行 JSON 当作 JSONL；每次 train() 仍通过 _device 获取同一个 UUID 锁。先停止模块 4 的 worker，避免与 benchmark 竞争 GPU；任一次失败会以非零退出，不会伪造汇总。100 steps 是可重复的测量 workload，不是质量阈值。


| 输出字段                           | 含义与来源                                                                                |
| ---------------------------------- | ----------------------------------------------------------------------------------------- |
| data_preparation_seconds           | 在模块 1 的 prepare 外层测得的墙钟时间；本次没有采集时填 null 和 not_measured，不能猜测。 |
| optimizer_step_seconds             | steady_seconds / steady_steps；只计算第 11 步以后。                                       |
| checkpoint_seconds_total           | 每次运行 progress 中 checkpoint_seconds 的和，包含完整 checkpoint 保存。                  |
| peak_gpu_memory_allocated/reserved | 每个训练 progress 的峰值 CUDA bytes。                                                     |
| wall_seconds_including_load        | 一次完整训练含模型加载的总墙钟时间，与 steady-state 分开报告。                            |

~~~bash
export BENCH_COMMIT="$(git rev-parse HEAD)"
export BENCH_ROOT="/data/benchmarks-$(date -u +%Y%m%dT%H%M%SZ)"
dc stop worker
dc run --rm --no-deps -i -T -e BENCH_COMMIT -e BENCH_ROOT -e DATA_PREPARATION_SECONDS -v "$PWD/configs:/workspace/configs:ro" worker python - <<'PY'
import csv,json,os,statistics,time
from pathlib import Path
from lora_pipeline.cli import _device,preflight
from lora_pipeline.common import atomic_write,canonical_json,load_manifest
from lora_pipeline.config import TrainConfig
root=Path(os.environ["BENCH_ROOT"])
if root.exists() and any(root.iterdir()): raise RuntimeError(f"benchmark root must be new or empty: {root}")
root.mkdir(parents=True,exist_ok=True)
raw=json.loads(Path("/workspace/configs/sd15-smoke.json").read_text());raw.update({"max_steps":100,"checkpoint_every":50,"local_files_only":True})
smoke=load_manifest(Path("/data/sd15-smoke-training/training-result.json"))
if smoke["base_model"] != raw["model_name"]: raise RuntimeError("benchmark base model differs from validated smoke")
raw["revision"]=smoke["base_revision"]
config=TrainConfig(**raw);inputs=load_manifest(Path("/data/real-input/training-input.json"))
def peak(progress,key):
 values=[x[key] for x in progress if key in x]
 return max(values) if values else None
def run(name):
 progress=[];start=time.monotonic()
 result=train(inputs,root/name/"training",config,progress=progress.append)
 wall=time.monotonic()-start
 atomic_write(root/name/"progress.json",canonical_json(progress))
 b,last=progress[9],progress[-1];seconds=last["elapsed_seconds"]-b["elapsed_seconds"];samples=last["samples_processed_this_run"]-b["samples_processed_this_run"]
 return {"run":name,"state":result["state"],"wall_seconds_including_load":wall,"train_elapsed_seconds":result["elapsed_seconds"],"warmup_steps_excluded":10,"steady_steps":last["global_step"]-b["global_step"],"steady_seconds":seconds,"optimizer_step_seconds":seconds/(last["global_step"]-b["global_step"]),"steady_samples":samples,"steady_samples_per_second":samples/seconds,"checkpoint_seconds_total":sum(x.get("checkpoint_seconds",0) for x in progress),"peak_gpu_memory_allocated":peak(progress,"gpu_memory_allocated"),"peak_gpu_memory_reserved":peak(progress,"gpu_memory_reserved"),"peak_cpu_rss_bytes":peak(progress,"cpu_rss_bytes"),"result_path":result["manifest_path"]}
with _device(config.device):
 from lora_pipeline.training import train
 selected_gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"]
 warmup=run("warmup");runs=[run(f"run-{i}") for i in range(1,4)]
prep=os.environ.get("DATA_PREPARATION_SECONDS")
summary={"measurement_status":"measured","data_preparation_seconds":float(prep) if prep else None,"data_preparation_status":"measured prepare wall time" if prep else "not_measured","commit":os.environ.get("BENCH_COMMIT"),"environment":preflight(),"input_manifest_sha256":inputs["manifest_sha256"],"model":{"name":config.model_name,"revision":config.revision},"config":config.snapshot(),"warmup":warmup,"runs":runs,"aggregate":{"sample_count":3,"steady_samples_per_second_mean":statistics.mean(x["steady_samples_per_second"] for x in runs),"steady_samples_per_second_stdev":statistics.stdev(x["steady_samples_per_second"] for x in runs),"wall_seconds_including_load_mean":statistics.mean(x["wall_seconds_including_load"] for x in runs)}}
summary["selected_gpu_uuid"]=selected_gpu_uuid
prepared=load_manifest(Path("/data/real-prepared/prepared.json"))
summary["accepted_unique_images"]=prepared["statistics"]["accepted_unique"]
summary["data_preparation_images_per_second"]=summary["accepted_unique_images"]/float(prep) if prep and float(prep)>0 else None
summary["evaluation_smoke"]=load_manifest(Path("/data/sd15-smoke-evaluation/evaluation.json"))
atomic_write(root/"summary.json",canonical_json(summary))
with (root/"summary.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(runs[0]));w.writeheader();w.writerows(runs)
print(json.dumps(summary["aggregate"],indent=2))
PY
dc cp "api:$BENCH_ROOT" "./$(basename "$BENCH_ROOT")"
dc start worker
~~~
提交 summary.json、summary.csv、三次 progress.json/training-result.json，并记录 cache 状态、OS/CPU/RAM、GPU UUID/显存、driver/CUDA、并发数、数据来源、失败/重试。summary.json 已有 commit、package/CUDA/GPU preflight、input hash、模型 revision 与配置。不要把 CPU tiny 时间、10-step smoke 或计划数值当作实测 benchmark。

数据准备计时精度为秒，包含一次性容器启动；这是整个 Docker prepare 调用的吞吐，不是纯算法吞吐。`evaluation_smoke` 保留模块 3 的技术检查、CLIP 诊断和评估总耗时，属于 10-step adapter 的验证结果，与三次 100-step 性能实验分别解释。评估总耗时包含加载、生成和 CLIP 计算，不能直接称为单图推理延迟。

当前交付的实测状态如下；在目标主机完成上述命令后，用导出的 summary.json/CSV 替换待测项。


| 结果                              | 当前状态                                 | 实测来源                                                               |
| --------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------- |
| GPU 训练吞吐、step 时间、峰值显存 | not_measured；sample_count=0；value=null | 三次 runs 与 aggregate                                                 |
| 数据准备吞吐、checkpoint 保存时间 | not_measured；value=null                 | data_preparation_images_per_second、各 run 的 checkpoint_seconds_total |
| 真实 SD 1.5 技术检查与 CLIP/A-B   | not_measured                             | evaluation.json、comparison.json 与配对图像                            |
| API P50/P95、2–4 GPU 并发扩展    | not_measured                             | 需另行运行多 owner 的受控并发负载，不能由本次串行实验推断              |

指标定义、并发实验与失败成本的报告方法见 [性能基准协议](docs/05-performance-benchmarks.md)。
