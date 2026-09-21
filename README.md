# Automatic LoRA Training Pipeline — Assignment Submission

本目录交付 **Part 1: System Design** 要求的两项成果。正式文档和图中标注使用英文；本文提供中文阅读导航。方案面向风格 LoRA：输入 100–1,000 张图片，2–4 张 GPU 处理并发任务，模型通过质量门槛后交付给推理消费者。基础模型与 GPU 型号通过训练配置抽象，不固定某一厂商或型号。

## 两个交付物

| 交付物 | 内容 | 文件 |
|---|---|---|
| 1. System Architecture Diagram | 从上传到模型交付的数据流、组件交互、共享 GPU 池、扩展与故障恢复；含总体架构图及两张辅助图 | [架构文档](docs/01-system-architecture.md) |
| 2. Technical Specification | 组件职责、调度、训练和质量策略、API 契约、数据模型、错误处理、安全与运维、验收场景 | [技术规格](docs/02-technical-specification.md) |

这是架构与技术设计交付。文中的时限、配额与策略默认值是设计选择；硬件性能需要按训练配置实测，质量阈值需要离线校准。未实现训练服务，也未宣称完成真实模型训练或部署。

## 图表预览与 PlantUML 代码

![System architecture](diagrams/system-architecture.png)

| 图表 | 可编辑源代码 | 矢量图（适合放大与文档排版） | 图片 |
|---|---|---|---|
| 总体架构、组件与数据流 | [system-architecture.puml](diagrams/system-architecture.puml) | [SVG](diagrams/system-architecture.svg) | [PNG](diagrams/system-architecture.png) |
| 任务阶段、质量门槛与有限重训 | [job-lifecycle.puml](diagrams/job-lifecycle.puml) | [SVG](diagrams/job-lifecycle.svg) | [PNG](diagrams/job-lifecycle.png) |
| 租约失效、设备隔离与旧结果拦截 | [lease-recovery.puml](diagrams/lease-recovery.puml) | [SVG](diagrams/lease-recovery.svg) | [PNG](diagrams/lease-recovery.png) |

`.puml` 文件包含完整的 PlantUML 代码，SVG/PNG 是从这些源文件实际渲染得到的。使用纯本地渲染；图表源码不需要提交到公共在线渲染服务。

## 与 Evaluation Criteria 的对应关系

| 评分维度 | 本方案的主要证据 |
|---|---|
| Understanding of ML pipeline architecture | 数据版本、分组去重与验证集隔离、内容标注、LoRA 训练、完整检查点、多维质量评估、保存后重新加载验证 |
| Knowledge of distributed systems | 事务性入队、任务租约与 fencing token、幂等 API、设备隔离、失败恢复、取消与发布竞争、单次生效的发布 |
| Consideration of production requirements | 多租户权限、上传限制、可追溯数据模型、任务配额、错误码、监控审计、备份和保留策略 |
| Creative problem-solving for resource constraints | 训练与评估共享 GPU、按用户公平且利用空闲容量的调度、有限重训、兼容缓存、提前拒绝无效数据、统一 GPU 时间预算 |

架构文档第 6 节提供英文评分映射；技术规格最后一节给出可验证场景。

## 复现渲染

环境：Bash、Java 21、`sha256sum`，以及固定版本的 PlantUML 1.2026.8 JAR。架构与流程图使用内置 Smetana 布局，无需安装 Graphviz；恢复图使用 PlantUML 的时序图布局。参见 [PlantUML Smetana 文档](https://plantuml.com/smetana02)及[命令行文档](https://plantuml.com/command-line)。

先从 Maven Central 下载官方发布的渲染器到临时目录：

```bash
curl --fail --location \
  https://repo.maven.apache.org/maven2/net/sourceforge/plantuml/plantuml/1.2026.8/plantuml-1.2026.8.jar \
  --output /tmp/creaition-plantuml-1.2026.8.jar
```

在项目根目录运行：

```bash
JAVA_BIN=/usr/lib/jvm/java-21-openjdk-amd64/bin/java \
  bash scripts/render-diagrams.sh /tmp/creaition-plantuml-1.2026.8.jar
```

如果 `java` 已指向 Java 21，可省略 `JAVA_BIN`；其他系统按实际 Java 路径设置。脚本先校验渲染器 SHA-256 和全部图表语法，再生成三组 SVG/PNG。脚本只会更新本目录内的六个渲染图片文件。

固定渲染器 SHA-256：

```text
0f77e5f769836b3dee340e207fe497c3e4c43e973d559e3c306915da9c32e34c
```

需要了解各项设计取舍时，从架构文档顺序阅读；需要核对接口、状态和字段时，直接阅读技术规格。引用的官方文档和研究资料放在相关论述附近，便于逐项验证。

## 交付检查

- 独立设计评审已完成，发现的接口、并发、数据版本与故障恢复问题已修正并复核。
- 三份 PlantUML 源码通过语法检查，已实际生成三张 SVG 和三张 PNG，并检查渲染结果。
- 六段 JSON 示例通过解析，示例 UUID/SHA-256 格式有效；文档中的 23 个本地链接均可解析。
- SVG 通过 XML 解析，渲染脚本通过 Bash 语法检查并实际执行成功。

这些是设计文档与交付资产的检查；技术规格中的验收场景是将来实现系统时的验证要求，不表示已运行真实训练或故障注入测试。
