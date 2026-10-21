# N550 YOLO26 板端演示

**Linux host预处理 → PCIe写DDR → N550 RVV+AMU推理 → PCIe读结果 → host绘框、显示和录像。** UART只输出调试文字；开发机远程查看host桌面。

固定YOLO26n、416×416、完整334节点，默认score>0.25、无NMS。软件同事从本页进入；详细命令集中在[操作手册](docs/OPERATIONS.md)。

## 开始使用

在物理连接S2C的Linux host仓库根目录执行：

```bash
# 打开统一菜单：1更新 → 3运行三帧 → 5查看数值报告 → 6查看性能
bash tools/host.sh
```

首次使用先按[环境准备](docs/OPERATIONS.md#1-准备与更新)安装离线视频依赖并确认图形会话。已有后台任务时先等待结束，菜单1会拒绝更新。

当前模型使用**直接RVV窗口打包与整层bias**，计时版号8、路径 `rvv_direct_layer_bias`。已移除三行FP16缓存，保留原FP16 RNE及AMU运算顺序；94层打包与QEMU全334节点/335个输出逐位回归通过，实板提速待测。host选择 **1更新 → 8排障 → 6原输入普通复跑**：先核对保存输入和空闲板状态，再加载当前固件一次，用同一输入完成golden及周期检查，无需重采探针或提供UART信息。最近一次实板基线仍是旧版`run-1vn8c4yu`的11.925217秒。原理与实测见[完整手册](docs/MODEL.md#optimization-principles)，证据见[验收记录](docs/VALIDATION.md)。

只保留当前方案和最新优化回归，不维护中间优化结果表；分析/文档修改先留本地，随模型代码集中推送。

主菜单连续编号1–9，排障集中在8的子菜单；完整功能表见[操作手册](docs/OPERATIONS.md#1-准备与更新)。原命名命令保持可用，例如 `bash tools/host.sh performance`。

## 按问题选择文档

| 要解决的问题 | 唯一维护位置 |
|---|---|
| 如何更新、运行、看结果、导出日志和清理 | [OPERATIONS：操作手册](docs/OPERATIONS.md) |
| 优化模型：整体结构、每步计算/形状、当前实现、数据通路、热点及约束 | **[MODEL：完整计算与优化手册](docs/MODEL.md)** |
| 核对节点资料与冻结模型、源码是否一致 | [机器审查表](docs/operator-audit.csv)、[来源哈希](docs/operator-audit.json) |
| 板端地址、硬件前提和缓存规则是什么 | [BOARD：平台契约](docs/BOARD.md) |
| 一帧怎样提交、完成、校验和恢复 | [PCIE：协议](docs/PCIE.md) |
| 错误字段、UART日志和自测怎样解释 | [DIAGNOSTICS：排障](docs/DIAGNOSTICS.md)、[串口识别](docs/SERIAL_PROBE.md) |
| 哪些输入和版本已验证、数值为何不同 | [VALIDATION：验收证据](docs/VALIDATION.md) |
| 怎样比较周期、帧时间与检测效果 | [完整手册：性能与检测效果](docs/MODEL.md#performance) |

维护原则：模型计算、实现、优化原理和实测统一更新MODEL；节点表由脚本生成，验证证据更新VALIDATION，操作步骤更新OPERATIONS。

## 交付与开发资料

交付以当前main分支及 `firmware/` 为准，BIN用于加载，配套ELF用于符号调试；旧UART模型版本已停用。[构建与发布核对](docs/OPERATIONS.md#6-构建与发布核对)不会自动替换交付固件。

本地的移植计划统一记录目标、责任、已确认决定和后续步骤；`docs/HANDOFF.md` 只保存私有环境上下文，不随Git发布。`yolo26_pc` 的PyTorch参考及旧QEMU复现命令属于开发资料，不能代替当前板端说明；未跟踪的环境、工具链、手册和原始采集不能整目录提交或删除。
