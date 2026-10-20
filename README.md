# N550 YOLO26 板端演示

**Linux host预处理 → PCIe写DDR → N550 RVV+AMU推理 → PCIe读结果 → host绘框、显示和录像。** UART只输出调试文字；开发机远程查看host桌面。

固定YOLO26n、416×416、完整334节点，默认score>0.25、无NMS。软件同事从本页进入；详细命令集中在[操作手册](docs/OPERATIONS.md)。

## 开始使用

在物理连接S2C的Linux host仓库根目录执行：

```bash
# 打开统一菜单：1更新 → 15运行三帧 → 17查看数值报告 → 18查看性能
bash tools/host.sh
```

首次使用先按[环境准备](docs/OPERATIONS.md#1-准备与更新)安装离线视频依赖并确认图形会话。已有后台任务时先等待结束，菜单1会拒绝更新。

当前v5将8个3×3深度卷积改为直接RVV计算，开发回归已通过，实板收益待测。运行同三帧确认 `VIDEO CHECK PASSED`，性能页应显示 `rvv_window_fp16_reuse_dw3x3`、input_im2col=0；保留v4运行 `run-sup5jvd9` 供自动比较。v4实测见[性能记录](docs/TUNING.md#v4现场结果run-sup5jvd9)，其同次数值结果仍以已有视频报告为准。历史证据以[验收记录](docs/VALIDATION.md)为准，实测数字只在[性能评估](docs/TUNING.md)维护。

## 按问题选择文档

| 要解决的问题 | 唯一维护位置 |
|---|---|
| 如何更新、运行、看结果、导出日志和清理 | [OPERATIONS：操作手册](docs/OPERATIONS.md) |
| 模型各模块做什么、为什么这样计算 | [MODEL：模型功能](docs/MODEL.md) |
| 算子怎样实现、数据怎样流动、接下来可优化什么 | [IMPLEMENTATION：实现总览](docs/IMPLEMENTATION.md) |
| 某个节点的形状、公式、布局及源码在哪里 | [334节点审查表](docs/operator-audit.csv)、[来源哈希](docs/operator-audit.json) |
| 板端地址、硬件前提和缓存规则是什么 | [BOARD：平台契约](docs/BOARD.md) |
| 一帧怎样提交、完成、校验和恢复 | [PCIE：协议](docs/PCIE.md) |
| 错误字段、UART日志和自测怎样解释 | [DIAGNOSTICS：排障](docs/DIAGNOSTICS.md)、[串口识别](docs/SERIAL_PROBE.md) |
| 哪些输入和版本已验证、数值为何不同 | [VALIDATION：验收证据](docs/VALIDATION.md) |
| 怎样比较周期、帧时间与检测效果 | [TUNING：性能和精度评估](docs/TUNING.md) |

维护原则：操作步骤只改OPERATIONS，内核变化更新IMPLEMENTATION与节点表，验证结果更新VALIDATION，实测和计时口径更新TUNING；其他文档引用对应位置。节点表是生成产物，不手改CSV。

## 交付与开发资料

交付以当前main分支及 `firmware/` 为准，BIN用于加载，配套ELF用于符号调试；旧UART模型版本已停用。[构建与发布核对](docs/OPERATIONS.md#6-构建与发布核对)不会自动替换交付固件。

本地的移植计划记录目标和后续步骤，决策文档记录授权；`docs/HANDOFF.md` 只保存私有环境上下文，不随Git发布。`yolo26_pc` 的PyTorch参考及旧QEMU实验属于开发资料，不能代替当前板端说明；未跟踪的环境、工具链、手册和原始采集不能整目录提交或删除。
