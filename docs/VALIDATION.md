# 交付与验收状态

本仓库提供两套独立的板端验证方案：PCIe 加载 `firmware/demo-pcie.bin`，UART 加载 `firmware/demo.elf`。`firmware/selftest.elf` 用于独立自测。PCIe 对应的 `firmware/demo-pcie.elf` 仅供符号查询和调试，不能作为 PCIe 加载文件。两套模型固件均包含模型图和权重，不需另行上传权重。

## 当前结论

| 检查范围 | 结果 |
|---|---|
| 交付文件 | `firmware/SHA256SUMS` 所列 9 项哈希核对通过；PCIe BIN 与对应 ELF 的可加载内容一致，CPU 入口为 `0x80000000`，PCIe 加载偏移为 `0x0` |
| 板端构建 | ESWIN GCC 14.1.1 构建通过，启用 `-Wall -Wextra -Werror`；32 MiB DDR 链接范围检查通过 |
| PCIe 软件链路 | 板端协议代码、模拟 DDR 与模拟 `pbcopy/pbload` 的 11 项回归通过；合成视频的两帧软件流程通过 |
| UART 软件链路 | 板端 C 协议与 Python 客户端的 7 项回归通过；完整帧传输、重试及诊断行为已在本机检查 |
| RVV/AMU 自测 | QEMU 下 VLEN 128/256/512 的正常与故障注入共 9 组回归通过 |
| 模型资产与参考 | 固定模型资产哈希检查通过；bus/zidane 的 PC 后处理与固定 QEMU FP16 检测基准逐字节一致 |

以上均为构建、本机或 QEMU 检查，**不构成真实 S2C 板卡的运行验收**。QEMU 不模拟 N550 非一致 DCache；PCIe 模拟工具和替身推理不能证明现场 PCIe 传输、模型数值或吞吐。

## 实板待验收

1. 按 [README](../README.md) 选择 PCIe 或 UART，核对固件哈希、DDR 独占范围、加载地址及启动方式，确认实际链路可用。
2. 在 DCache 开启时运行自测，验证 RVV→AMU→RVV 数据可见性、缓存维护和 AMU 完成等待语义。
3. 完成 bus/zidane 单图推理，对照 `reference/` 数值基准，检查类别、置信度、坐标和绘框结果。
4. 验证两帧及持续视频、真实 host 图形会话、异常诊断和各阶段耗时；实际吞吐以现场报告为准。

失败时保留输出目录中的 `report.json`、`diagnostics.jsonl` 及加载固件的哈希，并参照 [诊断说明](DIAGNOSTICS.md) 定位。软件流程与参数以 [根目录操作指南](../README.md) 为准。
