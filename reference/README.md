# 固定数值基准资产

本页只说明资产格式和用途；运行命令见[操作手册](../docs/OPERATIONS.md)，数值判据与验收范围见[VALIDATION](../docs/VALIDATION.md)。

bus/zidane原图位于 `yolo26_pc/samples/`，原图及PyTorch权重哈希在 `assets.lock.json`。

| 资产 | 格式与用途 |
|---|---|
| `*-qemu-fp16.bin` | 原QEMU RVV+AMU FP16候选：32字节Y26VDET v1头，加300×6个小端FP32；run_id仅标识来源 |
| `*-detections.bin` | 对应QEMU候选的既定后处理结果：16字节 `<8sII>` 头（Y26DETS1、版本1、检测数），加N×6个小端FP32 |
| `prepared/` | 固定图片的RGB CHW FP32输入，以及源图、输入、基准哈希和缩放补边参数 |
| `nodes/manifest.json` | 冻结节点及张量描述，供逐节点查验和当前审查表生成使用 |

`check_board_reference.py` 核对PC后处理与固定detections逐字节一致。`pcie_prepared.py` 使用prepared输入，在只有Python标准库的host完成单图收发、QEMU差异统计和PNG绘框；这些资产不是实时视频输入。

QEMU基准保留作历史数值对照，不代表实板验收结果。视频逐位判据使用 `firmware/host-verify` 的RTL规则；不能把已知QEMU差异混作RTL golden差异。当前单图验收使用BUS，zidane资产保留但不要求再次两图采集。
