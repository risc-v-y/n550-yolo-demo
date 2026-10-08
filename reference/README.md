# 固定数值基准

bus/zidane图片位于 `yolo26_pc/samples/`，图片和原始PyTorch权重的哈希在 `assets.lock.json`。

`*-qemu-fp16.bin` 是QEMU RVV+AMU FP16运行所得的固定候选框基准。格式为32字节Y26VDET v1头，随后300×6小端FP32候选框；文件内的run_id仅标识基准来源。

`*-detections.bin` 是对应QEMU结果经过既定后处理后的基准：16字节 `<8sII>` 头（Y26DETS1、版本1、检测数），随后N×6小端FP32。`check_board_reference.py` 检查当前PC后处理与它们逐字节一致。

板端运行同一图片时，在 `pcie_backend.py` 或 `uart_backend.py` 的单图命令中加 `--reference reference/bus-qemu-fp16.bin`（zidane使用对应文件）。报告记录原始候选框是否一致、类别顺序、坐标与分数最大差异，不设经验容差掩盖差异。基准来源是QEMU，不代表实板已验收。

`prepared/` 提供 bus/zidane 两张图片按正式 `prepare_input` 流程生成的 RGB CHW `[3,416,416]` 小端 FP32 输入文件，以及源图、输入和基准的 SHA-256 与缩放补边参数清单。软件同事可用 `python3 yolo26_pc/pcie_prepared.py --sample bus`（zidane同理）在 PCIe host 上免安装运行；脚本会核对清单哈希、读回候选框、与 QEMU 基准对照，并生成内嵌原图的 `annotated.svg`。输入文件用于固定图片验收，不是实时摄像头输入，也不代表实板结果。
