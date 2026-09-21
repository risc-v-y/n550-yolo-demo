# 固定数值基准

bus/zidane图片位于 `yolo26_pc/samples/`，图片和原始PyTorch权重的哈希在 `assets.lock.json`。

`*-qemu-fp16.bin` 来自2026-09-17已通过的QEMU RVV+AMU FP16图片协议验证。格式为32字节Y26VDET v1头，随后300×6小端FP32候选框。历史run_id只作来源记录。

`*-detections.bin` 是对应QEMU结果经过既定后处理后的基准：16字节 `<8sII>` 头（Y26DETS1、版本1、检测数），随后N×6小端FP32。`check_board_reference.py` 检查当前PC后处理与它们逐字节一致。

板端运行同一图片时，为 `uart_backend.py` 增加 `--reference reference/bus-qemu-fp16.bin`（zidane使用对应文件）。报告记录原始候选框是否一致、类别顺序、坐标与分数最大差异，不设经验容差掩盖差异。基准来源是QEMU，不是已验收的板端运行。
