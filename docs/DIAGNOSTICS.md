# 运行状态与故障诊断

模型数据链路只使用 PCIe；UART 是独立文字日志输出，可用 minicom/PuTTY 与推理同时查看。操作步骤统一见 [README](../README.md)。

## 文字日志

- 默认：启动配置、AMU 初始化、自测、READY、每帧 RUN/DONE、每 32 个节点一次进度、PCIe 状态发布和错误。
- `--board-trace`：每个节点开始/结束、输入输出编号/类型/形状和 `node_cycles`。参数通过 PCIe 配置，文字从 UART 输出。
- `--check-intermediates`：额外扫描中间 FP32 输出的 NaN/Inf；默认已检查输入和最终输出。这不等于完整数值精度验证。

节点编号从 0 开始，算子编号与 `scalar_model.h` 的 ModelOp 对应。示例：

```text
[YOLO] BOOT transport=PCIe debug=UART baud=114942 dlf_bits=4 nodes=334 input_bytes=2076672 output_bytes=7200
[YOLO] AMU_INIT frame=0
[YOLO] SELFTEST frame=0
[YOLO] READY frame=0
[YOLO] RUN frame=1 node=none op=18446744073709551615 phase=1
[YOLO] RUN frame=1 node=32 op=0 phase=2
[YOLO] DONE frame=1 node=333 op=<实际算子编号> phase=10 frame_cycles=...
```

示例末节点算子以实际图表为准。模型 RUN→DONE 的 `frame_cycles` 包含整帧执行、缓存维护和日志开销；独立自测的 DONE 计数不表示模型帧耗时。`node_cycles` 从节点开始日志发完后计至结束日志前，含维护及可选有限值检查。详细打印增加耗时，不能据此直接比较优化性能。

UART 等待有上限，初始化验证失败或发送超时后禁用后续日志，PCIe 继续工作。`board_uart_status=0` 表示可用；`-1` 空闲超时、`-2` DLF 掩码、`-3/-4/-5` 寄存器验证失败、`-6` 尚未初始化、`-7` 发送超时。`board_diag.tx_errors` 记录故障次数。UART MMIO 本身总线异常仍会触发 trap；嵌套异常可能直接停车，不能保证打印。

## host 日志与 DDR 快照

单图日志在 `yolo26_pc/outputs/board-pcie/<时间>/`，视频在 `outputs/live-demo/<时间>/`，部署日志在 `outputs/deploy-*/load.log`。`diagnostics.jsonl` 包括 PCIe 工具参数、退出码、输出尾部、状态变化和完成/失败时的板端快照。Python 异常保存 traceback。UART 文字不自动写入 JSONL，需使用终端会话日志另存。

`board_diag` 为 816 字节、64 字节对齐、版本 1 的固定结构，经缓存 clean 发布，可由 PCIe 或已有调试器读取；地址见 `firmware/pcie-symbols.txt`。保留旧 RX/RPC 计数字段以兼容布局，当前不再使用。主要信息为阶段、帧号、节点/形状、子阶段、错误/detail、AMU flags、异常 CSR 和 x0–x31。

CPU trap 使用独立 4 KiB 异常栈，保存整数寄存器及 `mcause/mepc/mtval`，尽力打印并发布 PCIe ERROR，然后停车；不自动恢复，不保存浮点/向量/矩阵寄存器。DDR、缓存指令或异常栈也故障时，记录可能不完整。

| 错误 | 含义 |
|---|---|
| 1/2 | 权重长度/图接口不符 |
| 3/4 | AMU 初始化/自测失败，detail 为原返回值 |
| 0x30 | 输入 NaN/Inf，查看 tensor/element/observed |
| 0x31 | 算子失败，查看 node/phase/detail |
| 0x32 | 输出 NaN/Inf，查看张量、下标和原始位 |
| 0x40–0x43 | PCIe 请求、会话、输入 CRC 或模型失败，见 PCIE.md |
| 0x100 | CPU trap，查看 trap_* 和 registers |

AMU 等待前记录阶段，返回后记录原始 `xmfflags`，不将所有非零 flags 判为错误。指令或总线访问无法返回时，C 无法在指令内部实现超时，由 host 超时及同事调试器定位 PC/反汇编。

## 自测定位

当前 `clp-selftest` 使用可精确表示的整数，连续32轮循环 `(M,N,K)=(3,5,17)/(1,1,1)/(2,3,32)/(3,5,31)`，逐位比较、不设误差阈值。`frame` 是轮次0–31，不是视频帧；轮次除以4的余数对应上述形状。UART 失败行也直接打印 M/N/K。

`tensor`（UART 的 `buffer`）是64字节块编号：0–7为A、8–15为B、16–23为AMU输出C、24–31为RVV输出D、32–63为额外保护区；`element` 是块内32位字下标。每个缓冲区首块为保护行，随后才是有效行。失败全局DDR地址为 `0xF0000000 + tensor×64 + element×4`，RVV别名为 `0x10000000 + tensor×64 + element×4`。

CLP 失败 `error=4`，`detail=-200-tensor`，`observed/expected` 是原始32位字。A/B 每字包含低/高两个FP16；K为奇数时末字高半字为保护值 `3555`，其他未使用半字也必须保留。C/D 行内前N字有效，其余为 `c49a4000`。`board_test_stage`：1全区写读检查，4依赖及复用，3全部通过；`board_error=0/4/3/0x100` 分别表示通过/数据不符/AMU初始化失败/trap。成功/失败均停车于 `clp_probe_halt`，只需一个最终临时硬件断点，见README。

旧 `selftest` 保留原尺寸和错误码：32 轮 `(3,5,17)/(1,1,1)/(2,3,32)/(3,5,31)`，`detail=-100/-101/-102/-103` 对应 A/B/AMU/RVV 输出；stage 1 RVV 缓存、2 AMU、4 复用、3 通过。它和模型启动自测尚未修正缓存 RVV 的错误交接，不能与新 CLP 固件混用符号或停点。

实板保持 DCache 开启；CLP 最小测试通过后还需模型迁移、单图及连续帧验收，见 [验收状态](VALIDATION.md)。

## UART 接口核对来源

已核对千问交接包固定提交 `21e7253b7caf7d34864fb82a807720f7eba00bbc` 的 [main_baremetal.c](https://github.com/trollsmash/n550_qwen3_bringup_handover/blob/21e7253b7caf7d34864fb82a807720f7eba00bbc/fpga_bringup/src/src/main_baremetal.c#L73)：`uart_init()` 初始化；`putc_raw()` 等待 LSR bit5 后写 THR；`putc_()` 处理换行；`P/U/I/X` 分别打印字符串、无符号数、有符号数和十六进制。

YOLO 使用独立 `board/uart.c` 的 `board_uart_init/put/text/u64/i64/hex`，保留动态 DLF 检测，超时后不写 THR 并关闭后续日志。仅核对平台 UART 接口，未参考或复用千问算子。
