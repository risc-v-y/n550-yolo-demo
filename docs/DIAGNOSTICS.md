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

示例末节点算子以实际图表为准。新固件的模型 `frame_cycles` 在计算完成后冻结，包含输入有限值检查、图执行、缓存维护和期间日志，排除DONE打印和PCIe外层CRC/发布；旧固件是在DONE打印期间读取计数。独立自测的DONE计数不表示模型帧耗时。`node_cycles` 从节点开始日志发完后计至结束日志前，含维护及可选有限值检查。详细打印增加耗时，不能据此直接比较优化性能。

UART 等待有上限，初始化验证失败或发送超时后禁用后续日志，PCIe 继续工作。`board_uart_status=0` 表示可用；`-1` 空闲超时、`-2` DLF 掩码、`-3/-4/-5` 寄存器验证失败、`-6` 尚未初始化、`-7` 发送超时。`board_diag.tx_errors` 记录故障次数。UART MMIO 本身总线异常仍会触发 trap；嵌套异常可能直接停车，不能保证打印。

## host 日志与 DDR 快照

单图日志在 `yolo26_pc/outputs/board-pcie/<时间>/`，顺序视频在 `yolo26_pc/outputs/host-video/run-*/`（旧摄像头入口仍在 `outputs/live-demo/<时间>/`），部署日志在 `outputs/deploy-*/load.log`。`diagnostics.jsonl` 包括 PCIe 工具参数、退出码、输出尾部、状态变化和完成/失败时的板端快照。Python 异常保存 traceback。UART 文字不自动写入 JSONL，需使用终端会话日志另存。

`board_diag` 的前816字节维持版本1布局；新固件追加两个uint64字段（计时表地址、长度），共832字节，64字节对齐。PCIe信息区通告实际长度，新Host兼容816/832字节。旧RX/RPC计数字段保留。诊断快照经缓存clean发布，地址见 `firmware/pcie-symbols.txt`。

`board_profile` 是独立8152字节计时表，包含16个uint64头字段、334组cache/compute/check周期及末尾uint64承载的CRC32。PCIe DONE之前完成CRC和缓存clean，绑定会话/帧号；Host校验后保存到逐帧JSON的 `pcie_timing.board_profile`。每帧结束后读取一次，正常推理不逐节点停顿。计时边界与查看方式见 [TUNING](TUNING.md)。

CPU trap 使用独立 4 KiB 异常栈，保存整数寄存器及 `mcause/mepc/mtval`，尽力打印并发布 PCIe ERROR，然后停车；不自动恢复，不保存浮点/向量/矩阵寄存器。DDR、缓存指令或异常栈也故障时，记录可能不完整。

| 错误 | 含义 |
|---|---|
| 1/2 | 权重长度/图接口不符 |
| 3/4 | AMU 初始化/自测失败，detail 为原返回值 |
| 0x30 | 输入 NaN/Inf，查看 tensor/element/observed |
| 0x31 | 算子失败，查看 node/phase/detail |
| 0x32 | 输出 NaN/Inf，查看张量、下标和原始位 |
| 0x40–0x44 | PCIe 请求、会话、输入 CRC、模型失败或采集取消，见 PCIE.md |
| 0x100 | CPU trap，查看 trap_* 和 registers |

对 `0x30/0x32`，`observed` 是检测到的FP32原始位；`expected=0` 是未设置的诊断字段，不代表模型期望值为零。

### 已修正：节点305常量视图检查误报

旧版 `5af2816` 的中间值扫描始终从 `arena+offset` 读取。节点305将常量515的 `[2,3549]` 视为常量516的 `[1,2,3549]`，两者共用权重区地址；检查却读到arena以外的DDR，可能以 `tensor=516 element=0 got=0xffffffff` 报 `0x32`，PCIe再发布 `0x43`。这段日志不能证明COPY计算出错。

当前扫描按 `constant` 标志选择权重区或arena，并跳过未保留的输出槽位。QEMU回归把旧错误地址填成NaN，防止统一RAM默认零值掩盖越界读取；常量、模型图和算子计算保持原值。更新固件后重新进行实板图片验证。

AMU 等待前记录阶段，返回后记录原始 `xmfflags`，不将所有非零 flags 判为错误。指令或总线访问无法返回时，C 无法在指令内部实现超时，由 host 超时及同事调试器定位 PC/反汇编。

## 自动数值查验

菜单11仅在需要回归时采集BUS全部334节点、335个有效输出，再普通模式复跑一次；完整且校验通过的旧证据优先复用。采集以512字节快照记录输入前CRC、实际输入文件、完整输出SHA256，使用各节点实板实际输入离线重放，区分本节点差异与上游传播。

静态查验程序按当前RTL有限窗口和规格化行为重放AMU，原QEMU对照单独保留。分类为 `exact`、`rtl_match`、`unexplained`、`input_unverified`、`missing`。所有输出逐位比较，输入缺失或损坏不能通过，不设置经验误差阈值。

菜单12读取三页结果；旧参考报告自动离线重分析一次，另存 `bus-rtl-analysis.json`，原报告及原始数据保留。缺完整数据时明确报错，不自动推理。菜单6查看后台任务日志，普通SSH断线后任务继续。当前BUS的差异原因和结论见 [验收状态](VALIDATION.md)。

`summary.json`、`nodes.csv`、`first_difference.json`、`snapshots/`、`tensors/`及工具日志保留host；通用首差异报告仍可帮助排障，不再使用独立首层分析或三分块固件。需要导出时使用 `tools/export_bus_evidence.py`，不依赖历史分块数据。

检测图默认PNG；菜单13可离线转换旧SVG。PNG生成使用librsvg/Cairo，不要求DISPLAY；查看窗口需要host图形会话。采集增加CRC、读回和等待开销，不能用采集周期评估性能。

## 自测定位

当前 `clp-selftest` 使用可精确表示的整数，连续32轮循环 `(M,N,K)=(3,5,17)/(1,1,1)/(2,3,32)/(3,5,31)`，逐位比较、不设误差阈值。`frame` 是轮次0–31，不是视频帧；轮次除以4的余数对应上述形状。UART 失败行也直接打印 M/N/K。

`tensor`（UART 的 `buffer`）是64字节块编号：0–7为A、8–15为B、16–23为AMU输出C、24–31为RVV输出D、32–63为额外保护区；`element` 是块内32位字下标。每个缓冲区首块为保护行，随后才是有效行。失败全局DDR地址为 `0xF0000000 + tensor×64 + element×4`，RVV别名为 `0x10000000 + tensor×64 + element×4`。

CLP失败 `error=4`，`detail=-200-tensor`，`observed/expected` 是原始32位字。A/B每字含低/高两个FP16，奇数K的末字高半字保护值为 `3555`；C/D行内前N字有效，其余为 `c49a4000`。`board_test_stage`：1全区写读检查，4依赖及复用，3通过；错误码0/4/3/0x100表示通过/数据不符/AMU初始化失败/trap。独立固件结束后停车于 `clp_probe_halt`；模型启动复用自测，通过后进入READY，失败保留诊断并发布PCIe ERROR。

历史缓存selftest及固定地址GDB脚本已移除。独立CLP自测仍保留；使用各固件对应的ELF确认符号，不能混用地址。

实板保持DCache开启，单图和连续帧验收见 [验收状态](VALIDATION.md)。CLP模型的数据为全局高地址，调试RVV时使用对应CLP别名；读取PCIe诊断仍使用低地址符号。

## UART 接口核对来源

已核对千问交接包固定提交 `21e7253b7caf7d34864fb82a807720f7eba00bbc` 的 [main_baremetal.c](https://github.com/trollsmash/n550_qwen3_bringup_handover/blob/21e7253b7caf7d34864fb82a807720f7eba00bbc/fpga_bringup/src/src/main_baremetal.c#L73)：`uart_init()` 初始化；`putc_raw()` 等待 LSR bit5 后写 THR；`putc_()` 处理换行；`P/U/I/X` 分别打印字符串、无符号数、有符号数和十六进制。

YOLO 使用独立 `board/uart.c` 的 `board_uart_init/put/text/u64/i64/hex`，保留动态 DLF 检测，超时后不写 THR 并关闭后续日志。仅核对平台 UART 接口，未参考或复用千问算子。
