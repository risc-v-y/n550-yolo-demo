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

单图日志在 `yolo26_pc/outputs/board-pcie/<时间>/`，顺序视频在 `yolo26_pc/outputs/host-video/run-*/`（旧摄像头入口仍在 `outputs/live-demo/<时间>/`），部署日志在 `outputs/deploy-*/load.log`。`diagnostics.jsonl` 包括 PCIe 工具参数、退出码、输出尾部、状态变化和完成/失败时的板端快照。Python 异常保存 traceback。UART 文字不自动写入 JSONL，需使用终端会话日志另存。

`board_diag` 为 816 字节、64 字节对齐、版本 1 的固定结构，经缓存 clean 发布，可由 PCIe 或已有调试器读取；地址见 `firmware/pcie-symbols.txt`。保留旧 RX/RPC 计数字段以兼容布局，当前不再使用。主要信息为阶段、帧号、节点/形状、子阶段、错误/detail、AMU flags、异常 CSR 和 x0–x31。

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

## 自动数值定位

操作统一见 [README第3.1节](../README.md#31-自动采集并在-host-上比较)。`node_capture.py` 自动完成334次暂停/读回/确认，在host用完整数据逐张量比较，无需串口抓取或GDB断点。基准来自现有同模型、同416输入、同FP16方案的QEMU输出；分片、输入、图和常量均校验SHA-256。

| 小包文件 | 用途 |
|---|---|
| summary.json | 采集状态、首差异位置、最终结果差异、普通模式重复结果、耗时及最后诊断 |
| nodes.csv | 每个有效输出的全量逐位一致性、差异元素数、最大/平均绝对差、RMSE、最大ULP、输入CRC、周期及AMU flags |
| first_difference.json | 第一个差异节点的形状、实际/参考原始位、有限个差异样本、输入来源与全字节对照、可用的重算结果 |
| annotated.png | host按现有规则绘制的检测图；全模型查验另存full-check-annotated.png |

优先回传 `node_report.py` 的三页终端截图（每页最多22行、80列），同一报告的ID一致。第一页包含总体结果和普通模式重复检查；第二页包含首差异的形状、输入全字节对照、完整张量指标与少量原始位；第三页包含各候选规则对实际/参考结果的匹配数量。失败时第二、三页改为异常及最后板端状态，优先排查采集中断。数值未知显示unknown，不作为通过。

完整 `tensors/`、首差异实际输入/参考、完整nodes.json及工具日志保留host，`return.zip`仍可选，上限1 MiB。失败报告保留已有CSV、首差异和异常；取消失败另记 `capture_abort_error`，现场核心/PCIe恢复后可使用 `--recover-paused` 重启采集。查看旧目录可直接指定路径，不必重新运行板端。

检测图默认PNG；旧SVG保留，可用菜单13离线转换，不会重新推理或改变数值报告。PNG通过host的librsvg/Cairo生成，不要求图形会话；缺库在实板操作前报告。PNG保留在host，不放入可选的小包。

旧诊断在找到首差异后重算少量位置。当前菜单11仅对bus全部334节点、335个有效输出重放，使用对应节点的实板输入；先逐节点采集一次，再普通推理复跑一次，正常共两次模型推理。原始512字节快照保存在`snapshots/`，记录输入前CRC、实际输入文件及SHA-256、完整输出SHA-256。输入被输出覆盖时，从此前完整输出重建并核对原CRC；无法核实则标记`input_unverified`，不能通过验收。正式固件、冻结图和原始图片输入的哈希须匹配，旧版/缺失/损坏采集不能复用。

菜单12直接显示已有bus分析，不重放已完成分析、不把旧zidane失败计入bus结论。缺少分析但有完整原始采集时才离线补分析一次，另存bus-analysis.json；没有完整采集则明确报错，不启动推理。旧报告和数据保留。停止/更新脚本先判断进程，运行中则SIGINT结束，已结束则跳过，再显示已有报告；不启动新的查验。操作入口见README。

新查验程序覆盖21类算子。AMU普通卷积按实际FP16值每16项与前次FP32累加值精确求和、RNE舍入，偏置最后单独FP32相加；非AMU路径保持现有运算顺序。分类为`exact`、`split16_match`、`unexplained`、`input_unverified`、`missing`；本节点按实际输入重放一致、但与QEMU不同的非AMU输出另标`upstream_propagation`，避免把上游差异当作本节点错误。全部异常保存在bus的`analysis.json`，菜单12统一显示三页摘要。不运行或保留两图查验入口。

旧入口的少量重算仍只是定位线索；新全量重放匹配同样不能替代RTL等价证明，16项规则也不是有限窗口PE的完整实现。全部张量原始字节均参与比较，不用经验阈值判为通过。非有限值单独统计；稳定重复和某张图绘框相同也不能推断其他输入精度。

捕获会增加CRC扫描、缓存维护、PCIe传输和等待，因此最后默认再执行同输入的普通模式，报告它与捕获结果是否逐字节一致。该检查用于发现采集对结果的影响，采集周期不能用于优化性能结论。

## 自测定位

当前 `clp-selftest` 使用可精确表示的整数，连续32轮循环 `(M,N,K)=(3,5,17)/(1,1,1)/(2,3,32)/(3,5,31)`，逐位比较、不设误差阈值。`frame` 是轮次0–31，不是视频帧；轮次除以4的余数对应上述形状。UART 失败行也直接打印 M/N/K。

`tensor`（UART 的 `buffer`）是64字节块编号：0–7为A、8–15为B、16–23为AMU输出C、24–31为RVV输出D、32–63为额外保护区；`element` 是块内32位字下标。每个缓冲区首块为保护行，随后才是有效行。失败全局DDR地址为 `0xF0000000 + tensor×64 + element×4`，RVV别名为 `0x10000000 + tensor×64 + element×4`。

CLP失败 `error=4`，`detail=-200-tensor`，`observed/expected` 是原始32位字。A/B每字含低/高两个FP16，奇数K的末字高半字保护值为 `3555`；C/D行内前N字有效，其余为 `c49a4000`。`board_test_stage`：1全区写读检查，4依赖及复用，3通过；错误码0/4/3/0x100表示通过/数据不符/AMU初始化失败/trap。独立固件结束后停车于 `clp_probe_halt`；模型启动复用自测，通过后进入READY，失败保留诊断并发布PCIe ERROR。

旧 `selftest` 保留原尺寸及 `detail=-100/-101/-102/-103`，仅供历史缓存定位，未修正原交接问题。正式模型已经改用CLP自测；各固件必须使用自己的ELF，不能混用固定地址停点。

实板保持DCache开启，单图和连续帧验收见 [验收状态](VALIDATION.md)。CLP模型的数据为全局高地址，调试RVV时使用对应CLP别名；读取PCIe诊断仍使用低地址符号。

## UART 接口核对来源

已核对千问交接包固定提交 `21e7253b7caf7d34864fb82a807720f7eba00bbc` 的 [main_baremetal.c](https://github.com/trollsmash/n550_qwen3_bringup_handover/blob/21e7253b7caf7d34864fb82a807720f7eba00bbc/fpga_bringup/src/src/main_baremetal.c#L73)：`uart_init()` 初始化；`putc_raw()` 等待 LSR bit5 后写 THR；`putc_()` 处理换行；`P/U/I/X` 分别打印字符串、无符号数、有符号数和十六进制。

YOLO 使用独立 `board/uart.c` 的 `board_uart_init/put/text/u64/i64/hex`，保留动态 DLF 检测，超时后不写 THR 并关闭后续日志。仅核对平台 UART 接口，未参考或复用千问算子。
