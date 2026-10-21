# 运行状态与故障诊断

模型数据链路只使用 PCIe；UART 是独立文字日志输出，可用 minicom/PuTTY 与推理同时查看。操作命令统一见[OPERATIONS](OPERATIONS.md)。本页只解释日志、错误、自测和排障字段。

## 文字日志

- 默认：启动配置、AMU 初始化、自测、权重准备、READY、每帧 RUN/DONE、每32个节点一次进度、PCIe状态发布和错误。
- `--board-trace`：每个节点开始/结束、输入输出编号/类型/形状和 `node_cycles`。参数通过 PCIe 配置，文字从 UART 输出。
- `--check-intermediates`：额外扫描中间 FP32 输出的 NaN/Inf；默认已检查输入和最终输出。这不等于完整数值精度验证。

节点编号从 0 开始，算子编号与 `scalar_model.h` 的 ModelOp 对应。示例：

```text
[YOLO] BOOT transport=PCIe debug=UART baud=114942 dlf_bits=4 nodes=334 input_bytes=2076672 output_bytes=7200
[YOLO] AMU_INIT frame=0
[YOLO] SELFTEST frame=0
[YOLO] WEIGHTS prepared layers=94 bytes=5033984 values=2391664 cycles=...
[YOLO] READY frame=0
[YOLO] RUN frame=1 node=none op=18446744073709551615 phase=1
[YOLO] RUN frame=1 node=32 op=0 phase=2
[YOLO] DONE frame=1 node=333 op=<实际算子编号> phase=10 frame_cycles=...
```

示例末节点算子以实际图表为准。新固件的模型 `frame_cycles` 在计算完成后冻结，包含输入有限值检查、图执行、缓存维护和期间日志，排除DONE打印和PCIe外层CRC/发布；旧固件是在DONE打印期间读取计数。独立自测的DONE计数不表示模型帧耗时。`node_cycles` 从节点开始日志发完后计至结束日志前，含维护及可选有限值检查。详细打印增加耗时，不能据此直接比较优化性能。

UART 等待有上限，初始化验证失败或发送超时后禁用后续日志，PCIe 继续工作。`board_uart_status=0` 表示可用；`-1` 空闲超时、`-2` DLF 掩码、`-3/-4/-5` 寄存器验证失败、`-6` 尚未初始化、`-7` 发送超时。`board_diag.tx_errors` 记录故障次数。UART MMIO 本身总线异常仍会触发 trap；嵌套异常可能直接停车，不能保证打印。

## 首个输入块自动诊断

排障菜单5／`bash tools/host.sh pack-probe`只定位当前首层输入打包异常，不改变FP16数值、AMU累加顺序或整层bias规则。它先验证当前PCIe状态、首层诊断及RUN请求；若仍在node0/phase5，读回原输入、核对请求CRC并保存，再加载当前固件复现。空闲板使用冻结BUS；忙于其他阶段或原输入无法核实时停止，不复位覆盖现场。

专用运行使用`DIAG_TRACE`：仅node0、第一个空间块记录细分标记；普通视频不启用探针。UART形如：

```text
[PACK] row.check col=25 k=0 row=1 vl=0 seq=...
[PACK] row.hit col=25 k=0 row=1 vl=0 seq=...
[PACK] gather.begin col=25 k=0 row=-1 vl=27 seq=...
[PACK] gather.done col=25 k=0 row=-1 vl=27 seq=...
[PACK] store.begin col=25 k=0 row=-1 vl=27 seq=...
[PACK] store.done col=25 k=0 row=-1 vl=27 seq=...
[PACK] column.done col=25 k=27 row=-1 vl=0 seq=...
[PACK] block.done col=32 k=27 row=-1 vl=0 seq=...
```

标记覆盖scratch准备、行检查/填充/命中、索引、gather和写入。`col`为本块空间列号，`k`为K项位置，`row=-1`表示该步骤不对应某一输入行。只有`store.done`之后才推进已完成K位置，`column.done`之后才推进已完成列数；相邻begin/done只缩小执行区间，不能单凭它断言某条硬件指令故障。

独立ELF符号`board_pack_probe`是64字节对齐的256字节版本1快照，保留帧、会话、输入/缓存地址、行标签、转换数量、阶段及已完成进度。偶数generation与覆盖前31个u64字段的64位逐字校验拒绝撕裂读取；host每次最多重读3次，另保存原始BIN和SHA256。原816/832字节诊断和24320字节计时布局保持不变。

软件检查列/K推进、VL范围、每列最多三次行检查及8192次事件上限；真实板端首块超过400M周期也报`0x35`。reason依次为1列越界、2列不推进、3重复行检查、4K不推进、5VL/K增量非法、6列K未完成、7块列未完成、8事件超限、9周期超限。40MHz下400M周期约10秒；若访存本身不能返回，软件检查无法执行，由host观察无推进或整帧超时后保存末次快照。QEMU周期速率不同，功能回归不使用实板周期阈值，阈值由确定的原生计数夹具验证。

结束标志为`PACK PROBE: COMPLETED/FAILED/STOPPED`；`COMPLETED`要求完整帧返回及有效`block.done`，不是原故障已修复的结论。新增调用、spill、日志和同步可能改变原现象，诊断帧自动排除性能排行。反馈末尾一页报告和UART最后几条标记即可，完整数据保留host；先保留现场，不自动重试整帧或复位。

## host 日志与 DDR 快照

单图日志在 `yolo26_pc/outputs/board-pcie/<时间>/`，顺序视频在 `yolo26_pc/outputs/host-video/run-*/`（旧摄像头入口仍在 `outputs/live-demo/<时间>/`），部署日志在 `outputs/deploy-*/load.log`。`diagnostics.jsonl` 包括 PCIe 工具参数、退出码、输出尾部、状态变化和完成/失败时的板端快照。Python 异常保存 traceback。UART 文字不自动写入 JSONL，需使用终端会话日志另存。

`board_diag` 的前816字节维持版本1布局；新固件追加两个uint64字段（计时表地址、长度），共832字节，64字节对齐。PCIe信息区通告实际长度，新Host兼容816/832字节。旧RX/RPC计数字段保留。诊断快照经缓存clean发布，地址见 `firmware/pcie-symbols.txt`。

`board_profile` 是独立计时表，完成后保存到逐帧JSON的 `pcie_timing.board_profile`。版本、布局、计时边界及无效记录规则只在[模型手册](MODEL.md#计时字段与解释)维护；诊断快照本身不是数值验收结果。

CPU trap 使用独立 4 KiB 异常栈，保存整数寄存器及 `mcause/mepc/mtval`，尽力打印并发布 PCIe ERROR，然后停车；不自动恢复，不保存浮点/向量/矩阵寄存器。DDR、缓存指令或异常栈也故障时，记录可能不完整。

| 错误 | 含义 |
|---|---|
| 1/2 | 权重长度/图接口不符 |
| 3/4 | AMU 初始化/自测失败，detail 为原返回值 |
| 5 | READY前权重准备失败；检查模型元数据、FP16转换和5MiB容量 |
| 0x30 | 输入 NaN/Inf，查看 tensor/element/observed |
| 0x35 | 首个输入块进度保护触发，reason见首块自动诊断；保留最后PACK快照 |
| 0x31 | 算子失败，查看 node/phase/detail |
| 0x32 | 输出 NaN/Inf，查看张量、下标和原始位 |
| 0x40–0x44 | PCIe 请求、会话、输入 CRC、模型失败或采集取消，见 PCIE.md |
| 0x100 | CPU trap，查看 trap_* 和 registers |

对 `0x30/0x32`，`observed` 是检测到的FP32原始位；`expected=0` 是未设置的诊断字段，不代表模型期望值为零。

### 旧版节点305误报

旧版 `5af2816` 将常量视图错误地从arena读取，可报 `tensor=516 element=0 got=0xffffffff / error=0x32`，随后PCIe发布0x43；这不能证明COPY计算出错。现已按constant标志选择权重或arena，并跳过未保留输出。遇到该旧特征先核对固件版本及配套符号。

AMU 等待前记录阶段，返回后记录原始 `xmfflags`，不将所有非零 flags 判为错误。指令或总线访问无法返回时，C 无法在指令内部实现超时，由 host 超时及同事调试器定位 PC/反汇编。

## 数值问题的定位资料

逐节点实际输入重放、golden规则及通过条件统一见[VALIDATION](VALIDATION.md)。主菜单8进入排障，子菜单2采集、3查看旧BUS报告；操作见[OPERATIONS](OPERATIONS.md#4-查验排障与反馈)，不为查看结果重新运行板子。

保留 `summary.json`、`nodes.csv`、`first_difference.json`、`snapshots/`、`tensors/` 及工具日志。首差异报告用于区分当前节点问题与上游传播；旧首层专项固件已移除。需要导出时使用 `tools/export_bus_evidence.py`。

单图/节点采集图的PNG生成使用librsvg/Cairo，不要求DISPLAY；视频PNG由OpenCV保存。只有查看窗口才需要图形会话。旧SVG需转换时运行 `bash tools/host.sh png`，不再占用菜单。节点采集增加CRC、读回和等待开销，不能用于评估推理性能。

## 自测定位

当前 `clp-selftest` 使用可精确表示的整数，连续32轮循环 `(M,N,K)=(3,5,17)/(1,1,1)/(2,3,32)/(3,5,31)`，逐位比较、不设误差阈值。`frame` 是轮次0–31，不是视频帧；轮次除以4的余数对应上述形状。UART 失败行也直接打印 M/N/K。

A/B/C/D各512字节，DDR相对偏移为 `0x000/0x200/0x400/0x600`；有效行从各区 `+0x40` 起、行距64字节。A为 `[M,K]`，B按 `[N,K]` 转置装载，C/D为 `[M,N]`；另2KiB为保护区，总计独占4KiB。

`tensor`（UART 的 `buffer`）是64字节块编号：0–7为A、8–15为B、16–23为AMU输出C、24–31为RVV输出D、32–63为额外保护区；`element` 是块内32位字下标。每个缓冲区首块为保护行，随后才是有效行。失败全局DDR地址为 `0xF0000000 + tensor×64 + element×4`，RVV别名为 `0x10000000 + tensor×64 + element×4`。

CLP失败 `error=4`，`detail=-200-tensor`，`observed/expected` 是原始32位字。A/B每字含低/高两个FP16，奇数K的末字高半字保护值为 `3555`；C/D行内前N字有效，其余为 `c49a4000`。`board_test_stage`：1全区写读检查，4依赖及复用，3通过；错误码0/4/3/0x100表示通过/数据不符/AMU初始化失败/trap。独立固件结束后停车于 `clp_probe_halt`；模型启动复用自测，通过后进入READY，失败保留诊断并发布PCIe ERROR。

历史缓存selftest及固定地址GDB脚本已移除。独立CLP自测仍保留；使用各固件对应的ELF确认符号，不能混用地址。

实板保持DCache开启，单图和连续帧验收见 [验收状态](VALIDATION.md)。CLP模型的数据为全局高地址，调试RVV时使用对应CLP别名；读取PCIe诊断仍使用低地址符号。

## UART 接口核对来源

已核对千问交接包固定提交 `21e7253b7caf7d34864fb82a807720f7eba00bbc` 的 [main_baremetal.c](https://github.com/trollsmash/n550_qwen3_bringup_handover/blob/21e7253b7caf7d34864fb82a807720f7eba00bbc/fpga_bringup/src/src/main_baremetal.c#L73)：`uart_init()` 初始化；`putc_raw()` 等待 LSR bit5 后写 THR；`putc_()` 处理换行；`P/U/I/X` 分别打印字符串、无符号数、有符号数和十六进制。

YOLO 使用独立 `board/uart.c` 的 `board_uart_init/put/text/u64/i64/hex`，保留动态 DLF 检测，超时后不写 THR 并关闭后续日志。仅核对平台 UART 接口，未参考或复用千问算子。
