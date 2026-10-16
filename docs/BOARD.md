# 板端条件

操作命令统一见 [README](../README.md)，本页仅定义平台条件和同步边界。

| 项目 | 固定配置 |
|---|---|
| CPU | N550，RV64，单 hart0，M 模式；其他 hart 停车 |
| 运算 | RVV+AMU FP16，FP32 激活与输出，334 节点 |
| DDR | 程序/权重/栈/诊断独占 `0x80000000–0x81FFFFFF` 32 MiB；模型数据独占 `0xF0000000–0xF0FFFFFF` 16 MiB |
| 数据区 | CLP共8,143,104字节：图工作区7,614,720、im2col 524,288、AMU两个输入块共4,096；常量9,678,644字节仍在原区 |
| RVV Memory port | `0x8000…` 地址使用 DCache，不能当作无缓存访问 |
| RVV CLP | `0x10000000–0x1FFFFFFF` 无缓存，NoC 映射为 `0xF0000000–0xFFFFFFFF` |
| AMU | 独立AXI使用全局DDR地址；模型的输入打包块和输出均在批准的16 MiB区 |
| DCache | 保持开启，cacheline 与维护步长均为 64 字节，标准 Zicbom |
| UART0 | DW_apb_uart，基址 `0x20100000`，32-bit MMIO，寄存器间距 4 字节 |
| UART 格式 | pclk=sclk=10 MHz，115200、8N1，无软硬流控，仅输出调试文字 |
| PCIe | 模型数据偏移范围 `0x70000000–0x70FFFFFF`；邮箱和诊断仍在原32 MiB区 |

## 加载与启动

平台方负责 DDR/时钟/引脚/缓存使能及 JTAG 配置。启动前 DCache 已开启，DDR 可访问，下载后的代码/数据对 CPU 可见，旧脏缓存不能覆盖下载内容。UART 时钟、复位和引脚须可访问；MMIO 总线异常会触发 CPU trap。

部署脚本使用现场已提供的 `reset1.sh` 保持 S2C 复位，加载完整 BIN 到 PCIe `0x0`，再用 `reset.sh` 释放复位。平台方须保证该时序有效、释放不清除 DDR、启动入口为 CPU `0x80000000`。我们的启动代码关闭中断，设置栈、异常入口，清 BSS，启用浮点/RVV 状态，不初始化 DDR、不关闭 DCache。

`demo-pcie.bin/elf` 为当前CLP模型版，启动通过RVV初始化高地址数据，执行32轮CLP自测，再发布READY。`.clp_data` 是NOLOAD，独立于普通BSS，BIN不含高地址空洞。独立 `clp-selftest.bin/elf` 保留现场已通过版本。历史缓存自测已移除。

UART 动态识别 4/5/6 位 DLF 并计算小数分频，验证寄存器；实际波特率的整数记录约为 114942/115273。空闲检测、分频验证或发送等待失败后禁用文字输出，记录 `board_uart_status` 和 `board_diag.tx_errors`，不阻止 PCIe 模型执行。每个 MMIO 访问本身无法返回或触发异常时，不能靠轮询上限恢复。

## 缓存与执行同步

RVV对模型数据的连续、跨步和索引读取均转换成CLP别名，标量及AMU使用全局DDR地址。普通缓存区的RVV写入只做fence，禁止写后invalidate；原错误交接已修正。编译器隐式RVV拷贝保持关闭，便于审查实际路径。

CLP 测试的数据路径为 RVV 别名写入 → `fence iorw,iorw` → AMU 全局地址读取/运算/存储 → 读取 `xmfflags` 等待 → fence → RVV 别名读取/存储 → fence。整个共享区不做标量访问或 CBO，消除该最小链路的 DCache 所有权切换；AMU 完成语义仍须实板确认。

标量若参与共享缓冲区，仍须在标量→无缓存读取前 clean、外部写入后标量读取前 invalidate，且完整缓存行独占、先等待外部完成。低地址诊断继续 clean 发布；PCIe 邮箱协议不变，见 [PCIe 协议](PCIE.md)。FENCE 负责顺序，不能替代缓存维护或 AMU 完成等待。

现场已报告独立CLP扩展版32轮通过，形状为 `(3,5,17)/(1,1,1)/(2,3,32)/(3,5,31)`。模型启动复用同一用例；当前BUS的334节点旧采集已完成逐位解释，见[验收状态](VALIDATION.md)。

A/B/C/D 各512字节，DDR 相对偏移为 `0x000/0x200/0x400/0x600`，有效行都从各区 `+0x40` 起、行距64字节。A 为 `[M,K]`、B 为转置的 `[N,K]`、C/D 为 `[M,N]`；剩余2 KiB为保护区。全部在已确认独占的4 KiB内，不扩大实板访问范围。

QEMU使用统一RAM，不模拟S2C重映射或物理DCache。模型测试使用 `0x84000000` 的模拟数据区，实板固件使用 `0xF0000000`；测试镜像不交付为板端固件。

## 故障定位

优先看 UART 文字及 host 日志，随后检查 `board_diag`。保留 `board_error`、`board_current_node`、`board_test_stage`、`board_trap_cause/pc/value` 和 `board_uart_status`；含义见 [诊断说明](DIAGNOSTICS.md)。

依据为用户提供的 core/NoC 地址图、DW_apb_uart 4.02a 手册及已确认板级参数；厂商手册不能替代实际 RTL 参数。
