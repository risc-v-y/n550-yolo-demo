# 板端条件

操作命令统一见 [README](../README.md)，本页仅定义平台条件和同步边界。

| 项目 | 固定配置 |
|---|---|
| CPU | N550，RV64，单 hart0，M 模式；其他 hart 停车 |
| 运算 | RVV+AMU FP16，FP32 激活与输出，334 节点 |
| DDR | 程序/栈/诊断独占 `0x80000000–0x81FFFFFF` 共 32 MiB；CLP 测试另占 `0xF0000000–0xF0000FFF` 4 KiB |
| 数据区 | 图工作区 7,614,720 字节，常量 9,678,644 字节，另有代码、scratch、256 KiB 主栈 |
| RVV Memory port | `0x8000…` 地址使用 DCache，不能当作无缓存访问 |
| RVV CLP | `0x10000000–0x1FFFFFFF` 无缓存，NoC 映射为 `0xF0000000–0xFFFFFFFF` |
| AMU | 独立 AXI 使用全局 DDR 地址；CLP 测试使用 `0xF0000000` 起的地址 |
| DCache | 保持开启，cacheline 与维护步长均为 64 字节，标准 Zicbom |
| UART0 | DW_apb_uart，基址 `0x20100000`，32-bit MMIO，寄存器间距 4 字节 |
| UART 格式 | pclk=sclk=10 MHz，115200、8N1，无软硬流控，仅输出调试文字 |
| PCIe | 运行时可访问 DDR，偏移为全局 DDR 地址减 `0x80000000`；CLP 测试为 `0x70000000` |

## 加载与启动

平台方负责 DDR/时钟/引脚/缓存使能及 JTAG 配置。启动前 DCache 已开启，DDR 可访问，下载后的代码/数据对 CPU 可见，旧脏缓存不能覆盖下载内容。UART 时钟、复位和引脚须可访问；MMIO 总线异常会触发 CPU trap。

部署脚本使用现场已提供的 `reset1.sh` 保持 S2C 复位，加载完整 BIN 到 PCIe `0x0`，再用 `reset.sh` 释放复位。平台方须保证该时序有效、释放不清除 DDR、启动入口为 CPU `0x80000000`。我们的启动代码关闭中断，设置栈、异常入口，清 BSS，启用浮点/RVV 状态，不初始化 DDR、不关闭 DCache。

`clp-selftest.bin/elf` 为当前独立 CLP 最小测试；共享数据运行时由 RVV 初始化，BIN 不包含到高地址的空洞。`demo-pcie.bin/elf` 为原模型固件，`selftest.bin/elf` 为旧缓存自测，两者未按新确认的 RVV 缓存条件修正，暂不能作为板端功能已通的证据。最小测试不接受图像任务。

UART 动态识别 4/5/6 位 DLF 并计算小数分频，验证寄存器；实际波特率的整数记录约为 114942/115273。空闲检测、分频验证或发送等待失败后禁用文字输出，记录 `board_uart_status` 和 `board_diag.tx_errors`，不阻止 PCIe 模型执行。每个 MMIO 访问本身无法返回或触发异常时，不能靠轮询上限恢复。

## 缓存与执行同步

平台已确认：RVV 的 Memory port 使用 DCache，只有 CLP 窗口无缓存。旧 `coherent_rvv.h` 的“RVV 写后 invalidate”可能丢失脏数据，旧 RVV→AMU 交接也缺少 clean；本轮仅增加独立 CLP 测试，不声称模型已修复。编译器隐式 RVV 栈拷贝已取消，属于便于审查的代码生成控制，不能据此认定缓存故障根因。

CLP 测试的数据路径为 RVV 别名写入 → `fence iorw,iorw` → AMU 全局地址读取/运算/存储 → 读取 `xmfflags` 等待 → fence → RVV 别名读取/存储 → fence。整个共享区不做标量访问或 CBO，消除该最小链路的 DCache 所有权切换；AMU 完成语义仍须实板确认。

标量若参与共享缓冲区，仍须在标量→无缓存读取前 clean、外部写入后标量读取前 invalidate，且完整缓存行独占、先等待外部完成。低地址诊断继续 clean 发布；PCIe 邮箱协议不变，见 [PCIe 协议](PCIE.md)。FENCE 负责顺序，不能替代缓存维护或 AMU 完成等待。

CLP 测试仅覆盖 M=N=K=1 的 FP16 乘法、FP32 RVV 加法、32 轮复用和 4 KiB 保护区；不验证全部矩阵形状或完整模型。QEMU 使用统一 RAM，不模拟 S2C 地址重映射、CLP 路由或物理 DCache 行为。

## 故障定位

优先看 UART 文字及 host 日志，随后检查 `board_diag`。保留 `board_error`、`board_current_node`、`board_test_stage`、`board_trap_cause/pc/value` 和 `board_uart_status`；含义见 [诊断说明](DIAGNOSTICS.md)。

依据为用户提供的 core/NoC 地址图、DW_apb_uart 4.02a 手册及已确认板级参数；厂商手册不能替代实际 RTL 参数。
