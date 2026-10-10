# 板端条件

操作命令统一见 [README](../README.md)，本页仅定义平台条件和同步边界。

| 项目 | 固定配置 |
|---|---|
| CPU | N550，RV64，单 hart0，M 模式；其他 hart 停车 |
| 运算 | RVV+AMU FP16，FP32 激活与输出，334 节点 |
| DDR | Memory port，独占 `0x80000000–0x81FFFFFF` 共 32 MiB |
| 数据区 | 图工作区 7,614,720 字节，常量 9,678,644 字节，另有代码、scratch、256 KiB 主栈 |
| CLP | 不使用 `0x10000000→0xF0000000` 重映射；标量/RVV/AMU 使用同一 DDR 地址 |
| DCache | 保持开启，cacheline 与维护步长均为 64 字节，标准 Zicbom |
| UART0 | DW_apb_uart，基址 `0x20100000`，32-bit MMIO，寄存器间距 4 字节 |
| UART 格式 | pclk=sclk=10 MHz，115200、8N1，无软硬流控，仅输出调试文字 |
| PCIe | CPU 运行期间可访问 DDR，地址为 CPU 地址减 `0x80000000` |

## 加载与启动

平台方负责 DDR/时钟/引脚/缓存使能及 JTAG 配置。启动前 DCache 已开启，DDR 可访问，下载后的代码/数据对 CPU 可见，旧脏缓存不能覆盖下载内容。UART 时钟、复位和引脚须可访问；MMIO 总线异常会触发 CPU trap。

部署脚本使用现场已提供的 `reset1.sh` 保持 S2C 复位，加载完整 BIN 到 PCIe `0x0`，再用 `reset.sh` 释放复位。平台方须保证该时序有效、释放不清除 DDR、启动入口为 CPU `0x80000000`。我们的启动代码关闭中断，设置栈、异常入口，清 BSS，启用浮点/RVV 状态，不初始化 DDR、不关闭 DCache。

模型固件只有 `demo-pcie.bin`；对应 ELF 供符号和调试。`selftest.bin`/`selftest.elf` 是独立缓存/RVV/AMU 自测，不接受图像任务。模型也会在启动时自动执行相同自测。

UART 动态识别 4/5/6 位 DLF 并计算小数分频，验证寄存器；实际波特率的整数记录约为 114942/115273。空闲检测、分频验证或发送等待失败后禁用文字输出，记录 `board_uart_status` 和 `board_diag.tx_errors`，不阻止 PCIe 模型执行。每个 MMIO 访问本身无法返回或触发异常时，不能靠轮询上限恢复。

## 缓存与执行同步

板端构建包含 `board/coherent_rvv.h`：连续/跨步 RVV 读前 clean，写前 flush，写后 fence 并 invalidate；索引数组在向量读取前 clean。AMU 输入打包后 clean，输出写前 flush，矩阵完成等待后 invalidate，再允许标量/RVV 读取。

`cbo.*` 维护缓存，`fence rw,rw` 保证内存顺序，AMU 完成等待保证矩阵执行结束，三者不能替代。当前 AMU 等待约定为读取 `xmfflags`，仍须实板确认完成语义。

维护覆盖首尾完整缓存行，单 hart、禁中断且不得有其他任务并发修改交接缓存行。PCIe 邮箱分离 host/板端写区；输入、输出的所有权转换见 [PCIe 协议](PCIE.md)。

自测覆盖标量↔RVV 部分缓存行、AMU 1×1 乘法，以及 32 轮 RVV→AMU→RVV 依赖和缓冲区复用。QEMU 不模拟 N550 非一致 DCache，不能替代实板自测与完整模型数值验收。

## 故障定位

优先看 UART 文字及 host 日志，随后检查 `board_diag`。保留 `board_error`、`board_current_node`、`board_test_stage`、`board_trap_cause/pc/value` 和 `board_uart_status`；含义见 [诊断说明](DIAGNOSTICS.md)。

依据为用户提供的 core/NoC 地址图、DW_apb_uart 4.02a 手册及已确认板级参数；厂商手册不能替代实际 RTL 参数。
