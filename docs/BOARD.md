# 板端平台契约

本页只定义软件依赖的硬件前提、地址映射和缓存/完成语义。运行命令见[OPERATIONS](OPERATIONS.md)，模型布局与各缓冲区的具体交接见[IMPLEMENTATION](IMPLEMENTATION.md)。

## 平台与地址

| 项目 | 固定配置 |
|---|---|
| CPU | N550，RV64，单hart0，M模式；其他hart停车 |
| 指令 | RVV、AMU FP16、标准Zicbom缓存指令 |
| 普通DDR | CPU `0x80000000–0x81FFFFFF`，独占32MiB，供程序、常量、BSS、栈和诊断使用 |
| 模型DDR | 全局 `0xF0000000–0xF0FFFFFF`，独占16MiB；当前预留布局见实现总览 |
| RVV Memory port | `0x8000…` 等普通内存地址经过DCache，不能当作无缓存访问 |
| RVV CLP | `0x10000000–0x1FFFFFFF` 无缓存，NoC重映射为 `0xF0000000–0xFFFFFFFF` |
| AMU | 独立AXI，使用全局DDR地址；不能给它RVV CLP别名 |
| DCache | 保持开启，cacheline和维护步长均为64字节 |
| PCIe | CPU运行期间可访问DDR；地址为CPU全局地址减 `0x80000000` |
| UART0 | DW_apb_uart，基址 `0x20100000`，32-bit MMIO，寄存器间距4字节 |
| UART格式 | pclk=sclk=10MHz，115200、8N1，无软硬流控，仅调试输出 |

完整模型只使用已批准的两个独占DDR区；CLP地址范围不意味着整个范围都能随意使用。符号以当前 `firmware/pcie-symbols.txt` 和配套ELF为准，禁止沿用旧固件的固定地址。

## 启动前提与责任

平台方负责DDR、时钟、引脚、缓存使能、PCIe驱动及JTAG。软件启动前，DDR已可访问，DCache开启；下载后的指令/数据对CPU可见，旧脏缓存不能覆盖下载内容。UART MMIO不可访问会触发总线异常，不只是打印超时。

现场 `reset1.sh` 保持S2C复位，PCIe加载完整BIN后 `reset.sh` 释放。平台方保证释放不清DDR，并从CPU `0x80000000` 启动。我们的启动代码关闭中断、设置栈与异常入口、清普通BSS、启用浮点/RVV状态；不初始化DDR，也不关闭DCache。

高地址 `.clp_data` 为独立NOLOAD段，BIN不含高地址空洞；启动时通过RVV初始化。程序执行CLP自测及权重准备后才发布READY。正常推理不重复部署。

## 缓存与执行完成

- 标量和AMU访问全局地址，RVV模型张量访存使用CLP别名。普通缓存内存的RVV写入只做fence，不能盲目写后invalidate；编译器隐式RVV拷贝保持关闭。
- 标量产生的数据在外部/无缓存读取前clean；外部写入后，标量读取前invalidate。共享缓冲区完整缓存行独占，禁止并发混用缓存写和无缓存写。
- 覆盖外部将要写的缓冲区前，先等待旧使用者完成并交出缓存所有权，防止旧脏行以后回写覆盖新数据。不能只在写后补一次维护。
- AMU指令完成由读取 `xmfflags` 的既有约定等待，再执行fence并交接结果。FENCE只约束顺序，不能替代CBO，也不能自行证明AMU已完成。
- 不含标量参与的最小CLP链路为：RVV别名写 → fence → AMU全局地址读/算/写 → 等待 → fence → RVV别名读/写 → fence；该共享区不做标量访问或CBO。
- 低地址邮箱、诊断和计时表仍通过缓存维护发布；各写入方分占缓存行，见[PCIE](PCIE.md)。

逐缓冲区的权重、输入元数据、scratch及输出规则只在[实现总览](IMPLEMENTATION.md#4-缓存与执行顺序)维护。CLP测试形状、保护值和错误字段见[DIAGNOSTICS](DIAGNOSTICS.md#自测定位)，已通过范围见[VALIDATION](VALIDATION.md)。

QEMU使用统一RAM，不模拟S2C重映射或物理DCache。测试镜像的数据区可位于 `0x84000000`，不能作为实板 `0xF0000000` 布局的固件加载。

平台依据为用户提供的core/NoC地址图、DW_apb_uart 4.02a手册及已确认板级参数；厂商手册不能替代实际RTL参数。
