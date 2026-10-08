# 板端条件与逐步验收

## 固定配置

| 项目 | 本次设置 |
|---|---|
| CPU | N550，RV64，单hart0，M模式；其他hart停车 |
| 运算 | RVV+AMU FP16，FP32激活与输出；不改数值计算顺序 |
| DDR | Memory port，独占0x80000000–0x81FFFFFF，平台已保证可直接读写 |
| 数据区 | 图工作区7,614,720字节；模型常量9,678,644字节；另有代码、scratch和256 KiB栈 |
| CLP | 首版不使用0x10000000→0xF0000000重映射，RVV/标量/AMU均使用DDR同一地址 |
| DCache | 保持开启；cacheline与本次缓存维护步长64字节；标准Zicbom |
| UART0 | Synopsys DW_apb_uart，0x20100000，32-bit MMIO、寄存器间距4字节 |
| 时钟/格式 | pclk=sclk=10 MHz，115200、8N1、无RTS/CTS和XON/XOFF |
| 中断 | UART中断号16，首版采用轮询，不配置中断控制器 |

小数分频已确认支持。驱动在UART空闲时保存DLF、写入低6位并读回识别4/5/6位掩码、恢复旧值，再按有效位宽四舍五入计算分频并验证寄存器。4/5位得到约114943 baud，6位约115274 baud。遇到不支持的掩码或忙状态超时，启动失败，不悄悄使用整数分频。

**两套方案共同的平台交接条件**：启动前DCache已开启，DDR已可读写，上述32 MiB没有其他使用者。UART固件及独立自测还要求UART时钟/复位/引脚可用；PCIe固件要求host可在CPU运行期间访问DDR，不初始化UART。代码不包含DDR控制器、时钟树、引脚复用或厂商缓存使能CSR设置。无论通过PCIe写入BIN还是以支持ELF的方式下载UART固件，都应保证已加载代码/数据对CPU可见，避免旧cache内容覆盖下载内容；入口 `_start` 位于CPU地址 `0x80000000`。我们的启动代码关闭中断、设置栈和异常入口、清BSS、启用浮点/RVV状态，不关闭DCache。

## 缓存同步

板端构建强制包含 `board/coherent_rvv.h`，QEMU构建不启用。连续/跨步RVV读前clean，写前flush、写后完成内存排序并invalidate；索引源在节点入口clean，标量生成的索引数组在向量读取前clean。Softmax内部标量归约读到的是已同步的RVV结果。

已确认平台支持标准 `cbo.clean/flush/inval` 和 FP16 AMU 指令。AMU输入打包后clean；输出各行在写前flush，矩阵完成等待后invalidate，再允许标量读取。`cbo.*`负责缓存内容，`fence rw,rw`负责DDR访问顺序，矩阵完成等待负责AMU执行完成；三者不能互相替代。当前完成等待采用读取 `xmfflags` 的N550约定，仍须实板确认，不能仅由“支持FP16指令”推断其完成语义。

缓存维护覆盖首尾完整cacheline，避免部分行旧脏数据回写覆盖新数据。首版单hart、禁中断，不允许其他任务并发修改交接缓存行。保守维护有开销，暂不优化；实际RVV/AMU完成语义与缓存可见性必须经板上自检和模型结果确认。

## 上板顺序与方案选择

软件同事按 [根目录操作指南](../README.md) 选择PCIe或UART完整流程，两种模型固件不能同时运行。PCIe加载 `firmware/demo-pcie.bin`（PCIe偏移 `0x0`），对应的ELF仅供符号/调试；[PCIe接口说明](PCIE.md)记录在 `0x81F00000` 保留的256字节状态接口，诊断通过DDR读取。以下仅列UART的板端检查顺序；两套方案共用模型、DDR独占范围和缓存/AMU交接条件。

1. **下载 `firmware/selftest.elf`**：验证UART、标量↔RVV部分cacheline写入、AMU 1×1乘法，再运行32轮RVV→AMU→RVV直接依赖和缓冲区复用自测。成功输出 `RVV->AMU->RVV REUSE 32 ROUNDS PASS` 和 `CACHE/RVV/AMU TEST PASS`。这是快速检查，不替代全模型验收。
2. **下载 `firmware/demo.elf`**：先运行相同自检，再进入二进制协议等待状态。关闭其他占用串口的软件，host执行100次1024字节ping校验。
3. **单图bus/zidane**：PC预处理、逐包传输、板端执行334节点、PC回传后绘框。记录上传、RUN往返、下载耗时；逐项核对与reference候选框及检测结果的差异，不设经验阈值掩盖错误。
4. **连续两帧以上**：确认帧号/结果对应，视频窗口等待时响应正常；断线、CRC错误、重复包、取消后重连都应明确处理。相同RUN重试只能推理一次。
5. **摄像头**：最后接入真实采集，确认预览、检测结果画面和录像。

## 故障定位

优先按 [诊断说明](DIAGNOSTICS.md) 查看 host 的 `diagnostics.jsonl` 或通过调试器读取 `board_diag`。它记录阶段、错误及原始返回码、节点、输入输出形状、AMU flags、异常 CSR 与整数寄存器，并执行缓存同步。

保留的旧符号包括：`board_error`（1权重长度、2图格式、3AMU能力、4自检失败、0x21–0x25 UART初始化失败、0x100异常）；`board_trap_cause/pc/value`；`board_test_stage`（1 RVV缓存、2 AMU、4直接依赖与复用、3全部通过）；`board_current_node`。旧符号用于兼容，完整诊断以 `board_diag` 为准。

UART固件不向协议串口写printf文本。没有INFO时先检查固件是否通过自检，再检查端口、接线、分频和时钟。传输异常可能需要等待板端当前RUN完成或由同事重新运行固件；程序不通过串口执行硬件复位。

依据：用户提供的core/NoC地址图、DW_apb_uart 4.02a手册及已确认的板级参数。厂商手册不是实际RTL参数配置的替代品；JTAG链路与调试器配置不属于本次交付范围。
