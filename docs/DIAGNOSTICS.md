# host 显示与故障诊断

链路：**host 视频/预处理 → UART → 开发板 → UART → host 后处理、绘框、保存 → 开发机远程查看 host 桌面**。这里 host 指物理连接开发板的 Linux 电脑。程序与权重由软件同事加载，网络转发暂不实施。

## 使用

同事加载本次配套 `demo.elf` 后，在 host 的仓库根目录执行：

```bash
python3.10 -m venv .venv-host
.venv-host/bin/python -m pip install -r yolo26_pc/requirements-board.txt
# 先查串口；此操作不发送数据
.venv-host/bin/python yolo26_pc/serial_probe.py
# 将 ttyUSB0 换成实际设备，关闭 PuTTY/minicom
.venv-host/bin/python yolo26_pc/uart_backend.py --port /dev/ttyUSB0 --ping 100
.venv-host/bin/python yolo26_pc/uart_backend.py --port /dev/ttyUSB0 --image yolo26_pc/samples/bus.jpg
# 图像通过后，运行视频。窗口开在 host 的当前图形会话中。
.venv-host/bin/python yolo26_pc/live_demo.py --backend uart --serial-port /dev/ttyUSB0 --source yolo26_pc/samples/pexels-3796613.mp4 --max-results 2
```

必须在能显示窗口的 host 桌面会话执行；普通 SSH 终端未必有图形显示环境。可先加 `--no-display` 验证文件输出。Python 3.10 的 UART 模式只需上述三个依赖，不执行面向本地 PyTorch 的 `setup.ps1`。

新版 host 的模型演示入口要求固件支持诊断扩展；旧固件只适合原版本 host 或串口 HELLO 探测。请配套更新，协议错误 2 可能意味着仍在运行旧固件。

## 日志分级

- 默认：固定内存诊断记录持续更新；UART 发送帧开始/结束及错误快照。host 保存 `diagnostics.jsonl`，每条立即 flush，并输出到终端。
- `--board-trace`：每个节点开始/结束都发送快照，包含节点号（从 0 开始）、算子、输入输出张量编号/类型/形状、阶段和周期计数。仅定位问题时启用。
- `--check-intermediates`：检查每个节点 FP32 输出的 NaN/Inf。默认已经检查输入和最终输出；中间检查有额外扫描开销，不检查完整数值精度。

逐节点快照 816 字节，加协议头后每条 848 字节；334 个节点开始/结束的纯串口传输额外约 49 秒（115200、8N1），实际还有软件开销。`node_cycles` 是节点开始日志发完之后到结束日志发送之前的 cycle 差，包含维护与可选检查开销，不能直接当作优化后性能。没有心跳线程或看门狗，持续收到日志不延长 host 的 RUN 总等待期限；必要时调整 `--infer-timeout`。

单图/通信日志默认在 `yolo26_pc/outputs/board-uart/<时间>/`；视频日志、报告及录像在 `yolo26_pc/outputs/live-demo/<时间>/`。host 异常保存类型、消息和完整 Python traceback。JSONL 中仍保留之前已经写出的记录，即使后续任务失败；不保证突然断电时磁盘落盘。

## 板端无法正常应答时

通过同事已有调试器查看 ELF 符号 **`board_diag`**：816 字节、64 字节对齐的静态结构，具体地址查本版本 `symbols.txt`；地址不保证跨构建相同。含启动阶段、帧号、节点、子阶段、错误、UART 计数、最近 AMU flags、异常 CSR 及 x0–x31。另保留旧的 `board_error`、`board_current_node`、`board_test_stage` 符号；优先读取经过缓存同步的 `board_diag`。

CPU trap 使用独立 4 KiB 异常栈，保存异常前的整数寄存器及 `mcause/mepc/mtval`，然后停止；不保存浮点/向量/矩阵寄存器，不尝试自动恢复执行。正常诊断更新执行缓存 clean；若 DDR、本身的缓存指令或异常栈访问也故障，记录可能不完整。异常记录期间再次 fault 会进入停车入口，不保证能通过 UART 输出。

AMU 等待前保存阶段，等待返回后保存原始 `xmfflags`。不擅自解释其位定义，也不将所有非零 flags 都判为错误。若等待指令或总线访问本身无法返回，C 代码不能在该指令内部实现超时；host 超时后由同事暂停 CPU，结合 PC、反汇编和最后记录定位。

| 错误 | 含义 |
|---|---|
| 1/2 | 权重长度/图接口不匹配；初始化 UART 前失败，需调试器 |
| 3/4 | AMU 初始化/自检失败，detail 保留返回值；UART 已初始化时 demo 仍可响应诊断，但拒绝模型任务 |
| 0x21–0x25 | UART 初始化失败，需调试器 |
| 0x30 | 输入出现 NaN/Inf；tensor/element/observed 保存位置和原始位 |
| 0x31 | 算子失败；node、phase、detail 保留定位信息 |
| 0x32 | 输出出现 NaN/Inf，含张量编号、元素偏移和原始位 |
| 0x100 | CPU 异常，查看 trap_* 与 registers |

串口快照中的 `rx_timeouts` 包括无数据时的正常轮询超时，不单独作为链路故障判断；`rx_errors` 是 UART LSR 报错，`tx_errors` 是发送轮询超时，另统计 CRC 错误、协议拒绝和重复请求。错误快照不意味着自动恢复，数值结果仍需与基准核对。

依赖版本按 Python 3.10 选择：[NumPy 2.2.6 发布元数据](https://pypi.org/pypi/numpy/2.2.6/json)、[OpenCV 4.11.0.86 发布元数据](https://pypi.org/pypi/opencv-python/4.11.0.86/json)。实板 UART、缓存可见性、真实 trap 和 GUI 会话仍需实验室验证。
