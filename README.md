# N550 YOLO26 板端验证操作指南

本页供物理连接 S2C 板卡的 Linux 实验室电脑（下称 **host**）上的软件同事使用。**PCIe 和 UART 是两套独立方案**：分别使用 `firmware/demo-pcie.elf` 和 `firmware/demo.elf`，同一时刻只加载、运行其中一个模型固件。两套方案都由 host 传图、N550 推理、host 绘框及录像；当前优先验收 PCIe。

本仓库已包含板端源码、预编译固件、样例和数值参考。固件构建、PCIe 模拟联调、UART 协议回归已经完成；**真实板卡的传输、缓存可见性、模型数值及吞吐仍须现场验证**。以下是待执行的上板流程，不能把本机模拟结果当作实板通过。

## 共同准备

以下命令均在 **host 的仓库根目录**执行。开发机可远程查看 host 的图形桌面；本流程不依赖开发机摄像头或跨机器转发。

```bash
git clone https://github.com/risc-v-y/n550-yolo-demo.git
cd n550-yolo-demo
sha256sum -c firmware/SHA256SUMS
python3 --version
python3 -m venv .venv-host
.venv-host/bin/python -m pip install -r yolo26_pc/requirements-board.txt
```

host 的 Python 应为已确认的 3.10；虚拟环境只安装板端 host 所需的 NumPy、OpenCV 和 pyserial，无需 PyTorch。若 host 缺少 `venv`、`pip` 或依赖安装权限，请先补齐。固件哈希检查必须全部通过；`firmware/` 是本次交付的预编译文件，重新构建的输出不会自动替换它。

加载任何固件前，由板级同事确认 DDR 可用且 `0x80000000–0x81FFFFFF` 这 32 MiB 由本演示独占、DCache 已开启、下载后的代码和数据对 CPU 可见。用现有加载工具按 **ELF 加载段地址**写入 DDR，从 `_start` 启动；不能把整个 ELF 当成裸二进制写到单一地址。DDR 初始化、复位、J-Link/JTAG 加载及启动由板级同事负责，固件不执行这些板级初始化。详细条件见 [板端条件](docs/BOARD.md)。切换 PCIe/UART 方案时，应加载对应固件并重新启动。

两套模型固件已嵌入模型图和权重，不用额外上传 `.pt` 或逐帧加载权重。输入为等比例缩放补边后的 RGB CHW `[1,3,416,416]` FP32（2,076,672 字节），结果为 `[300,6]` FP32 候选框（7,200 字节）；host 以 `score > 0.25` 绘框，不增加 NMS。两套方案都保留全部 334 个节点。请记录使用的 Git 提交、ELF 哈希、host 输出目录和板端启动方式，便于复现。

## 方案一：PCIe

链路：**host 采集/预处理 → PCIe 写 DDR → N550 RVV+AMU 推理 → PCIe 读结果 → host 绘框/录像**。板端使用 `firmware/demo-pcie.elf`；它不初始化 UART，状态和诊断从 DDR 读取。

### 1. 检查工具并启动固件

host 必须已有可用的 PCIe 驱动、设备访问权限及 `pbcopy`、`pbload`。先确认命令位置：

```bash
command -v pbcopy
command -v pbload
```

不在 `PATH` 内时，在下列 Python 命令末尾加 `--pbcopy /绝对路径/pbcopy --pbload /绝对路径/pbload`。由板级同事加载并启动 `firmware/demo-pcie.elf` **一次**，保持程序运行；不要用另一个仓库会复位并重写程序/权重的批处理脚本逐帧驱动。CPU 运行期间须允许 PCIe 访问 DDR。仅运行一个 host 控制程序，不要让其他 PCIe 工具同时改写本演示缓冲区。

### 2. 握手和单图

```bash
# 读取固件接口、建立会话；这一步不运行图像推理
.venv-host/bin/python yolo26_pc/pcie_backend.py

# 分别上传、推理、读回并对照历史 QEMU FP16 结果
.venv-host/bin/python yolo26_pc/pcie_backend.py --image yolo26_pc/samples/bus.jpg --reference reference/bus-qemu-fp16.bin
.venv-host/bin/python yolo26_pc/pcie_backend.py --image yolo26_pc/samples/zidane.jpg --reference reference/zidane-qemu-fp16.bin
```

每次运行会打印并建立独立的 `yolo26_pc/outputs/board-pcie/<时间>/` 目录。检查 `report.json` 的 `status`、板端信息、检测框和参考差异；单图还保存 `source.png`、`annotated.png`、`candidates.npy` 和 `diagnostics.jsonl`。握手成功只说明通信接口可读，仍须核对两张图的类别、置信度、坐标和绘框效果。不要预设经验阈值掩盖数值差异。

### 3. 两帧视频与连续演示

在能显示窗口的 host 桌面会话运行：

```bash
.venv-host/bin/python yolo26_pc/live_demo.py \
  --backend pcie \
  --source yolo26_pc/samples/pexels-3796613.mp4 \
  --max-results 2 --save-frames
```

先确认两帧的帧号、原图、检测图、候选框和录像；随后去掉 `--max-results 2` 连续运行，以 Q/Esc 退出。普通 SSH 终端没有图形桌面时加 `--no-display` 检查文件输出；host 本地摄像头可用 `--camera 0` 替换 `--source`。视频循环播放，推理选择当前最新帧，不保证处理素材的每一帧。

输出位于 `yolo26_pc/outputs/live-demo/<时间>/`，包括 `raw.mp4`、`demo.mp4`、`report.json`、`diagnostics.jsonl`；`--save-frames` 另外保存对应编号的原图、检测图、候选框和逐帧 JSON。录像采用 20 FPS 写文件，**实际处理速度以 `report.json` 的结果帧率和各阶段耗时为准**。

### 4. PCIe 验收和故障记录

记录握手、bus/zidane 数值对照、两帧视频、持续运行及实际耗时。工具返回 0 仍需由程序核对读回长度、CRC、会话和帧号。默认 `--infer-timeout 900` 覆盖一帧的传输及推理，`--tool-timeout 30` 限制一次工具调用；超时或退出不会复位板卡，也不能中断已开始的板端推理。失败后保留输出目录和 `diagnostics.jsonl`，先由板级同事检查板端状态，再重新连接。接口、缓存交接和错误码见 [PCIe 说明](docs/PCIE.md)，日志解释见 [诊断说明](docs/DIAGNOSTICS.md)。

## 方案二：UART

链路：**host 采集/预处理 → UART 上传 → N550 RVV+AMU 推理 → UART 下载 → host 绘框/录像**。板端使用 `firmware/demo.elf`，不是 PCIe 固件。UART 时钟、复位、引脚和接线须已就绪；host 串口设为 **115200、8N1，关闭硬件及软件流控**。

### 1. 可选：独立自测

需要先定位 UART、缓存、RVV 或 AMU 问题时，在 host 列出候选串口，按接线确认设备：

```bash
python3 yolo26_pc/serial_probe.py
```

先用 minicom/PuTTY 按上述串口参数打开正确端口，再由板级同事加载、启动 `firmware/selftest.elf`。期望依次看到：

```text
UART OK
RVV->AMU->RVV REUSE 32 ROUNDS PASS
CACHE/RVV/AMU TEST PASS
```

这验证启动时的自测，不能代替全模型数值验收。输出只在启动时发送；若终端打开太晚，请同事重新启动自测。两个模型固件启动时也运行相同自测；切换到模型演示前，须加载相应模型 ELF。

### 2. 加载 UART 模型、确认端口与通信

由板级同事加载并启动 `firmware/demo.elf`。关闭占用串口的 minicom/PuTTY，按实际设备替换 `PORT`：

```bash
python3 yolo26_pc/serial_probe.py
PORT=/dev/ttyUSB0
.venv-host/bin/python yolo26_pc/serial_probe.py --probe "$PORT"
.venv-host/bin/python yolo26_pc/uart_backend.py --port "$PORT" --ping 100
```

探测需确认固件 INFO 与预期输入/输出大小，100 次 ping 应全部通过。`firmware/selftest.elf` 只打印自测结果，不响应模型握手；若 probe 无响应，检查端口、占用、接线、时钟及板端启动状态，见 [串口探测](docs/SERIAL_PROBE.md)。

### 3. 单图数值与绘框

```bash
.venv-host/bin/python yolo26_pc/uart_backend.py --port "$PORT" --image yolo26_pc/samples/bus.jpg --reference reference/bus-qemu-fp16.bin
.venv-host/bin/python yolo26_pc/uart_backend.py --port "$PORT" --image yolo26_pc/samples/zidane.jpg --reference reference/zidane-qemu-fp16.bin
```

每次结果保存在 `yolo26_pc/outputs/board-uart/<时间>/`：`report.json`、`diagnostics.jsonl`、`annotated.png`、`candidates.npy`。检查 `status`、检测框、与参考候选框的差异和绘框效果。115200、8N1 下上传一帧约需 180 秒；UART 的 `--infer-timeout 900` 是每次 RUN 的等待上限，不包含上传时间。

### 4. 两帧视频与连续演示

```bash
.venv-host/bin/python yolo26_pc/live_demo.py \
  --backend uart --serial-port "$PORT" \
  --source yolo26_pc/samples/pexels-3796613.mp4 \
  --max-results 2 --save-frames
```

检查两帧结果后，去掉 `--max-results 2` 连续运行；无图形桌面可加 `--no-display`。host 本地摄像头可用 `--camera 0` 替换 `--source`。输出目录和逐帧文件与 PCIe 视频相同，但 `report.json` 标记 UART 传输；结果帧率才是处理速度。串口传输的是二进制协议，普通终端不能直接显示检测图。

### 5. UART 验收和故障记录

保留串口探测、ping、bus/zidane、两帧及持续运行的日志和耗时。失败时检查 `diagnostics.jsonl`、`report.json` 和板端 `board_diag`；必要时由板级同事重新启动固件。协议格式和重试约定见 [UART 协议](docs/UART_PROTOCOL.md)，错误定位见 [诊断说明](docs/DIAGNOSTICS.md)。

## 按需查阅与重新构建

| 文档 | 用途 |
|---|---|
| [板端条件](docs/BOARD.md) | DDR、缓存、AMU、UART 和加载条件 |
| [PCIe 说明](docs/PCIE.md) | DDR 接口、缓存交接、工具约定和失败处理 |
| [串口探测](docs/SERIAL_PROBE.md) | 端口、权限及占用排查 |
| [诊断说明](docs/DIAGNOSTICS.md) | host 日志、板端错误码与调试器读取 |
| [UART 协议](docs/UART_PROTOCOL.md) | 串口收发程序的协议细节 |
| [验证记录](docs/VALIDATION.md) | 已完成的本机检查与待完成的实板验收 |

只有需要修改或复验 C/汇编代码时，才在 Linux/WSL 构建机使用配套 ESWIN 工具链执行 `bash yolo26_riscv/build_board.sh`。工具链须支持 `xewmatrix1p0`，可通过 `TOOLCHAIN_PREFIX=/path/to/bin/riscv64-unknown-elf` 指定；构建结果位于 `yolo26_riscv/build/n550-board/`，不会自动替换 `firmware/` 下的交付文件。样例视频来源为 [Pexels 3796613](https://www.pexels.com/video/people-walking-on-the-street-3796613/)，SHA-256：`fcd2af324d05ae09da2570ffd693da084f6afdb209b078b419c30741742b43c5`。
