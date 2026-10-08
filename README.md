# N550 YOLO26 板端验证操作指南

本页供物理连接 S2C 板卡的 Linux 实验室电脑（下称 **host**）上的软件同事使用。**PCIe 和 UART 是两套独立方案**：PCIe 上板加载 `firmware/demo-pcie.bin`，UART 使用 `firmware/demo.elf`，同一时刻只加载、运行其中一个模型固件。PCIe 对应的 `demo-pcie.elf` 保留用于构建、符号和调试，不能直接交给 PCIe 加载工具。当前优先验收 PCIe：先用已预处理的固定图片跑通并生成绘框图，再在 host 具备 NumPy/OpenCV 后进行实时摄像头演示。

本仓库包含板端源码、预编译固件、样例和数值基准。本机的构建与协议检查已通过；**真实板卡的传输、缓存可见性、模型数值及吞吐仍须现场验收**。以下步骤用于上板验证，本机检查不代表实板通过。

## 共同准备

以下命令均在 **host 的仓库根目录**执行。开发机可远程查看 host 的图形桌面；本流程不依赖开发机摄像头或跨机器转发。

```bash
git clone https://github.com/risc-v-y/n550-yolo-demo.git
cd n550-yolo-demo
sha256sum -c firmware/SHA256SUMS
python3 --version
```

PCIe 第一步只使用 Python 3 标准库和现场已有的 `pbcopy/pbload`，**不执行 `pip`、不创建 `venv`，也不要求 host 联网**；本机在 Python 3.10 下验证，脚本最低使用 Python 3.8 的标准库功能。第二步的图像依赖在对应小节单独说明。固件哈希检查必须全部通过；`firmware/` 是本次交付的预编译文件，重新构建的输出不会自动替换它。

加载任何固件前，由板级同事确认 DDR 可用且 `0x80000000–0x81FFFFFF` 这 32 MiB 由本演示独占、DCache 已开启、下载后的代码和数据对 CPU 可见。PCIe 使用已转换的裸二进制，从 CPU 地址 `0x80000000`（PCIe 偏移 `0x0`）加载并启动；UART 的 ELF 仍需由支持 ELF 的加载方式按加载段地址写入，不能直接改名为 BIN。DDR 初始化、复位及启动由板级同事负责，固件不执行这些板级初始化；复位/启动步骤须保证已写入的 DDR 内容不会丢失。详细条件见 [板端条件](docs/BOARD.md)。切换 PCIe/UART 方案时，应加载对应固件并重新启动。

两套模型固件已嵌入模型图和权重，不用额外上传 `.pt` 或逐帧加载权重。输入为等比例缩放补边后的 RGB CHW `[1,3,416,416]` FP32（2,076,672 字节），结果为 `[300,6]` FP32 候选框（7,200 字节）；host 以 `score > 0.25` 绘框，不增加 NMS。两套方案都保留全部 334 个节点。请记录使用的 Git 提交、实际加载文件的哈希、host 输出目录和板端启动方式，便于复现。

## 方案一：PCIe

链路：**host 采集/预处理 → PCIe 写 DDR → N550 RVV+AMU 推理 → PCIe 读结果 → host 绘框/录像**。板端加载 `firmware/demo-pcie.bin`；它不初始化 UART，状态和诊断从 DDR 读取。

### 1. 检查工具并启动固件

host 必须已有可用的 PCIe 驱动、设备访问权限及 `pbcopy`、`pbload`。先确认命令位置：

```bash
command -v pbcopy
command -v pbload
```

由板级同事按现场 S2C 时序停止 CPU、确认 DDR 已初始化，再在仓库根目录写入本仓库的完整固件镜像：

```bash
pbcopy -d firmware/demo-pcie.bin:0x0
```

`0x0` 是 PCIe 偏移，对应 CPU 入口 `0x80000000`；`pbcopy` 只负责写入，不负责启动。写入后由板级同事按现场已验证的复位/启动步骤从该入口启动 **一次**，并确认复位不清除 DDR。不要把 `demo-pcie.elf` 改名为 BIN 上传。现有 `run_yolo26n.sh` 属于同一 S2C 板的另一套 YOLO 方案，会写入其他程序和数据段，不能原样套用或逐帧复位本演示。程序运行期间须允许 PCIe 访问 DDR；仅运行一个 host 控制程序，不要让其他 PCIe 工具同时改写本演示缓冲区。`pbcopy`、`pbload` 不在 `PATH` 内时，加载命令使用实际绝对路径，并在下列 Python 命令末尾加 `--pbcopy /绝对路径/pbcopy --pbload /绝对路径/pbload`。

### 2. 第一步：预处理图片上板、数值对照与静态绘框（免安装）

```bash
# 读取固件接口、建立会话；这一步不运行图像推理
python3 yolo26_pc/pcie_backend.py

# 分别传输已预处理的 bus/zidane，运行推理并生成可查看的绘框图
python3 yolo26_pc/pcie_prepared.py --sample bus
python3 yolo26_pc/pcie_prepared.py --sample zidane
```

两张图片的原始 JPEG、相同预处理流程产生的 RGB CHW FP32 输入、输入哈希和缩放补边参数已随仓库交付，见 `reference/prepared/manifest.json`；host 不需解码或预处理图片。每次命令建立独立的 `yolo26_pc/outputs/board-pcie/<时间>/` 目录，保存 `report.json`、`diagnostics.jsonl`、原始候选框 `candidates.bin` 和内嵌原图、检测框及类别标签的 `annotated.svg`。SVG 可在 host 的浏览器打开，或复制到有浏览器的电脑查看。检查报告中的 `status=completed`、`reference.byte_identical`、类别顺序、坐标/分数最大差异，以及绘框效果；`completed` 只表示传输及结果检查完成，数值差异仍须人工验收。握手成功不代表图像推理通过，不设经验阈值掩盖差异。

### 3. 第二步：NumPy/OpenCV 实时摄像头与视频演示

本步才需要 NumPy 2.2.6 和带 GUI 功能的 OpenCV 4.11.0.86，版本清单见 `yolo26_pc/requirements-board.txt`。先检查 host 已有的 Python 环境；缺少依赖时由部署方按 host 的 Linux 架构准备并验证兼容的离线包，第一步不受影响。不要求现场联网或在 host 上运行安装命令；本仓库不附带跨平台通用的 Python 依赖包。无需 PyTorch，PCIe 不需要 pyserial。

```bash
python3 -c 'import numpy, cv2; print(numpy.__version__, cv2.__version__)'
```

只有这条检查成功后，才在能显示窗口的 host 桌面会话执行下面的摄像头命令；若摄像头暂不可用，可先用仓库样例视频检查完整软件流程：

```bash
python3 yolo26_pc/live_demo.py --backend pcie --camera 0 --save-frames
```

```bash
python3 yolo26_pc/live_demo.py \
  --backend pcie \
  --source yolo26_pc/samples/pexels-3796613.mp4 \
  --max-results 2 --save-frames
```

先确认两帧的帧号、原图、检测图、候选框和录像；随后去掉 `--max-results 2` 连续运行，以 Q/Esc 退出。普通 SSH 终端没有图形桌面时加 `--no-display` 检查文件输出。视频循环播放，推理选择当前最新帧，不保证处理素材的每一帧。

输出位于 `yolo26_pc/outputs/live-demo/<时间>/`，包括 `raw.mp4`、`demo.mp4`、`report.json`、`diagnostics.jsonl`；`--save-frames` 另外保存对应编号的原图、检测图、候选框和逐帧 JSON。录像采用 20 FPS 写文件，**实际处理速度以 `report.json` 的结果帧率和各阶段耗时为准**。

### 4. PCIe 验收和故障记录

记录握手、bus/zidane 数值对照、两帧视频、持续运行及实际耗时。工具返回 0 仍需由程序核对读回长度、CRC、会话和帧号。默认 `--infer-timeout 900` 覆盖一帧的传输及推理，`--tool-timeout 30` 限制一次工具调用；超时或退出不会复位板卡，也不能中断已开始的板端推理。失败后保留输出目录和 `diagnostics.jsonl`，先由板级同事检查板端状态，再重新连接。接口、缓存交接和错误码见 [PCIe 说明](docs/PCIE.md)，日志解释见 [诊断说明](docs/DIAGNOSTICS.md)。

## 方案二：UART

链路：**host 采集/预处理 → UART 上传 → N550 RVV+AMU 推理 → UART 下载 → host 绘框/录像**。板端使用 `firmware/demo.elf`，不是 PCIe 固件。这是独立备选方案，不属于上面的免安装 PCIe 第一步：串口枚举/HELLO 探测只需 Python 标准库，当前 UART ping/模型后端需要 pyserial，图像与视频还需要 NumPy/OpenCV。UART 时钟、复位、引脚和接线须已就绪；host 串口设为 **115200、8N1，关闭硬件及软件流控**。

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
python3 yolo26_pc/serial_probe.py --probe "$PORT"
python3 yolo26_pc/uart_backend.py --port "$PORT" --ping 100
```

探测需确认固件 INFO 与预期输入/输出大小，100 次 ping 应全部通过。`firmware/selftest.elf` 只打印自测结果，不响应模型握手；若 probe 无响应，检查端口、占用、接线、时钟及板端启动状态，见 [串口探测](docs/SERIAL_PROBE.md)。

### 3. 单图数值与绘框

```bash
python3 yolo26_pc/uart_backend.py --port "$PORT" --image yolo26_pc/samples/bus.jpg --reference reference/bus-qemu-fp16.bin
python3 yolo26_pc/uart_backend.py --port "$PORT" --image yolo26_pc/samples/zidane.jpg --reference reference/zidane-qemu-fp16.bin
```

每次结果保存在 `yolo26_pc/outputs/board-uart/<时间>/`：`report.json`、`diagnostics.jsonl`、`annotated.png`、`candidates.npy`。检查 `status`、检测框、与参考候选框的差异和绘框效果。115200、8N1 下上传一帧约需 180 秒；UART 的 `--infer-timeout 900` 是每次 RUN 的等待上限，不包含上传时间。

### 4. 两帧视频与连续演示

```bash
python3 yolo26_pc/live_demo.py \
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
| [交付与验收状态](docs/VALIDATION.md) | 当前交付结论、本机检查及实板待验项目 |

只有需要修改或复验 C/汇编代码时，才在 Linux/WSL 构建机使用配套 ESWIN 工具链执行 `bash yolo26_riscv/build_board.sh`。工具链须支持 `xewmatrix1p0`，可通过 `TOOLCHAIN_PREFIX=/path/to/bin/riscv64-unknown-elf` 指定；构建结果位于 `yolo26_riscv/build/n550-board/`，其中 PCIe 固件同时生成 ELF 和可加载的 BIN，不会自动替换 `firmware/` 下的交付文件。样例视频来源为 [Pexels 3796613](https://www.pexels.com/video/people-walking-on-the-street-3796613/)，SHA-256：`fcd2af324d05ae09da2570ffd693da084f6afdb209b078b419c30741742b43c5`。
