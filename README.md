# N550 YOLO26：板端验证入口

**软件同事从本页开始，当前优先使用 PCIe 完成单图、连续帧和视频演示；UART 验证入口保留在第6节。** 程序加载、启动一次后，host 自动传图、等待推理、读结果和显示，无需逐帧人工操作。

**版本注意：旧 [v0.2.0 Release](https://github.com/risc-v-y/n550-yolo-demo/releases/tag/v0.2.0-host-diagnostics) 不包含 PCIe 固件和新增32轮同步自测。** PCIe等新增文件由跨设备补充包提供；在另一台设备克隆仓库后，须将补充包解压到仓库根目录，再按文末步骤构建或使用包内配套ELF。

当前验证：PCIe 11项本机联调、UART 7项回归及固件构建通过。PCIe联调使用模拟DDR/工具和测试替身推理；**真实板卡传输、缓存可见性和模型数值仍待现场验收**。

**host** 指物理连接 S2C 开发板的 **Linux 实验室电脑，已有 Python 3.10**。当前链路：

**host 读取视频、预处理 → PCIe → N550 RVV+AMU 推理 → PCIe → host 后处理、绘框、保存 → 开发机远程查看 host 桌面。**

以下命令均在 **host 的仓库根目录**执行。Windows 开发机只用于远程查看，无需为本流程安装 Python。先用附带视频验证；开发机摄像头和跨机器数据转发尚未接入。

## 1. 准备材料和板端加载

将本次源码和配套固件复制到 host。构建输出目录为 `yolo26_riscv/build/n550-board/`：

| 材料 | 用途 |
|---|---|
| `selftest.elf` | 已编译的 N550 自测程序，检查 UART、缓存、RVV 和 AMU |
| **`demo-pcie.elf`** | **当前PCIe演示使用**；模型、权重、自测及DDR收发接口 |
| `demo.elf` | UART模型固件；按第6节使用 |
| `pcie-symbols.txt` / `demo-pcie.disasm` / `demo-pcie.map` | PCIe固件的符号、反汇编和内存布局 |
| `sha256.txt` | 本次构建的固件与模型资产哈希 |
| 仓库中的 `yolo26_pc/`、`reference/` | host 程序、依赖清单、测试图片/视频及对照结果 |

软件同事负责通过现有 J-Link/PCIe 工具加载和启动固件。**按 ELF 加载段指定的地址写入 DDR，再从 `_start` 运行；不能把整个 ELF 文件当作裸数据直接写到某个地址。** 若加载工具只接受裸二进制，由同事处理转换及加载地址。

程序预留 `0x80000000–0x81FFFFFF` 共32 MiB DDR。启动前须满足 [板端交接条件](docs/BOARD.md)：DDR可用且该区域独占、DCache已开启、下载后的代码和数据缓存一致。PCIe需host驱动与 `pbcopy/pbload` 可用，并允许CPU运行时访问DDR；UART固件及独立自测另外要求UART时钟/复位/引脚可用。固件不包含这些板级初始化。

两种模型ELF均已嵌入 `model/constants.bin`，加载时权重一并进入DDR，运行时通过 `model_weights` 加张量偏移访问，无需另传 `.pt` 或逐帧重新加载。不要使用另一个仓库会复位、重写镜像和权重的批处理脚本驱动逐帧演示。

## 2. 准备 host Python 环境

```bash
# 确认 host 的 python3 是已知的 Python 3.10
python3 --version
# 创建独立环境；安装图像处理和串口依赖，无需 PyTorch
python3 -m venv .venv-host
.venv-host/bin/python -m pip install -r yolo26_pc/requirements-board.txt
```

若host缺少venv/pip或依赖安装条件，请同事补齐环境。PCIe命令示例假设 `pbcopy`、`pbload` 位于PATH；否则在Python命令末尾加 `--pbcopy /绝对路径/pbcopy --pbload /绝对路径/pbload`。调用账户须具有设备访问权限。

## 3. 独立自测（可选，需UART）

两个模型固件启动时也会运行相同自测。需要独立定位UART、缓存或AMU问题时，再单独加载 `selftest.elf`：先执行 `python3 yolo26_pc/serial_probe.py` 列举串口，根据设备信息和接线确认端口。

先在 host 使用 minicom/PuTTY 打开候选串口，设置 **115200、8N1，关闭硬件和软件流控**，再让同事加载并启动 `selftest.elf`。

期望依次看到：

```text
UART OK
RVV->AMU->RVV REUSE 32 ROUNDS PASS
CACHE/RVV/AMU TEST PASS
```

`UART OK` 表示该端口收到了板端输出，后两句表示新增32轮用例及全部自测通过。失败时打印轮次、缓冲区、元素下标和实际/期望值，解释见 [诊断说明](docs/DIAGNOSTICS.md)。程序不会列出 host 端口名，也不知道 Linux 将其命名为 ttyUSB0 还是其他设备。输出只在启动时发送；如果打开终端太晚，请同事重新启动自测。无输出时需检查候选端口、板端启动状态和接线。

## 4. PCIe：加载模型，验证通信与单图

由同事加载并启动 **`demo-pcie.elf`**，保持程序运行。此固件不初始化UART，状态与诊断通过DDR读取。三个ELF分别运行，不同时驻留。

```bash
# 读取接口并建立会话，不执行图像推理
.venv-host/bin/python yolo26_pc/pcie_backend.py
# 自动上传、推理、下载、绘框，并记录与历史QEMU结果的差异
.venv-host/bin/python yolo26_pc/pcie_backend.py --image yolo26_pc/samples/bus.jpg --reference reference/bus-qemu-fp16.bin
.venv-host/bin/python yolo26_pc/pcie_backend.py --image yolo26_pc/samples/zidane.jpg --reference reference/zidane-qemu-fp16.bin
```

输出在 `yolo26_pc/outputs/board-pcie/<时间>/`：`source.png`、`annotated.png`、`candidates.npy`、`report.json`、`diagnostics.jsonl`。指定 `--output` 时目录必须尚不存在。检查绘框效果及报告中的类别、置信度、坐标差异；握手通过不代表模型已经通过。

host检查工具退出状态、读回长度、CRC、会话和帧号；仅允许一个控制程序。DDR地址、缓存交接和故障处理详见 [PCIe说明](docs/PCIE.md)。

## 5. 运行视频并保存演示结果

在 **能显示窗口的 host 桌面会话**执行：

```bash
.venv-host/bin/python yolo26_pc/live_demo.py \
  --backend pcie \
  --source yolo26_pc/samples/pexels-3796613.mp4 \
  --max-results 2 --save-frames
```

首次处理两帧后自动停止；持续演示时去掉 `--max-results 2`，按Q/Esc退出。左侧持续预览，右侧每完成一帧推理才更新。程序顺序收发，并选取当前最新采集帧；不保证处理视频素材的每一帧。host本地摄像头可用 `--camera 0` 替换 `--source` 参数。普通SSH终端未必有图形显示环境，可加 `--no-display` 先验证文件输出。

输出目录为 `yolo26_pc/outputs/live-demo/<时间>/`：

| 文件 | 内容 |
|---|---|
| `demo.mp4` / `raw.mp4` | 预览与检测结果组成的演示录像 / 原始预览录像 |
| `report.json` / `diagnostics.jsonl` | 帧结果与耗时汇总 / 板端诊断和 host 异常堆栈 |
| `source-*.png` / `detected-*.png` | 加 `--save-frames` 后保存的推理原图 / 检测图，同编号对应 |
| `candidates-*.npy` / `result-*.json` | 加 `--save-frames` 后保存的候选框和逐帧结果 |

检查中间NaN/Inf可加 `--check-intermediates`；`--board-trace` 仅用于UART。PCIe在完成/失败时读取板端诊断快照，并记录工具返回值与传输耗时，见 [诊断说明](docs/DIAGNOSTICS.md)。

### 当前格式与耗时

固定 416×416、batch=1、COCO80、334 节点。host 将图片等比例缩放补边、转 RGB、除以 255，生成 `[1,3,416,416]` FP32 输入；模型使用 AMU 的计算块再转换为 FP16。FP32 是现有模型接口，不是 UART 的要求。板端返回 `[300,6]` FP32 候选框，host 按 `score > 0.25` 过滤、还原坐标和绘框，不额外增加 NMS。

一帧输入2,076,672字节、结果7,200字节。PCIe实际速度待现场测量；报告中的结果帧率表示处理速度，录像自身的20 FPS不代表推理速度。PCIe的 `--infer-timeout 900` 覆盖整帧传输与推理，`--tool-timeout 30` 限制单次工具调用。超时或退出不会复位板卡，也不代表板端推理已停止。

## 6. 保留的UART验证入口

由同事加载 **`demo.elf`**。关闭占用串口的minicom/PuTTY，按探测结果替换端口：

```bash
python3 yolo26_pc/serial_probe.py
PORT=/dev/ttyUSB0
.venv-host/bin/python yolo26_pc/serial_probe.py --probe "$PORT"
.venv-host/bin/python yolo26_pc/uart_backend.py --port "$PORT" --ping 100
.venv-host/bin/python yolo26_pc/uart_backend.py --port "$PORT" --image yolo26_pc/samples/bus.jpg --reference reference/bus-qemu-fp16.bin
.venv-host/bin/python yolo26_pc/uart_backend.py --port "$PORT" --image yolo26_pc/samples/zidane.jpg --reference reference/zidane-qemu-fp16.bin
```

视频复用第5节命令，将 `--backend pcie` 替换为 `--backend uart --serial-port "$PORT"`。单图输出在 `yolo26_pc/outputs/board-uart/<时间>/`，视频输出与第5节相同。

115200、8N1下仅上传一帧约180秒；UART的 `--infer-timeout 900` 是每次RUN等待上限，不包含上传。串口数据为二进制协议，由host程序解码，终端不能直接显示检测图。`selftest.elf` 不响应模型握手；旧v0.1.0模型固件不支持新增诊断命令。

## 按需查阅

| 文档 | 什么时候看 |
|---|---|
| [Codex上下文交接](docs/HANDOFF.md) | 换设备或新会话时恢复已确认决策、工作状态和未提交修改 |
| [板端条件](docs/BOARD.md) | 加载前确认地址、缓存、UART 和启动状态 |
| [PCIe说明](docs/PCIE.md) | DDR接口布局、缓存交接、工具约定和失败处理 |
| [串口探测](docs/SERIAL_PROBE.md) | 不清楚端口、权限或占用情况 |
| [诊断说明](docs/DIAGNOSTICS.md) | 失败、卡住、逐节点日志及调试器读取诊断区 |
| [UART 协议](docs/UART_PROTOCOL.md) | 修改收发程序或检查协议细节 |
| [验证记录](docs/VALIDATION.md) | 区分已经完成的本机检查与待完成的实板验收 |

## 可选：重新编译固件

需要改 C/汇编代码时，从 [v0.1.0 Release](https://github.com/risc-v-y/n550-yolo-demo/releases/tag/v0.1.0-board-bringup) 获取配套 Linux x86-64 ESWIN 工具链（v0.2.0 沿用），在具备 Python3 的 Linux/WSL 构建机、仓库根目录执行：

```bash
mkdir -p toolchain
tar -xzf /path/to/eswin-riscv-toolchain-linux-x86_64.tar.gz -C toolchain
bash yolo26_riscv/build_board.sh
```

输出在 `yolo26_riscv/build/n550-board/`，包括UART模型 `demo.elf`、PCIe模型 `demo-pcie.elf` 和独立自测 `selftest.elf`。必须使用支持 `xewmatrix1p0` 的配套工具链。可通过 `TOOLCHAIN_PREFIX=/path/to/bin/riscv64-unknown-elf` 指定已有安装。模型常量和图位于 `model/`，更换模型需同步重新生成图、常量及数值参考。

附带视频：[Pexels 3796613](https://www.pexels.com/video/people-walking-on-the-street-3796613/)，SHA256 为 `fcd2af324d05ae09da2570ffd693da084f6afdb209b078b419c30741742b43c5`。QEMU 历史结果仅作数值参考，不代表实板已经通过。
