# N550 YOLO26：板端验证入口

**软件同事从本页开始，依次完成自测、串口、单图和视频验证。** 配套固件为 [v0.2.0-host-diagnostics](https://github.com/risc-v-y/n550-yolo-demo/releases/tag/v0.2.0-host-diagnostics)。软件构建和本机测试已通过，真实开发板仍待验收。

**host** 指物理连接 S2C 开发板的 **Linux 实验室电脑，已有 Python 3.10**。当前链路：

**host 读取视频、预处理 → UART → N550 RVV+AMU 推理 → UART → host 后处理、绘框、保存 → 开发机远程查看 host 桌面。**

以下命令均在 **host 的仓库根目录**执行。Windows 开发机只用于远程查看，无需为本流程安装 Python。先用附带视频验证；开发机摄像头和跨机器数据转发尚未接入。

## 1. 准备材料和板端加载

将仓库源码下载或复制到 host，并从上述 Release 下载配套文件：

| 材料 | 用途 |
|---|---|
| `selftest.elf` | 已编译的 N550 自测程序，检查 UART、缓存、RVV 和 AMU |
| `demo.elf` | 已编译的模型程序，已包含权重、偏置等模型常量 |
| `board-build-info.zip` | 内存布局、符号表、反汇编和构建测试记录 |
| `SHA256SUMS` | 校验下载的固件及构建记录 |
| 仓库中的 `yolo26_pc/`、`reference/` | host 程序、依赖清单、测试图片/视频及对照结果 |

软件同事负责通过现有 J-Link/PCIe 工具加载和启动固件。**按 ELF 加载段指定的地址写入 DDR，再从 `_start` 运行；不能把整个 ELF 文件当作裸数据直接写到某个地址。** 若加载工具只接受裸二进制，由同事处理转换及加载地址。

程序为 demo 预留 `0x80000000–0x81FFFFFF` 共 32 MiB DDR。启动前须满足 [板端交接条件](docs/BOARD.md)：DDR 可用、该区域独占、DCache 已开启、UART 时钟/复位/引脚可用，以及下载后的代码和数据缓存一致性。固件不包含这些板级初始化。

构建时 `model/constants.bin` 已嵌入 `demo.elf`，加载 ELF 时一并进入 DDR。运行时通过 `model_weights` 符号加张量偏移访问，不需要另传 `.pt` 或每帧重新加载权重。直接上板可使用现成 ELF，无需先编译。

## 2. 准备 host Python 环境并查找串口

```bash
# 确认 host 的 python3 是已知的 Python 3.10
python3 --version
# 创建独立环境；安装图像处理和串口依赖，无需 PyTorch
python3 -m venv .venv-host
.venv-host/bin/python -m pip install -r yolo26_pc/requirements-board.txt
# 仅列举设备、USB 标识、权限和可见占用，不打开串口或发送数据
.venv-host/bin/python yolo26_pc/serial_probe.py
```

若 host 缺少 venv/pip 或依赖安装条件，请同事补齐环境。串口探测本身只依赖标准库，也可直接执行 `python3 yolo26_pc/serial_probe.py`。

根据 USB 型号、序列号或接线确认候选设备。下文用变量保存实际设备名；**将示例替换为探测到的端口**，也可使用 `/dev/serial/by-id/...` 稳定路径：

```bash
PORT=/dev/ttyUSB0
```

无占用记录不证明设备空闲；权限与占用问题见 [串口查找说明](docs/SERIAL_PROBE.md)。

## 3. 运行 selftest.elf

先在 host 使用 minicom/PuTTY 打开候选串口，设置 **115200、8N1，关闭硬件和软件流控**，再让同事加载并启动 `selftest.elf`。

期望依次看到：

```text
UART OK
CACHE/RVV/AMU TEST PASS
```

`UART OK` 表示该端口收到了板端输出，后一句表示自测通过。程序不会列出 host 端口名，也不知道 Linux 将其命名为 ttyUSB0 还是其他设备。输出只在启动时发送；如果打开终端太晚，请同事重新启动自测。无输出时需检查候选端口、板端启动状态和接线。

## 4. 运行 demo.elf，验证通信与单图

关闭占用该串口的 minicom/PuTTY，由同事重新加载并启动 **`demo.elf`**。两个 ELF 分别运行，不需要同时驻留。

```bash
# 验证与模型固件的 HELLO/INFO 握手；不执行推理
.venv-host/bin/python yolo26_pc/serial_probe.py --probe "$PORT"
# 100 次、每次 1024 字节的串口回显校验；不执行推理
.venv-host/bin/python yolo26_pc/uart_backend.py --port "$PORT" --ping 100
# 发送图片、板端推理、host 绘框，并记录与历史 QEMU 候选框的差异
.venv-host/bin/python yolo26_pc/uart_backend.py --port "$PORT" --image yolo26_pc/samples/bus.jpg --reference reference/bus-qemu-fp16.bin
.venv-host/bin/python yolo26_pc/uart_backend.py --port "$PORT" --image yolo26_pc/samples/zidane.jpg --reference reference/zidane-qemu-fp16.bin
```

`selftest.elf` 不响应 HELLO，必须切换为 `demo.elf` 后再执行这些命令。模型固件传输的是二进制结果/诊断协议，host 程序负责解码；不能依靠串口终端直接显示检测图。

每次运行自动创建 `yolo26_pc/outputs/board-uart/<时间>/`，保存日志和报告；单图成功时另有 `annotated.png`、`candidates.npy`。如指定 `--output`，目录必须尚不存在。

请配套更新源码与 v0.2.0 固件。旧 v0.1.0 固件不支持新增诊断命令，新 host 程序可能收到协议错误 2；握手通过也不代表完整模型已经通过。

## 5. 运行视频并保存演示结果

在 **能显示窗口的 host 桌面会话**执行：

```bash
.venv-host/bin/python yolo26_pc/live_demo.py \
  --backend uart --serial-port "$PORT" \
  --source yolo26_pc/samples/pexels-3796613.mp4 \
  --max-results 2 --save-frames
```

首次处理两帧后自动停止；持续演示时去掉 `--max-results 2`，按 Q/Esc 退出。左侧持续预览，右侧显示最新完成推理的那一帧及检测框。普通 SSH 终端未必有图形显示环境，可加 `--no-display` 先验证文件输出。

输出目录为 `yolo26_pc/outputs/live-demo/<时间>/`：

| 文件 | 内容 |
|---|---|
| `demo.mp4` / `raw.mp4` | 预览与检测结果组成的演示录像 / 原始预览录像 |
| `report.json` / `diagnostics.jsonl` | 帧结果与耗时汇总 / 板端诊断和 host 异常堆栈 |
| `source-*.png` / `detected-*.png` | 加 `--save-frames` 后保存的推理原图 / 检测图，同编号对应 |
| `candidates-*.npy` / `result-*.json` | 加 `--save-frames` 后保存的候选框和逐帧结果 |

定位算子问题时，在上述命令末尾加 `--board-trace`；检查中间 NaN/Inf 时加 `--check-intermediates`。两者均增加运行开销，具体含义和错误码见 [诊断说明](docs/DIAGNOSTICS.md)。

### 当前格式与耗时

固定 416×416、batch=1、COCO80、334 节点。host 将图片等比例缩放补边、转 RGB、除以 255，生成 `[1,3,416,416]` FP32 输入；模型使用 AMU 的计算块再转换为 FP16。FP32 是现有模型接口，不是 UART 的要求。板端返回 `[300,6]` FP32 候选框，host 按 `score > 0.25` 过滤、还原坐标和绘框，不额外增加 NMS。

一帧输入为 2,076,672 字节，115200、8N1 下仅数据传输约 180 秒，另有分包和推理耗时。当前验收功能，不要求达到 5 FPS。`--infer-timeout 900` 是 RUN 命令每次等待上限，不包含上传时间；host 超时不代表板端计算已停止。

## 按需查阅

| 文档 | 什么时候看 |
|---|---|
| [板端条件](docs/BOARD.md) | 加载前确认地址、缓存、UART 和启动状态 |
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

输出在 `yolo26_riscv/build/n550-board/`。必须使用支持 `xewmatrix1p0` 的配套工具链。可通过 `TOOLCHAIN_PREFIX=/path/to/bin/riscv64-unknown-elf` 指定已有安装。模型常量和图位于 `model/`，更换模型需同步重新生成图、常量及数值参考。

附带视频：[Pexels 3796613](https://www.pexels.com/video/people-walking-on-the-street-3796613/)，SHA256 为 `fcd2af324d05ae09da2570ffd693da084f6afdb209b078b419c30741742b43c5`。QEMU 历史结果仅作数值参考，不代表实板已经通过。
