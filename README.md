# N550 YOLO26 板端验证操作指南

在物理连接 S2C 的 Linux 实验室电脑（下称 **host**）运行：**host 采集/预处理 → PCIe 写 DDR → N550 RVV+AMU 推理 → PCIe 读结果 → host 绘框、显示和录像**。开发机通过已有远程桌面查看 host。**UART 只输出调试文字，已删除 UART 图片/结果传输方案。**

仓库提供源码、预编译固件、样例和数值基准。本机检查不代表实板通过；加载、缓存可见性、模型数值及吞吐仍须现场验收。以下命令均在 host 的仓库根目录执行。

## 1. 准备与查看调试串口

```bash
git clone https://github.com/risc-v-y/n550-yolo-demo.git
cd n550-yolo-demo
sha256sum -c firmware/SHA256SUMS
python3 --version
command -v pbcopy
command -v pbload
python3 yolo26_pc/serial_probe.py
```

固定图片验证仅需 Python 3 标准库（建议 3.10+）及现场已有的 PCIe 工具；无需 PyTorch、NumPy、OpenCV、pyserial、pip 或 venv。串口枚举不发送数据，需结合接线或 USB 序列号确认实际端口。

**先打开串口终端，再加载固件**，才能看到启动信息。例如另一终端执行：

```bash
minicom -D /dev/ttyUSB0 -b 115200
```

按实际接线替换端口，在 minicom/PuTTY 设置 **115200、8N1、关闭硬件及软件流控**。它可以与 PCIe 演示同时运行；UART 不响应 HELLO/ping，也不接收模型任务。端口排查见 [串口说明](docs/SERIAL_PROBE.md)。

host 时间因 S2C 许可回调时，已配置 SSH 公钥的设备改用 SSH 拉取和推送，避免 HTTPS 证书时间冲突：

```bash
git remote set-url origin git@github.com:risc-v-y/n550-yolo-demo.git
git pull --ff-only
```

`ssh -T git@github.com` 出现 `successfully authenticated` 和 `does not provide shell access` 是正常认证结果。SSH 解决 Git 传输；PCIe 工具仍需有效的 S2C 许可环境。

## 2. 复位、加载 DDR、释放复位

平台方须保证 DDR 可用、CPU `0x80000000–0x81FFFFFF` 的 32 MiB 独占、DCache 开启；复位释放不会清除已写入的 DDR，CPU 从 `0x80000000` 启动。平台初始化和 JTAG 配置仍由板级同事负责，见 [板端条件](docs/BOARD.md)。

将参数指向 **现场已有 `reset1.sh` 和 `reset.sh` 的目录**：

```bash
bash tools/load_board.sh --reset-dir /实际路径/s2c
```

脚本核对 BIN 哈希并获取 PCIe 互斥锁，然后执行 **`reset1.sh` 保持复位 → `pbcopy -d 完整BIN路径:0x0` → `reset.sh` 释放复位**。在复位脚本所在目录调用它们，保留相对路径依赖。每步检查退出码和失败文字，日志在 `yolo26_pc/outputs/deploy-*/load.log`。加载失败不会调用 `reset.sh`；释放失败报错，由平台方确认板卡状态。脚本成功只表示部署命令完成，启动与推理还需验证。

默认加载 `firmware/demo-pcie.bin`，包含程序、模型图及权重，不用单独上传 `.pt`。PCIe 偏移 `0x0` 对应 CPU `0x80000000`；`demo-pcie.elf` 供符号和 JTAG 调试，不能改名为 BIN 加载。

可先独立自测，再重新加载模型：

```bash
bash tools/load_board.sh --reset-dir /实际路径/s2c --selftest
# UART 应出现 CACHE/RVV/AMU TEST PASS
bash tools/load_board.sh --reset-dir /实际路径/s2c
```

独立自测还提供 `firmware/selftest.elf` 供已有调试器使用。模型启动自动执行相同自测，然后打印 READY。**复位、加载只在部署时执行一次，逐帧运行不复位、不重载权重。** 仅运行一个 PCIe 控制程序，其他工具不得并发修改本演示 DDR。工具不在 PATH 时给加载脚本加 `--pbcopy /绝对路径/pbcopy`，给 Python 命令加 `--pbcopy /绝对路径/pbcopy --pbload /绝对路径/pbload`。

## 3. 固定图片：免安装验证

```bash
# 读取接口并建立会话，不运行图像推理
python3 yolo26_pc/pcie_backend.py
python3 yolo26_pc/pcie_prepared.py --sample bus
python3 yolo26_pc/pcie_prepared.py --sample zidane
```

原始 JPEG、正式预处理流程生成的输入、坐标映射与哈希已交付，见 `reference/prepared/manifest.json`。输入是 RGB NCHW `[1,3,416,416]` 小端 FP32，2,076,672 字节；保留全部 334 节点和现有 RVV+AMU FP16 计算。返回 `[300,6]` FP32，7,200 字节，字段为输入图坐标下的 `x1,y1,x2,y2,score,class`。host 按 `score > 0.25` 过滤并还原坐标，不增加 NMS。

结果在 `yolo26_pc/outputs/board-pcie/<时间>/`：报告、日志、`candidates.bin` 和内嵌原图的 `annotated.svg`。用浏览器查看 SVG；检查报告的 `status=completed`、`reference.byte_identical`、类别顺序、坐标/分数差异和绘框效果。完成状态仅表示收发及结果检查完成，数值仍须验收，不设经验阈值掩盖差异。

## 4. 摄像头或视频演示

此步需要 host 已有兼容的 NumPy/OpenCV 和图形桌面，版本见 `yolo26_pc/requirements-board.txt`。缺失时由部署方准备匹配 host 的离线依赖，固定图片步骤不受影响。

```bash
python3 -c 'import numpy, cv2; print(numpy.__version__, cv2.__version__)'
python3 yolo26_pc/live_demo.py --backend pcie --camera 0 --save-frames
```

摄像头暂不可用时，每次重新运行完整视频流程：

```bash
python3 yolo26_pc/live_demo.py --backend pcie \
  --source yolo26_pc/samples/pexels-3796613.mp4 --max-results 2 --save-frames
```

确认两帧后去掉 `--max-results 2` 持续运行，Q/Esc 退出。无图形会话时加 `--no-display`。视频循环播放，推理取当前最新帧；窗口显示持续预览和最近一次完成推理的对应画面。

输出在 `yolo26_pc/outputs/live-demo/<时间>/`：`raw.mp4`、可演示的 `demo.mp4`、报告和日志；`--save-frames` 另存对应帧号的原图、检测图、候选框和 JSON。录像写入为 20 FPS，**实际推理帧率以报告为准**。

## 5. 运行状态与故障

UART 默认打印启动配置、AMU 初始化、自测、READY、每帧开始/结束、每 32 个节点的进度和错误。给 Python 命令加 `--board-trace`，打印每个节点开始/结束、形状及周期；`--check-intermediates` 增加中间 FP32 NaN/Inf 检查。详细打印会增加耗时。

UART 忙或发送超时后关闭文字输出，PCIe 模型继续运行；总线异常不属于可继续的超时。host 核对长度、CRC、会话和帧号，异常保存日志和 DDR 诊断。`--infer-timeout 900` 限制一帧，`--tool-timeout 30` 限制单次工具调用；退出或超时不复位板卡，也不能中断已开始的推理。

### 单命令读取 DDR

在 host 仓库根目录运行，Python 3.10 标准库即可。地址参数是 **CPU 地址**，脚本自动减去 `0x80000000` 得到 PCIe 偏移；读取范围限制在本演示预留的 `0x80000000–0x81FFFFFF`。

```bash
# 读取目标区的16字节，按16位十六进制显示，保存原始数据和日志
python3 tools/read_ddr.py 0x80003a40 16 --width 16

# 读取源区的16字节，按16位十六进制显示，用于与目标区比较
python3 tools/read_ddr.py 0x80003dc0 16 --width 16

# 检查程序入口的16字节，按32位十六进制显示
python3 tools/read_ddr.py 0x80000000 16 --width 32

# 检查起始4KiB是否全零；屏幕默认只预览前256字节
python3 tools/read_ddr.py 0x80000000 0x1000 --width 32
```

`0x80003a40`、`0x80003dc0` 是当前独立 `selftest` 固件的 FP16 目标/源区；模型固件或新编译固件须按对应 ELF 确认地址。结果保存在终端打印的 `yolo26_pc/outputs/ddr-read/run-随机编号/`：`001-ddr.bin` 是完整读回数据，`001-ddr.log` 是平台工具输出，`data.txt` 是显示的预览，`report.json` 和 `diagnostics.jsonl` 记录地址、长度、哈希、耗时及失败原因。

`All bytes zero: True` 仅在完整读取且所有字节为零时出现，覆盖整段请求范围，不仅是预览；读取失败返回非零退出码并保留日志/已有数据。脚本复用现有 PCIe 互斥锁、短临时文件名、错误文字及长度检查，只读 DDR。需要改变显示长度时加 `--max-print 字节数`。工具从 PATH 查找，找不到时尝试 `/rv/fpga/bin/pbload`；也可加 `--pbload /绝对路径/pbload`。许可工作目录默认是当前目录，修改时加 `--tool-cwd /许可工作目录`。

比较自测的不同阶段时先用已有 GDB 停住 CPU，再读取 DDR；CPU 运行中的读回不是原子快照。硬件断点每次只占用一个，`thbreak` 命中后自动删除；失败定位用断点自动停核，不依赖人工赶上 UART 输出。

### RVV 写回定位：缓存失效前后对照

仅用于当前交付的独立 `firmware/selftest.bin`。先按第 2 节加载它，用现有 GDB/OpenOCD 连接并停住启动 hart；GDB 必须支持 Python，并使用默认的 all-stop 模式。脚本核对本地 BIN/ELF 哈希及板上入口、各停点的代码；这些局部检查不代表完整 DDR 镜像校验。**不下载程序、不调用复位、不修改固件。**

在已连接的 GDB 中执行：

```gdb
# 查看现有断点；确认不再需要后删除，避免占用有限的硬件槽位
info breakpoints
delete breakpoints

# 将 GDB 工作目录切到 host 仓库根目录；按实际路径修改
cd /rv/fpga/yhb/n550-yolo-demo

# 注册定位命令，此时不会运行 CPU
source tools/rvv_store_probe.gdb

# 从入口重跑自测，依次暂停准备步骤，最终停在缓存失效前 0x80000b26
yolo-rvv-before
```

脚本依次经过 `0x800001dc → 0x800001ec → 0x80000a8e → 0x80000b26`，每次只设置一个临时硬件断点，命中或中断后清理；已有启用的断点/观察点会使脚本拒绝启动。重跑会清空 BSS 和旧自测记录，保留板卡硬件状态。

保持 GDB 停核，在另一个 host 终端的仓库根目录执行：

```bash
# 读取缓存失效前的三行 RVV 输出，共192字节，保存数据和日志
python3 tools/read_ddr.py 0x80003500 0xc0 --width 32
```

读取完成后回到 GDB：

```gdb
# 继续到 0x80000c2a：缓存失效及 fence 已执行，尚未比较结果
yolo-rvv-after
```

再次保持停核，在 host 终端执行同一条读取命令：

```bash
# 读取缓存失效后的同一范围；另存一份数据，用于与前一次对照
python3 tools/read_ddr.py 0x80003500 0xc0 --width 32
```

三个输出行从 `0x80003500 / 0x80003540 / 0x80003580` 开始，每行 16 个 FP32，其中前 5 个有效、后 11 个为保护值 `0xc49a4000`（−1234）。有效值应分别为 `[3,−33,15,28,27]`、`[11,36,26,−33,−8]`、`[26,−35,−5,11,−36]`；首元素的十六进制值为 `40400000 / 41300000 / 41d00000`。

提交两次读取的 `data.txt` 或终端输出及各自结果目录。失效前正确、失效后改变时，应进一步查缓存维护/覆盖；两次均错误时，继续查存储路径、地址及完成顺序。**断点暂停会改变时序，暂停后正确不等于连续运行通过。**

若预期断点未命中，CPU 可能已经进入退出的 WFI 循环，脚本不会自动超时；按 **Ctrl+C** 停核，脚本报告实际 PC、保留的错误字段并清理自己的断点，不自动重试。地址、固件或其他前置条件不符会明确中止。两次采集完成后可执行：

```gdb
# 查看没有遗留定位断点，然后继续到自测结论；退出 WFI 循环需 Ctrl+C
info breakpoints
continue
```

### UART 无输出：只读采集 DDR

保留当前板端状态，停止其他 PCIe 控制程序，minicom 可以继续打开。`--firmware` 必须对应**当前已加载的 BIN**：独立自测选 `selftest`，模型选 `demo-pcie`。

```bash
python3 tools/collect_board_diag.py --firmware selftest \
  --tool-cwd "$PWD" \
  --load-log yolo26_pc/outputs/deploy-实际目录/load.log
```

`--tool-cwd` 是 PCIe 工具的工作目录，须可写并能找到许可，沿用此前 `pbcopy` 加载成功时的目录和环境；它不是复位脚本目录。当前 host 在仓库根目录通过许可检查，示例使用 `$PWD`。工具不在 PATH 时加 `--pbload /绝对路径/pbload`。`--load-log` 填本次加载日志；暂时没有可省略。已有终端录像文本可加 `--uart-log /路径/uart.log`，工具不会打开或抢占串口。今后可用 `minicom -C /路径/uart.log -D /dev/ttyUSB2 -b 115200` 留存文字，端口仍按实际接线确认。

现场 `pbload` 在长输出路径下出现 `buffer overflow detected`；短文件名测试已执行到 `pbload end`，实际文件长度和内容仍须核验。采集工具在工作目录创建唯一的短文件名，传给 `pbload` 后将数据归档到结果目录并检查长度；失败或超时也归档已有数据，不把结果目录的长路径传给工具。

默认采集 3 次，间隔 1 秒，单次工具超时 30 秒；可用 `--samples`、`--interval`、`--tool-timeout` 调整。仅调用 `pbload`，不写 DDR、不提交推理任务、不复位。先校验 BIN/ELF 哈希及 DDR 中的 256 字节程序前缀，再从对应 ELF 解析 UART、自测、异常和 816 字节诊断结构地址；该前缀检查不代表全固件/权重核验。各变量是顺序读取的现场观察，不是原子快照。

终端打印结果目录 `yolo26_pc/outputs/host-diagnostics/run-随机编号/`，内含 `report.json`、`diagnostics.jsonl`、每次 `pbload` 的输出和原始 `.bin`，以及指定的加载/串口日志。`collected` 只表示采集完整，**不表示板端通过**；`partial`/`failed` 仍保留日志，应一并提交。重点看 `uart_status_name`、`test_stage_name`、诊断 `stage_name/error/trap_pc/trap_cause/trap_value`；启动太早失败或缓存不可见时，DDR 记录也可能未初始化，需结合平台调试器确认。

### 通过 SSH 上传诊断

将下面 `RUN_DIR` 换成采集工具打印的实际目录。使用独立临时 Git 目录，日志提交到 `host-diagnostics` 分支，保持当前源码工作区不变；此分支需要仓库写权限。

```bash
RUN_DIR="$(realpath yolo26_pc/outputs/host-diagnostics/run-实际编号)"
UPLOAD_DIR="$(mktemp -d)"
(
  set -e
  test -f "$RUN_DIR/report.json"
  git -C "$UPLOAD_DIR" init -q
  git -C "$UPLOAD_DIR" config user.name "N550 host diagnostics"
  git -C "$UPLOAD_DIR" config user.email "n550-host@users.noreply.github.com"
  git -C "$UPLOAD_DIR" remote add origin git@github.com:risc-v-y/n550-yolo-demo.git
  if git -C "$UPLOAD_DIR" ls-remote --exit-code --heads origin host-diagnostics; then
    git -C "$UPLOAD_DIR" fetch --depth=1 origin host-diagnostics
    git -C "$UPLOAD_DIR" checkout -b host-diagnostics FETCH_HEAD
  else
    test "$?" -eq 2
    git -C "$UPLOAD_DIR" checkout --orphan host-diagnostics
  fi
  cp -R -- "$RUN_DIR" "$UPLOAD_DIR/"
  git -C "$UPLOAD_DIR" add -- "$(basename "$RUN_DIR")"
  git -C "$UPLOAD_DIR" commit -m "Collect N550 board diagnostics $(basename "$RUN_DIR")"
  git -C "$UPLOAD_DIR" push origin HEAD:refs/heads/host-diagnostics
)
```

推送成功后提供 `run-随机编号` 即可定位。若提示远程分支已前进，保留采集目录，重新执行上传步骤；不使用强制推送。

| 文档 | 用途 |
|---|---|
| [BOARD.md](docs/BOARD.md) | DDR、UART、缓存和 AMU 平台条件 |
| [PCIE.md](docs/PCIE.md) | 邮箱、收发及缓存交接协议 |
| [DIAGNOSTICS.md](docs/DIAGNOSTICS.md) | 打印、错误码及调试器定位 |
| [VALIDATION.md](docs/VALIDATION.md) | 已检查范围和实板待验收事项 |

修改源码后，在现有 ESWIN 工具链的 Linux/WSL 构建机执行 `bash yolo26_riscv/build_board.sh`，可用 `TOOLCHAIN_PREFIX=/path/to/bin/riscv64-unknown-elf` 指定工具链。结果在 `yolo26_riscv/build/n550-board/`，不会自动替换交付目录 `firmware/`。
