# N550 YOLO26 板端验证操作指南

在物理连接 S2C 的 Linux 实验室电脑（下称 **host**）运行：**host 采集/预处理 → PCIe 写 DDR → N550 RVV+AMU 推理 → PCIe 读结果 → host 绘框、显示和录像**。开发机通过已有远程桌面查看 host。**UART 只输出调试文字，已删除 UART 图片/结果传输方案。**

仓库提供源码、预编译固件、样例和数值基准。本机检查不代表实板通过；加载、缓存可见性、模型数值及吞吐仍须现场验收。以下命令均在 host 的仓库根目录执行。

**当前先验证独立 CLP 通路。平台已确认 Memory port 上的 RVV 使用 DCache，旧自测/模型中的缓存交接需修正；`demo-pcie` 尚未迁移到 CLP，不能据新最小测试声称模型已跑通。**

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
minicom -D /dev/ttyUSB2 -b 115200
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
# 加载本轮独立 CLP 测试；按实际复位脚本目录修改
bash tools/load_board.sh --reset-dir /实际路径/s2c --clp-selftest
```

脚本核对 BIN 哈希并获取 PCIe 互斥锁，然后执行 **`reset1.sh` 保持复位 → `pbcopy -d 完整BIN路径:0x0` → `reset.sh` 释放复位**。在复位脚本所在目录调用它们，保留相对路径依赖。每步检查退出码和失败文字，日志在 `yolo26_pc/outputs/deploy-*/load.log`。加载失败不会调用 `reset.sh`；释放失败报错，由平台方确认板卡状态。脚本成功只表示部署命令完成，启动与推理还需验证。

默认加载 `firmware/demo-pcie.bin`，包含程序、模型图及权重，不用单独上传 `.pt`。PCIe 偏移 `0x0` 对应 CPU `0x80000000`；`demo-pcie.elf` 供符号和 JTAG 调试，不能改名为 BIN 加载。

### 当前先运行 CLP 最小测试

平台已确认独占 NoC DDR `0xF0000000–0xF0000FFF` 的 4 KiB。RVV 经 CLP 别名 `0x10000000–0x10000FFF` 无缓存访问，AMU 使用全局 DDR 地址，PCIe 偏移从 `0x70000000` 开始。代码、栈和诊断仍在原 32 MiB 区；不允许标量访问这段共享数据。

```bash
# 拉取已发布的 CLP 固件和脚本，更新本地工作区
git pull --ff-only

# 校验交付文件完整性
sha256sum -c firmware/SHA256SUMS

# 保持复位、加载 CLP 测试程序到 DDR 入口、释放复位；按现场路径修改目录
bash tools/load_board.sh --reset-dir /rv/fpga/bin --clp-selftest

# 只读采集诊断和完整 4 KiB 测试区，自动核对最终数据及保护值
python3 tools/collect_board_diag.py --firmware clp-selftest --samples 1 \
  --tool-cwd "$PWD"

# 可选：查看 RVV 输出的三行，每行16个 FP32；前5个有效，其余为保护值
python3 tools/read_ddr.py 0xf0000640 0xc0 --width 32 --clp-test
```

上一版 1×1 CLP 测试已由现场报告 UART 与 PCIe 验收均通过。本轮扩展版仍只占同一 4 KiB，连续 32 轮循环测试 `(M,N,K)=(3,5,17)、(1,1,1)、(2,3,32)、(3,5,31)`，每种形状执行 8 次。

AMU 计算 `C[m,n]=Σ A[m,k]×B[n,k]`，B 按转置后的 `[N,K]` 存储；RVV 对每行的 N 个结果加 `round+1`，得到 D。输入为 FP16 整数 −3 至 3，C/D 为 FP32，结果均可精确表示，逐位比较、不设误差阈值。N=5 在 VLEN=128 下分为4+1个元素，覆盖多次向量读写及尾段；K=17/31 检查奇数尾部，K=32 检查完整输入行。

| 缓冲区 | DDR 起点 | 有效数据起点 | 数据形状 |
|---|---|---|---|
| A | `0xF0000000` | `0xF0000040` | FP16 `[M,K]` |
| B | `0xF0000200` | `0xF0000240` | FP16 `[N,K]` |
| C | `0xF0000400` | `0xF0000440` | FP32 `[M,N]` |
| D | `0xF0000600` | `0xF0000640` | FP32 `[M,N]` |

各区512字节、行距64字节；首行、未使用行和列为保护值，`0xF0000800–0xF0000FFF` 为额外保护区。每轮检查四个完整缓冲区，最后检查剩余2 KiB。共享数据交接只用 `fence iorw,iorw` 和读取 `xmfflags` 等待，不对共享区执行 CBO；低地址诊断仍需 clean 后供 PCIe 读取。

UART 先打印四种 `CLP SHAPES`，最终应出现 `CLP RVV->AMU->RVV REUSE 32 ROUNDS PASS`。采集须显示 `CLP validation passed`、`completed_32_rounds=True`、`DDR_match=True`。最后一轮形状为 `[3,5,31]`，D 的三行有效结果为 `[60,−30,34,28,−27]`、`[28,34,−30,60,87]`、`[−32,56,95,71,−30]`；各行剩余11字为 `c49a4000`。输入半字保护值为 `3555`，其他保护字为 `35553555`。完整原始数据及差异保存在采集目录。扩展版仍待实板验证，不将上一版通过视为本版通过。

成功或失败后固件自动进入 `clp_probe_halt` 停车，保留诊断，不需要赶在 UART 打印时手动停核。如需用现有 GDB 连续重跑并停在最终结论处：

```gdb
# 在已连接并停核的启动 hart 上切到仓库根目录
cd /rv/fpga/yhb/n550-yolo-demo

# 加载本次 CLP 固件的调试符号，不下载程序
symbol-file firmware/clp-selftest.elf

# 查看旧断点；确认不用后删除，释放有限硬件槽位
info breakpoints
delete breakpoints

# 只设一个最终临时硬件断点，从入口连续重跑，中间不暂停
thbreak clp_probe_halt
set $pc = 0x80000000
continue

# 查看结论；通过时阶段为3、退出码为0、最后轮次为31
p board_test_stage
p board_error
p board_diag.frame

# 查看失败位置及原始数据；测试失败时保留这些字段
p board_diag
```

若停点未到达，按 Ctrl+C 停核并记录 PC/trap；不要不断增加断点。CLP 最小测试通过后再迁移模型数据布局及标量交接。现有 `selftest.bin/elf` 和 `demo-pcie.bin/elf` 保留作历史定位，未按新硬件信息修正；下面图片/视频入口留待模型修正后验收。

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

### 旧缓存定位入口

`tools/rvv_store_probe.gdb` 仅对应旧 `selftest.bin/elf`，不能用于 CLP 固件。原方案将缓存 RVV 写入后直接 invalidate，可能丢弃脏数据，RVV→AMU 也缺少 clean；之前取消隐式 RVV 栈拷贝并不证明故障修复。当前按第 2 节运行 CLP 测试，不再使用旧固定地址作新停点。

### UART 无输出：只读采集 DDR

保留当前板端状态，停止其他 PCIe 控制程序，minicom 可以继续打开。`--firmware` 必须对应**当前已加载的 BIN**：CLP 测试选 `clp-selftest`，旧独立自测选 `selftest`，模型选 `demo-pcie`。

```bash
# 只读采集当前 CLP 固件状态；加载日志路径替换为本次实际目录
python3 tools/collect_board_diag.py --firmware clp-selftest --samples 1 \
  --tool-cwd "$PWD" \
  --load-log yolo26_pc/outputs/deploy-实际目录/load.log
```

`--tool-cwd` 是 PCIe 工具的工作目录，须可写并能找到许可，沿用此前 `pbcopy` 加载成功时的目录和环境；它不是复位脚本目录。当前 host 在仓库根目录通过许可检查，示例使用 `$PWD`。工具不在 PATH 时加 `--pbload /绝对路径/pbload`。`--load-log` 填本次加载日志；暂时没有可省略。已有终端录像文本可加 `--uart-log /路径/uart.log`，工具不会打开或抢占串口。今后可用 `minicom -C /路径/uart.log -D /dev/ttyUSB2 -b 115200` 留存文字，端口仍按实际接线确认。

现场 `pbload` 在长输出路径下出现 `buffer overflow detected`；短文件名测试已执行到 `pbload end`，实际文件长度和内容仍须核验。采集工具在工作目录创建唯一的短文件名，传给 `pbload` 后将数据归档到结果目录并检查长度；失败或超时也归档已有数据，不把结果目录的长路径传给工具。

默认采集 3 次，间隔 1 秒，单次工具超时 30 秒；可用 `--samples`、`--interval`、`--tool-timeout` 调整。仅调用 `pbload`，不写 DDR、不提交推理任务、不复位。先校验 BIN/ELF 哈希及 DDR 中的 256 字节程序前缀，再从对应 ELF 解析 UART、自测、异常和 816 字节诊断结构地址；该前缀检查不代表全固件/权重核验。各变量是顺序读取的现场观察，不是原子快照。

终端打印结果目录 `yolo26_pc/outputs/host-diagnostics/run-随机编号/`，内含 `report.json`、`diagnostics.jsonl`、每次 `pbload` 的输出和原始 `.bin`，以及指定的加载/串口日志。`collected` 只表示采集完整，**不表示板端通过**；`partial`/`failed` 仍保留日志，应一并提交。CLP 固件还会归档完整测试区，单独报告 `clp_result.status`；只有 `passed` 才表示本最小测试及 PCIe 最终数据一致，失败或未完成返回非零退出码。重点看 `uart_status_name`、`test_stage_name`、诊断 `stage_name/error/trap_pc/trap_cause/trap_value`；启动太早失败或缓存不可见时，DDR 记录也可能未初始化，需结合平台调试器确认。

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

CLP 独立固件的构建命令为 `bash yolo26_riscv/build_clp_selftest.sh`，输出在 `yolo26_riscv/build/clp-selftest/`；QEMU 回归命令为 `python3 yolo26_riscv/board/tests/test_clp.py`。均在配置现有 ESWIN 工具链/QEMU 的 Linux/WSL 构建机执行；QEMU 使用统一 RAM，只检查运算与故障报告，不验证真实 CLP 路由。
