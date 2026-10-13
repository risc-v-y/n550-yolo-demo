# N550 YOLO26 板端验证操作指南

在物理连接 S2C 的 Linux 实验室电脑（下称 **host**）运行：**host 采集/预处理 → PCIe 写 DDR → N550 RVV+AMU 推理 → PCIe 读结果 → host 绘框、显示和录像**。开发机通过已有远程桌面查看 host。**UART 只输出调试文字，已删除 UART 图片/结果传输方案。**

仓库提供源码、预编译固件、样例和数值基准。本机检查不代表实板通过；加载、缓存可见性、模型数值及吞吐仍须现场验收。以下命令均在 host 的仓库根目录执行。

**当前交付为 CLP 模型版，增加自动逐节点采集。实板 bus 已完成334节点，两次结果逐字节一致，但与QEMU基准存在差异，原因尚未确定。先按第3.1节自动定位，再继续精度验收。**

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

固定图片验证、逐节点自动采集及截图报告支持 Python 3.8+ 标准库及现场已有的 PCIe 工具；无需 PyTorch、NumPy、OpenCV、pyserial、pip 或 venv。host 使用 Python 3.8.10 时可直接执行文中的 `python3` 命令。串口枚举不发送数据，需结合接线或 USB 序列号确认实际端口。

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

平台方须保证 DDR 可用，程序/权重/栈所在 `0x80000000–0x81FFFFFF` 的32 MiB，以及模型数据所在 `0xF0000000–0xF0FFFFFF` 的16 MiB均独占，DCache开启；复位释放不会清除已写入的 DDR，CPU 从 `0x80000000` 启动。平台初始化和 JTAG 配置仍由板级同事负责，见 [板端条件](docs/BOARD.md)。

将参数指向 **现场已有 `reset1.sh` 和 `reset.sh` 的目录**：

```bash
# 复位、加载正式模型固件、释放复位；按现场复位脚本和工具路径修改
bash tools/load_board.sh --reset-dir /rv/fpga/bin --pbcopy /rv/fpga/bin/pbcopy
```

脚本核对 BIN 哈希并获取 PCIe 互斥锁，然后执行 **`reset1.sh` 保持复位 → `pbcopy -d 完整BIN路径:0x0` → `reset.sh` 释放复位**。在复位脚本所在目录调用它们，保留相对路径依赖。每步检查退出码和失败文字，日志在 `yolo26_pc/outputs/deploy-*/load.log`。加载失败不会调用 `reset.sh`；释放失败报错，由平台方确认板卡状态。脚本成功只表示部署命令完成，启动与推理还需验证。

默认加载 `firmware/demo-pcie.bin`，包含程序、模型图及权重，不用单独上传 `.pt`。PCIe 偏移 `0x0` 对应 CPU `0x80000000`；`demo-pcie.elf` 供符号和 JTAG 调试，不能改名为 BIN 加载。

模型数据实际占8,143,104字节（约7.77 MiB）。高地址数据为 NOLOAD，不写入 BIN，由启动代码通过 RVV CLP 清零；BIN 约9.31 MiB，不会包含两个地址区之间的大空洞。标量使用全局 DDR 地址，参与交接时仍做缓存维护；权重及栈上索引数组使用正常缓存路径。

UART 应依次出现 BOOT、CLP 数据地址、四种 SHAPES、`32 ROUNDS PASS`、READY。模型启动复用已在现场通过的32轮CLP自测；自测失败会发布PCIe ERROR并保留诊断。看到READY后再执行第3节。

### 独立 CLP 自测：按需复验

提交 `9e253cf` 的四种形状、32轮自测已由现场报告全部成功。独立 `clp-selftest.bin/elf` 保持该已验证版本；旧 `selftest.bin/elf` 仅供历史缓存定位，不用于模型启动。

```bash
# 可选：加载独立4 KiB自测，会替换当前运行的模型
bash tools/load_board.sh --reset-dir /rv/fpga/bin \
  --pbcopy /rv/fpga/bin/pbcopy --clp-selftest

# 读取自测诊断及完整4 KiB，逐字核对；不是模型图像推理
python3 tools/collect_board_diag.py --firmware clp-selftest --samples 1 \
  --pbload /rv/fpga/bin/pbload --tool-cwd "$PWD"

# 复验后重新加载正式模型，等待READY再提交图像
bash tools/load_board.sh --reset-dir /rv/fpga/bin --pbcopy /rv/fpga/bin/pbcopy
```

独立自测的数据布局、错误字段和最终 `clp_probe_halt` 停点见 [诊断说明](docs/DIAGNOSTICS.md)。每次只需一个临时硬件断点，不依赖人工追赶UART。它只验证四种矩阵形状，完整模型仍须实板单图及连续帧验收。

## 3. 固定图片：免安装验证

```bash
# 查看固件描述并建立PCIe会话，不运行图像推理
python3 yolo26_pc/pcie_backend.py --pbcopy /rv/fpga/bin/pbcopy --pbload /rv/fpga/bin/pbload

# 上传bus输入、推理并保存原始结果、参考差异和绘框图
python3 yolo26_pc/pcie_prepared.py --sample bus --pbcopy /rv/fpga/bin/pbcopy --pbload /rv/fpga/bin/pbload --board-trace --check-intermediates

# 同样验证zidane；程序和权重保持加载，不逐帧复位
python3 yolo26_pc/pcie_prepared.py --sample zidane --pbcopy /rv/fpga/bin/pbcopy --pbload /rv/fpga/bin/pbload --board-trace --check-intermediates
```

原始 JPEG、正式预处理流程生成的输入、坐标映射与哈希已交付，见 `reference/prepared/manifest.json`。输入是 RGB NCHW `[1,3,416,416]` 小端 FP32，2,076,672 字节；保留全部 334 节点和现有 RVV+AMU FP16 计算。返回 `[300,6]` FP32，7,200 字节，字段为输入图坐标下的 `x1,y1,x2,y2,score,class`。host 按 `score > 0.25` 过滤并还原坐标，不增加 NMS。

结果在 `yolo26_pc/outputs/board-pcie/<时间>/`：报告、日志、`candidates.bin` 和内嵌原图的 `annotated.svg`。用浏览器查看 SVG；检查报告的 `status=completed`、`reference.byte_identical`、类别顺序、坐标/分数差异和绘框效果。完成状态仅表示收发及结果检查完成，数值仍须验收，不设经验阈值掩盖差异。

首次验收保留上述逐节点打印和中间值检查。旧版 `5af2816` 在节点305存在常量输出检查误报，已修正；更新后须重新加载本包的 `demo-pcie.bin`，仅拉取源码不会替换板上程序。原因见 [诊断说明](docs/DIAGNOSTICS.md)。

### 3.1 自动采集并在 host 上比较

仍只需 Python 3.10 标准库及已有PCIe工具，无需GDB断点。停止其他PCIe控制程序，让CPU保持运行；minicom可继续查看日志。

```bash
# 更新源码、采集基准及固件；fetch不会更新工作区，须执行pull
git pull --ff-only

# 重新加载新增采集接口的固件，等待UART出现READY
bash tools/load_board.sh --reset-dir /rv/fpga/bin --pbcopy /rv/fpga/bin/pbcopy

# 自动采集bus全部节点、逐张量比较、绘框，最后再跑一次普通模式检查采集影响
python3 yolo26_pc/node_capture.py --sample bus \
  --pbcopy /rv/fpga/bin/pbcopy --pbload /rv/fpga/bin/pbload
```

板端每完成一个节点自动暂停；host读取并核对CRC、比较完整张量，再自动通知板端继续。334节点全部处理完才结束，首个差异节点还会读取实际输入并尝试数值重算。无需人工赶UART或反复设置断点。采集增加CRC扫描、读回和握手开销，耗时不代表正常推理速度；最后的普通模式结果用于检查采集是否改变结果。

采集结束自动显示第一页摘要。**优先回传以下三页的终端截图，无需传文件**；每页最多22行、80列，使用英文短标签以便终端显示和截图识别。三张图的 `ID` 应一致。

```bash
# 第一页：采集完整性、总体差异、普通推理重复结果
python3 yolo26_pc/node_report.py --page 1

# 第二页：首个差异节点、输入输出形状、实际值及参考原始位
python3 yolo26_pc/node_report.py --page 2

# 第三页：候选累加规则的数值重算与匹配结果
python3 yolo26_pc/node_report.py --page 3
```

命令默认读取最近一次结束的采集报告，按完成顺序记录，不受host回调日期影响。失败时第二、三页优先显示错误与最后节点。已有旧版采集结果也可直接查看：`python3 yolo26_pc/node_report.py 结果目录 --page 1`，无需重跑；第二、三页替换页码即可。这些查看命令只读host文件，不访问板子。

结果默认在 `yolo26_pc/outputs/node-capture/<时间>/`；完整张量、实际输入、参考和工具日志全部保留host，后续缺少某项证据可再提取。`return.zip`保留为可选的小包。`completed`表示采集完成，不能当作数值精度通过；规则重算匹配也不能独立证明FPGA采用该舍入规则。

`reference/nodes/` 为一次性下载的bus/zidane无损QEMU基准，合计约136 MiB，分片不超过24 MiB，清单绑定冻结模型和输入哈希。每张图采集的完整输出约90 MiB。另一张图把 `--sample bus` 改为 `--sample zidane`。每节点默认超时300秒、单次工具30秒，可用 `--node-timeout`、`--tool-timeout` 调整。

Ctrl-C或采集失败会尽力发送ABORT并生成已有结果的小包。进程被强杀导致板端停在 `state=5` 时，在同一命令后加 `--recover-paused`，自动终止旧采集并重新开始；无须重新加载固件。若核心被GDB暂停或PCIe工具不可用，先恢复现场通路。详细字段与限制见 [诊断说明](docs/DIAGNOSTICS.md)。

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

host 重复打印 `pbload: exit 0` 只表示读回工具正常退出，不表示模型完成。PCIe `state=1/2/3/4/5` 分别为READY/BUSY/DONE/ERROR/节点采集暂停；GDB若停住CPU，须先执行 `continue` 才能处理请求。

UART 忙或发送超时后关闭文字输出，PCIe 模型继续运行；总线异常不属于可继续的超时。host 核对长度、CRC、会话和帧号，异常保存日志和 DDR 诊断。`--infer-timeout 900` 限制一帧，`--tool-timeout 30` 限制单次工具调用；退出或超时不复位板卡，也不能中断已开始的推理。

### 单命令读取 DDR

在 host 仓库根目录运行，Python 3.10 标准库即可。地址参数是 **全局 DDR 地址**，脚本减去 `0x80000000` 得到 PCIe 偏移。默认允许原32 MiB区；读模型数据时加 `--model-data`，额外允许已批准的 `0xF0000000–0xF0FFFFFF`，不要传RVV别名。

```bash
# 读取目标区的16字节，按16位十六进制显示，保存原始数据和日志
python3 tools/read_ddr.py 0x80003a40 16 --width 16

# 读取源区的16字节，按16位十六进制显示，用于与目标区比较
python3 tools/read_ddr.py 0x80003dc0 16 --width 16

# 检查程序入口的16字节，按32位十六进制显示
python3 tools/read_ddr.py 0x80000000 16 --width 32

# 读取本版模型输入开头16字节，仅用于停核后定位，不判断整帧通过
python3 tools/read_ddr.py 0xf0081000 16 --width 32 --model-data

# 检查起始4KiB是否全零；屏幕默认只预览前256字节
python3 tools/read_ddr.py 0x80000000 0x1000 --width 32
```

`0x80003a40`、`0x80003dc0` 是当前独立 `selftest` 固件的 FP16 目标/源区；模型固件或新编译固件须按对应 ELF 确认地址。结果保存在终端打印的 `yolo26_pc/outputs/ddr-read/run-随机编号/`：`001-ddr.bin` 是完整读回数据，`001-ddr.log` 是平台工具输出，`data.txt` 是显示的预览，`report.json` 和 `diagnostics.jsonl` 记录地址、长度、哈希、耗时及失败原因。

`All bytes zero: True` 仅在完整读取且所有字节为零时出现，覆盖整段请求范围，不仅是预览；读取失败返回非零退出码并保留日志/已有数据。脚本复用现有 PCIe 互斥锁、短临时文件名、错误文字及长度检查，只读 DDR。需要改变显示长度时加 `--max-print 字节数`。工具从 PATH 查找，找不到时尝试 `/rv/fpga/bin/pbload`；也可加 `--pbload /绝对路径/pbload`。许可工作目录默认是当前目录，修改时加 `--tool-cwd /许可工作目录`。

比较自测的不同阶段时先用已有 GDB 停住 CPU，再读取 DDR；CPU 运行中的读回不是原子快照。硬件断点每次只占用一个，`thbreak` 命中后自动删除；失败定位用断点自动停核，不依赖人工赶上 UART 输出。

### 旧缓存定位入口

`tools/rvv_store_probe.gdb` 仅对应旧 `selftest.bin/elf`，不能用于 CLP 固件。原方案将缓存 RVV 写入后直接 invalidate，可能丢弃脏数据，RVV→AMU 也缺少 clean；之前取消隐式 RVV 栈拷贝并不证明故障修复。当前模型已修正交接并接入CLP；旧固定地址不能用于本版模型。

### UART 无输出：只读采集 DDR

保留当前板端状态，停止其他 PCIe 控制程序，minicom 可以继续打开。`--firmware` 必须对应**当前已加载的 BIN**：CLP 测试选 `clp-selftest`，旧独立自测选 `selftest`，模型选 `demo-pcie`。

```bash
# 只读采集当前正式模型的启动/节点/trap状态；日志换成本次实际目录
python3 tools/collect_board_diag.py --firmware demo-pcie --samples 1 \
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

完整模型 QEMU 回归：`python3 yolo26_riscv/board/tests/test_model.py`；加 `--capture` 验证节点快照/握手并逐字节核对两帧全部输出。它执行实际启动、自测、PCIe 邮箱和两帧334节点，保存原图、候选框、绘框图和差异报告；使用统一RAM模拟地址，不证明实板缓存一致性。

街景视频两帧的开发验证也已完成，每帧11个框。复跑时，先在已有NumPy/OpenCV环境运行 `python tools/check_clp_video.py`，记录打印的清单路径；再在Linux/WSL运行 `python3 yolo26_riscv/board/tests/test_model.py --manifest 清单路径`。结束后在图像环境运行 `python tools/check_clp_video.py --render 本次结果目录`，生成原帧/检测PNG、`raw.mp4`、`demo.mp4`。视频将两个实际检测结果各停留2秒，播放帧率不代表推理速度；真实连续演示使用第4节的live_demo入口。
