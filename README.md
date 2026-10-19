# N550 YOLO26 板端演示

链路：**host 采集/预处理 → PCIe 写 DDR → N550 RVV+AMU 推理 → PCIe 读结果 → host 绘框、显示、录像**。host 是物理连接 S2C 的 Linux 电脑；开发机远程查看其桌面。UART 只输出调试文字。

当前正式演示固定 YOLO26n、416×416、完整334节点，score>0.25，不加NMS。现阶段开展板端耗时统计与离线检测效果对比；正式参数待评估后再调整，量化和跟踪暂不开展。

## 当前状态

BUS 完整采集已完成离线定位：94个AMU卷积、6,766,084个输出按最新RTL行为逐位一致，其余240个节点按实际输入重放一致；普通推理与采集推理的最终7200字节完全一致。正式查验程序已更新，保留原QEMU对照。范围和证据见 [验收状态](docs/VALIDATION.md)。

现场视频三帧的最终输出已与RTL golden逐位一致；误检和重复框也存在于原始模型。权重预打包的实板模型周期已下降25.0%。本轮将94层AMU卷积的窗口提取与FP16输入打包融合，每个空间块生成一次并供所有输出通道复用，实际收益待现场验证；每个输出的K累加及偏置规则保持不变。

现场操作：**1更新 → 15运行同一视频前三帧 → 确认VIDEO CHECK PASSED → 18选择新运行并截图三页**，不需重新逐节点采集。15加载新固件一次，UART仍显示94层权重准备；18第二页应显示 `Input path: fused_window_fp16_reuse`、帧内权重转换为0。保留上一版 `run-ls19m5ja` 和原始 `run-t5240cg4`；18按相同输入哈希自动优先比较上一版。

菜单 **18：性能三页报告**、**19：阈值/重复框去重对照PNG** 均可离线分析已保存结果。18兼容旧计时表；没有板端计数的历史运行仍只报Host计时。核心频率不清楚就回车，仅显示周期，不能填设计目标频率。

检测效果对比可直接 **1更新 → 19 → 选择旧三帧结果**，默认首/中/末帧，无需重新推理。比较阈值0.25/0.35、是否按同类别IoU>0.5去重；不改演示默认值。输出四组PNG、精确删除的候选框编号和可选人工标注模板；框少不代表精度高。计时边界、标注格式和结果位置见 [性能与检测效果评估](docs/TUNING.md)。

## 1. 更新和查看现有结果

以下命令均在host仓库根目录执行。已有后台任务运行时，菜单1会拒绝更新；先等待任务结束。

```bash
# 打开统一操作菜单，选1更新程序和文档；使用已配置的Git SSH
bash tools/host.sh
```

选 **12** 查看BUS三页结果。旧报告使用旧参考规则时，自动对完整旧数据离线重分析一次，另存 `bus-rtl-analysis.json`；不重新加载固件、不访问PCIe、不修改原summary.json。之后重复查看直接读新报告。缺原始采集时明确报错，不自动重跑板子。

若需指定目录：

```bash
# 将路径替换为已有full-check结果目录；只离线重分析
python3 tools/full_check.py --reanalyze --report "实际结果目录"

# 查看保存的新分析第二页；第三页将2改为3
python3 tools/full_check.py --report "实际结果目录" --page 2
```

保留菜单编号：1更新；2/3查看采集/首差异；5后台采集BUS；6进度；7选择旧采集；10加载正式模型；11完整BUS查验；12三页汇总；13旧SVG转PNG；14检查host视频能力；15运行视频三帧；16连续视频检查；17视频报告/错框标记/看PNG；18性能报告；19检测效果对照；20清理临时材料；0退出。旧首层专项菜单4/8/9已移除。

## 2. 视频依赖与演示

host支持Linux x86_64、Python3.8。依赖随Git提供，安装不联网、不改系统时间，也不安装PyTorch。

```bash
# 安装到项目本地目录，核对依赖哈希并测试解码、前后处理、PNG及MP4
bash tools/setup_host_video.sh

# 在可用图形会话中运行：检查host、加载模型一次、按顺序处理三帧
bash tools/host.sh video
```

依赖和显示已验证过的host无需重复安装，直接在菜单选 **1更新 → 16连续视频检查 → 17查看报告**。16默认连续30帧，可输入视频路径、修改帧数（0为全片）或关闭显示。15仍为三帧入口；14仅在需要排查host依赖时使用。

显示要求有效DISPLAY；使用已打通的SSH图形转发或host远程桌面。窗口内按 **B** 标记当前显示的已完成帧，按 **Q/Esc** 停止；窗口标题含运行编号及帧号。等待下一帧时按B，标记的是仍在显示的上一帧。源视频帧号从0开始，PCIe/板端会话帧号从1开始。图形运行依赖当前SSH会话，断线不保证继续。

每帧完成推理后立即更新画面，并保存原帧、检测PNG、候选框和报告，最终生成可演示的 `demo.mp4`。终端打印实际输出目录 `yolo26_pc/outputs/host-video/run-*/`。只有实际处理成功的帧才进入结果；失败保留已完成帧和错误信息。

处理完视频后自动在host CPU上生成逐帧golden：从相同输入独立前推全部334节点，用已有静态 `firmware/host-verify` 的AMU RTL数值规则计算，不使用实板中间结果；无需PyTorch、编译器或额外下载。golden计算在显示和录像结束后进行，不访问板子，单独统计耗时并打印帧号/节点进度。

结束标志为 **`VIDEO CHECK PASSED`**（功能与全部已完成帧的数值检查均通过）、**`VIDEO CHECK FAILED`**（功能异常、数值差异或参考计算失败）或 **`VIDEO CHECK INCOMPLETE`**（主动停止或检查未完成）。功能检查覆盖PCIe会话/帧号/CRC、保存的输入输出哈希、候选合法性、坐标还原/绘框及MP4；数值检查逐位比较每帧 `[300,6]` 全部1,800个FP32数，包含正负零差异，不设误差阈值。允许无检测帧、还原前超出416边界的候选框。

**17不运行板子**，可直接选择当前已有三帧结果，自动补算golden，无需先选16。新运行由指针定位，避免host回拨时间选错。报告分别显示FUNCTIONAL和NUMERICAL，数值异常列出帧号、差异数量、首个差异的候选/字段/双方十六进制值，以及坐标/置信度/类别差异；发现差异仍检查后续帧。可选填错框帧号和说明，逐页截图三页汇总，最后选择用eog打开预览及标记帧。

原始report.json和逐帧结果不改动；报告另存 `video-summary.json`、`video-page-1/2/3.txt`，golden另存 `golden-NNNNNN.bin/json`，含335个节点输出哈希。缓存绑定输入、权重、图、静态程序、数值规则及调度代码，并核对输出哈希；重复查看直接使用有效缓存。中断后选17可复用已完成帧，只重算未完成的golden，不重跑板子；缓存数据损坏明确报错，不静默覆盖。

发现问题时，发给我 **三页报告截图＋错框PNG（含运行编号和帧号）** 即可；完整JSON、诊断与原图留在host。每帧文件为 `source-NNNNNN.png`、`detected-NNNNNN.png`、`candidates-NNNNNN.bin`、`result-NNNNNN.json`。板端状态、trap诊断和错误堆栈记录在报告/`diagnostics.jsonl`；UART文字不自动采集。

移植数值是否与参考一致由程序判断，无需逐帧肉眼确认。golden采用已确认的AMU RTL规则；原QEMU/PyTorch结果可能有已知舍入差异，不混用为逐位判据。模型自身的漏检、错认需要真实标注才能自动评估；演示观感可人工抽查。报告保留人工标记。PCIe往返含工具调用和轮询开销，不能当作纯AMU耗时；SSH显示也影响整帧速度。

```bash
# 自选视频并处理全片；一次只提交一帧，处理完再显示和继续下一帧
python3 tools/host_video.py --load --source /绝对路径/input.mp4 --max-frames 0

# 无图形会话时仍可处理全片并生成检测PNG和演示视频
python3 tools/host_video.py --load --source /绝对路径/input.mp4 --max-frames 0 --no-display

# 离线分析指定旧视频结果、标记第1帧；不加载模型、不访问PCIe
python3 tools/video_report.py /实际结果目录 --flag 1 --note "错框位置说明"
```

默认样本为 `yolo26_pc/samples/pexels-3796613.mp4`，来源与SHA记录在视频程序中。MP4播放帧率与板端实际处理速度分别记录，不代表推理实时帧率。

## 3. 加载模型与单图检查

先打开UART终端，设置115200、8N1、软硬流控关闭。实际端口结合接线确认；串口扫描不发送数据。

```bash
# 列出串口身份和占用情况
python3 yolo26_pc/serial_probe.py

# 示例：查看调试日志，按实际接线替换端口
minicom -D /dev/ttyUSB2 -b 115200

# 使用现场复位脚本：保持复位、加载BIN到DDR、释放复位
bash tools/load_board.sh --reset-dir /rv/fpga/bin --pbcopy /rv/fpga/bin/pbcopy
```

默认 `firmware/demo-pcie.bin` 已含程序、模型图和权重。PCIe偏移0对应CPU地址0x80000000；ELF供符号调试，不能当BIN加载。每帧不复位、不重新加载权重。启动UART依次输出BOOT、CLP地址、自测、READY；等待READY后提交输入。

```bash
# 固定BUS单图，保存结果、参考差异及检测PNG
python3 yolo26_pc/pcie_prepared.py --sample bus --pbcopy /rv/fpga/bin/pbcopy --pbload /rv/fpga/bin/pbload
```

单图输出在 `yolo26_pc/outputs/board-pcie/`。host输入为等比例补边至416×416、RGB/NCHW、小端FP32、除以255，共2,076,672字节；输出为 `[300,6]` FP32，共7200字节，字段是 `x1,y1,x2,y2,score,class`。按score>0.25过滤、还原并裁剪坐标，不加NMS。

## 4. 通用查验与排障

- **11**：只在修改模型、算子、固件或硬件后需要回归时使用。完整旧证据优先复用；不足时加载正式固件，采集334节点后普通推理复跑一次。后台任务普通SSH断线后继续，6看进度、12看结果。
- `firmware/host-verify`：静态Linux x86_64查验程序，host不需要编译器。RTL参考处理有限对齐窗口、16项累加、规格化和偏置；QEMU参考单独保留。分类为exact、rtl_match、unexplained、input_unverified、missing。
- 仅当前规则下覆盖完整、输入已核实、输出逐位匹配、候选合法且普通复跑一致，才显示STATUS: passed。不使用经验误差阈值；结论仅覆盖该输入。
- **13**：旧SVG离线转PNG。已有图形会话可用eog打开PNG，无需Firefox。

```bash
# 只读取板端诊断及程序前缀，保存日志；不复位、不提交推理
python3 tools/collect_board_diag.py --firmware demo-pcie --samples 1 --pbload /rv/fpga/bin/pbload --tool-cwd /rv/fpga/bin

# 将已有完整BUS证据校验并导出；不推理、不要求首层专项数据
python3 tools/export_bus_evidence.py

# 查看通用DDR读取参数，后续按需指定地址与长度
python3 tools/read_ddr.py --help
```

通过Git SSH上传某次诊断时，使用封装脚本；只提交指定目录到host-diagnostics分支，源码工作区保持不变。

```bash
# 将路径替换为采集工具打印的诊断目录；需要仓库写权限
bash tools/upload_diagnostics.sh "yolo26_pc/outputs/host-diagnostics/run-实际编号"
```

独立CLP自测仍保留，必要时用加载脚本的 `--clp-selftest` 选择；测试后重新加载正式模型。历史缓存自测、首层三分块固件和一次性迁移工具已移除。

不要整目录删除outputs：旧报告可能引用其他运行目录中的张量。保留本次BUS原始证据、分析结论及有效运行输出。

清理使用 **1更新 → 20**，或下面的封装命令。仅清理安装缓存、源码目录的Python字节码、白名单中的旧构建产物，以及未被引用、无帧数据且在访问板子前就失败的视频试运行。固件、工具链、Python环境、BUS证据、有效/中断帧、计时与精度对照、板端或PCIe故障记录均保留。当前构建的 `n550-board` 和 `host-verify` 保留。扫描所有结果JSON的跨目录引用，无法核实时保留；运行任务持有锁时拒绝清理。

```bash
# 在host仓库根目录清理已确认可删除的材料；不访问或复位板子
bash tools/host.sh cleanup

# 仅预览清理清单，不删除文件
python3 tools/cleanup_host.py
```

结束标志为 `HOST CLEANUP COMPLETED`，完整删除/保留记录在 `yolo26_pc/outputs/cleanup/host-cleanup-*.json`。清理不按文件日期判旧，兼容host回拨系统时间；不会整目录删除outputs。两个旧UART版Release及标签 `v0.1.0-board-bringup`、`v0.2.0-host-diagnostics` 已删除，交付以当前main分支和 `firmware/` 为准。

平台与同步条件见 [BOARD](docs/BOARD.md)，PCIe协议见 [PCIE](docs/PCIE.md)，日志/错误字段见 [DIAGNOSTICS](docs/DIAGNOSTICS.md)。现场复位、PCIe驱动、DDR及JTAG由平台同事提供。

## 构建机

```bash
# 在配置现有ESWIN工具链的Linux/WSL开发机重新构建；不访问板子
bash yolo26_riscv/build_board.sh
```

构建结果在 `yolo26_riscv/build/n550-board/`，不会自动替换firmware交付文件。原QEMU和标量开发参考保留；当前不再要求第二张图验收。
