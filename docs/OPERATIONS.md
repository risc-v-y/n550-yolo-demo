# Host操作手册

本页集中维护运行命令、菜单、结果位置和清理步骤。host指物理连接S2C的Linux电脑；所有命令均在host仓库根目录执行，构建章节除外。当前实现见[模型手册](MODEL.md#implementation)，证据及通过条件见[VALIDATION](VALIDATION.md)。

## 1. 准备与更新

host使用Linux x86_64、Python3.8；视频依赖随Git提供，不安装PyTorch，不访问在线软件源，也不需要调整系统时间。

```bash
# 打开统一菜单，选1更新；已有后台任务运行时先等待结束
bash tools/host.sh

# 首次使用或依赖异常时：核对安装包，安装到项目目录并检查视频前后处理
bash tools/setup_host_video.sh
```

依赖已通过检查就不用重复安装。离线包来源与安装兼容说明见[host_deps](../host_deps/py38-linux-x86_64/README.md)。窗口显示需要有效DISPLAY，使用已打通的SSH图形转发或host远程桌面；图形任务依赖当前会话，SSH断线不保证继续。无图形会话可用 `--no-display` 保存结果。

| 菜单 | 用途 |
|---|---|
| 1 | 更新程序和文档 |
| 2 | 检查host视频处理能力，不访问板子 |
| 3 | 性能版三帧演示，只保存MP4和汇总 |
| 4 | 性能版自选视频，默认30帧，0为全片 |
| 5 | 新版简要报告；兼容旧详细数值报告和PNG |
| 6 | 性能三页；普通模式完整节点计时标记未采集 |
| 7 | 已有帧的检测效果对照 |
| 8 | 排障子菜单：1进度、2完整BUS查验、3旧BUS报告、4重载诊断固件、5首块诊断、6原输入复跑、7诊断版视频 |
| 9 | 清理临时材料 |
| 10 | 两种固件验收：诊断版3帧＋性能版30帧，后台运行 |
| 11 | 最新验收三页报告 |
| 0 | 退出 |

主菜单连续编号1–11，排障子菜单0返回主菜单。旧独立采集、首差异和SVG转换不再占菜单；原有命名命令仍可使用。除10、11和兼容别名17外的旧两位序号不再接受；17仅作为5的离线别名，兼容未修改的golden校验脚本提示，避免改变其源码哈希而影响旧缓存。不要把查看报告误当作重新运行板子。

## 2. 视频运行与结果

### 更新后的自动验收

菜单 **1更新 → 10自动验收**，无需先跑排障。固定已通过的`m128/n32/pack25/compute0/graph11`和串行上传，先加载诊断版验证前三帧及完整节点计时，再加载性能版连续处理前30帧。不再访问旧基线固件，也不重跑旧18组、增量5组或重叠上传。

性能版测量路径与菜单3/4相同；验收仅暂存帧材料于内存，全部计算、显示提交和录像结束后才落盘和运行golden，避免把这些分析开销计入帧时间。两份固件前三帧输入哈希必须一致；所有候选1800字、绘框像素及录像帧数核对通过后，才写入绑定两份固件SHA的selected.json。通信、数值或记录异常立即停止，保留已取得材料，不自动重试写入。SSH普通断线后后台任务继续。

- **8→1**：查看进度和日志尾部，不重新启动任务。
- **11**：查看最新三页结果；未结束时明确显示running。
- 失败报告区分完成帧数、golden核对帧数及录像帧数，第三页显示错误；完整帧号和头差异保存在diagnostic或performance子目录的report.json。诊断版头读回不一致时另有request-readback.json；性能版不执行该读回。
- 结束标志为 **MODE ACCEPTANCE PASSED/FAILED**；通过后回到 **3/4** 演示。失败保留现场，不自动复位或重试写入。
- 结果在`yolo26_pc/outputs/optimization/run-*/`：report.json、terminal.log、diagnostic/demo.mp4、performance/demo.mp4及各自汇总。latest.json指向本次；selected.json只在通过后更新。成功删除本次逐帧临时证据，错误证据保留。

```bash
# 从已有统一入口启动后台自动验收
bash tools/host.sh optimize

# 查看三页结果；不会访问板子
bash tools/host.sh optimization-report
```

验收默认无窗口，报告中的显示耗时标记未采集；需要包含OpenCV显示提交计时，可从有效图形会话用`python3 tools/verify_modes.py --display --output /绝对路径/新结果目录`运行。两种方式均重绘核对像素、解码检查MP4，不把OpenCV提交时间等同远端屏幕呈现。30帧只是本次覆盖范围。

### 日常视频演示

```bash
# 加载性能版，处理默认视频前三帧并显示、录像
bash tools/host.sh video

# 自选视频，处理完整文件；完成一帧立即显示并加入MP4
python3 tools/host_video.py --load --source /绝对路径/input.mp4 --max-frames 0

# 无图形会话时处理全片，只保存MP4及汇总
python3 tools/host_video.py --load --source /绝对路径/input.mp4 --max-frames 0 --no-display

# 排障视频：加载诊断版，开启数据CRC、UART进度和完整节点计时
bash tools/host.sh video-diagnostic

# 仅排障需要时：FP32串行路径，保留PNG、BIN、完整profile并计算golden
python3 tools/host_video.py --load --detailed --max-frames 3
```

默认视频为`yolo26_pc/samples/pexels-3796613.mp4`。菜单4可选视频、帧数（0为全片）和显示开关。文件视频按序处理、不丢帧；下一帧解码预处理与当前推理并行。上传默认串行，即收到当前结果后再上传下一帧；仅使用验收通过并保存的重叠模式时，才在当前推理期间上传下一槽。窗口持续处理事件，收到新结果时更新；Q/Esc停止。

正常输出位于`yolo26_pc/outputs/host-video/run-*/`，仅demo.mp4和report.json，结束标志为`VIDEO COMPLETED`。该标志表示功能完成，**普通模式没有golden比较**，不是数值验收通过。录像按源视频FPS播放，不代表推理FPS。失败/中断保留相关输入、原帧、候选（若已取得）和近期通信事件；host停止不表示已中断板端计算。

菜单5/6读取新版报告。性能版仅记录整帧传输至结果、连续完成间隔及显示提交；板端周期和细分计时明确为NOT COLLECTED，不填零。诊断版另有全部节点计时。逐帧PNG、错框标记、菜单7检测对照需要旧详细结果或显式--detailed运行。旧报告仍可查看。

性能版正常运行没有UART进度，异常/trap才打印并发布最小错误快照。它不能发现所有静默数据位错；出现异常后切换诊断版重新运行相同视频，增加校验和定位信息，但不能恢复上一次未采集的历史轨迹。

<a id="saved-input-replay"></a>

### 特殊排障入口

仅在复查旧首块现场时使用 **8→5首块诊断**或 **8→6原输入普通复跑**，保留原输入、独立golden和完整profile；当前计时版号9，兼容读取旧7/8版。正常优化与演示不依赖这些入口，现场字段见[DIAGNOSTICS](DIAGNOSTICS.md)。

## 3. 固件加载与固定图片

先按[串口识别](SERIAL_PROBE.md)确认实际端口；UART仅便于观察日志，不传输图片或模型结果。

```bash
# 只扫描串口身份、权限及占用，不打开或发送串口数据
python3 yolo26_pc/serial_probe.py

# 示例调试终端：按实际接线替换端口，并设置8N1、关闭软硬流控
minicom -D /dev/ttyUSB2 -b 115200

# 保持复位 → 加载完整BIN到DDR → 释放复位；工具由平台方提供
bash tools/load_board.sh --reset-dir /rv/fpga/bin --pbcopy /rv/fpga/bin/pbcopy

# READY后提交固定BUS输入，保存候选、QEMU参考差异及检测PNG
python3 yolo26_pc/pcie_prepared.py --sample bus --pbcopy /rv/fpga/bin/pbcopy --pbload /rv/fpga/bin/pbload
```

默认加载 `firmware/demo-pcie.bin`，已含程序、图与原始权重；ELF仅用于调试。部署一次，逐帧不复位、不重载权重。启动日志顺序含BOOT、自测、权重准备、READY。加载失败不释放复位。地址及启动前提见[BOARD](BOARD.md)。

单图输出在 `yolo26_pc/outputs/board-pcie/`。该命令保留原QEMU差异统计；已知RTL舍入差异不能直接判成移植错误，说明见[VALIDATION](VALIDATION.md)。

## 4. 查验、排障与反馈

主菜单8进入排障：1看后台进度、2启动完整BUS查验、3看已有BUS三页、4重载正式固件、5首块定位、6加载新版并复用原输入验收。只有需要逐节点定位时才启动查验；优先复用校验通过的完整旧BUS证据，不足时才重新采集。后台查验普通SSH断线后继续。新优化先用同一保存输入做独立golden与计时，不默认重采全部节点。

排障子菜单3对旧规则报告自动离线重分析一次，另存 `bus-rtl-analysis.json`；不修改原 `summary.json`，缺原始数据明确报错，不自动重跑板子。

```bash
# 指定已有full-check目录离线重分析；不访问板子
python3 tools/full_check.py --reanalyze --report "实际结果目录"

# 查看已保存分析第二页，第三页将2改为3
python3 tools/full_check.py --report "实际结果目录" --page 2

# 读取诊断和程序前缀；不复位、不提交推理
python3 tools/collect_board_diag.py --firmware demo-pcie --samples 1 --pbload /rv/fpga/bin/pbload --tool-cwd /rv/fpga/bin

# 若出错时运行的是性能版，用对应ELF读取最小错误快照；不重新加载
python3 tools/collect_board_diag.py --firmware demo-fast --samples 1 --pbload /rv/fpga/bin/pbload --tool-cwd /rv/fpga/bin

# 校验并导出已有BUS证据；不推理、不要求旧首层专项数据
python3 tools/export_bus_evidence.py

# 查看通用DDR读取参数，后续按需指定地址和长度
python3 tools/read_ddr.py --help

# 通过Git SSH上传指定诊断目录到host-diagnostics分支；需要仓库写权限
bash tools/upload_diagnostics.sh "yolo26_pc/outputs/host-diagnostics/run-实际编号"
```

错误码、自测字段、trap和UART状态见[DIAGNOSTICS](DIAGNOSTICS.md)；PCIe失败、取消和超时语义见[PCIE](PCIE.md)。独立CLP自测可用加载脚本的 `--clp-selftest` 选择，结束后重新加载正式模型，避免沿用旧ELF地址。

## 5. 清理

```bash
# 执行已有白名单及引用检查后清理；不访问或复位板子
bash tools/host.sh cleanup

# 只预览清单，不删除
python3 tools/cleanup_host.py
```

菜单9等价于清理入口。删除安装缓存、源码字节码、白名单中间构建，以及已由新成功回归替代且无引用的旧 `clp-model-validation` 优化回归；同版优先按latest指针选择，不因host时钟回拨删除最新结果。无帧、未被引用且访问板子前失败的试运行也可删除。固件、工具链、环境、最新优化回归、BUS原始证据、正常/中断视频和板端/PCIe故障材料保留；当前 `n550-board` 和 `host-verify` 正式构建保留。

结果JSON可能跨目录引用张量，不能整目录删除outputs；无法核实引用时保留，有任务持锁时拒绝清理。结束标志 `HOST CLEANUP COMPLETED`，记录位于 `yolo26_pc/outputs/cleanup/host-cleanup-*.json`；不按文件日期判旧。

## 6. 构建与发布核对

以下在配置现有ESWIN工具链的Linux/WSL开发机执行，不访问板子：

```bash
# 从同一源码构建性能版demo-fast与诊断版demo-pcie
bash yolo26_riscv/build_board.sh

# 检查已准备的交付固件、冻结模型、节点资料及BIN/ELF一致性
python3 tools/check_board_release.py
```

构建输出位于 `yolo26_riscv/build/n550-board/`，不会自动替换 `firmware/`。可用`--mode performance`或`--mode diagnostic`只构建一种；交付目录须同步两份BIN、ELF、符号、反汇编和校验清单后核对。加载用BIN，GDB用相同模式的ELF，不能混用地址；原QEMU及标量实验不是板端固件。
