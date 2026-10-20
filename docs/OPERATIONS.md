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
| 3 | 默认视频三帧验证，保存PNG及MP4 |
| 4 | 自选视频连续运行，默认30帧，0为全片 |
| 5 | 已有视频数值报告、错框标记及PNG |
| 6 | 已有运行的性能三页及完整节点计时 |
| 7 | 已有帧的检测效果对照 |
| 8 | 排障子菜单：1进度、2完整BUS查验、3旧BUS报告、4重载固件 |
| 9 | 清理临时材料 |
| 0 | 退出 |

主菜单连续编号1–9，排障子菜单0返回主菜单。旧独立采集、首差异和SVG转换不再占菜单；原有命名命令仍可使用。除17外的旧两位序号不再接受；17仅作为5的离线别名，兼容未修改的golden校验脚本提示，避免改变其源码哈希而影响旧缓存。不要把查看报告误当作重新运行板子。

## 2. 视频运行与结果

新版验收按 **1更新 → 3同一视频前三帧 → 确认VIDEO CHECK PASSED → 6新运行三页**。5可重看报告、补算未完成的golden及查看PNG。v6已有 `run-bivzu9qm` 三帧有效实板计时；同次数值结果尚未提供，直接用5查看旧运行，不必重跑。性能版本、基线匹配和检测效果对照见[模型手册](MODEL.md#performance)。第二页应显示 `rvv_bulk_v6`，固定图input_im2col=0；保留 `run-2i3ciq3m`，自动优先按相同输入匹配v5基线。

```bash
# 检查host并加载固件一次，按顺序处理默认视频的前三帧
bash tools/host.sh video

# 自选视频并处理全片，每处理完一帧就显示该帧结果
python3 tools/host_video.py --load --source /绝对路径/input.mp4 --max-frames 0

# 没有图形会话时处理全片，仍保存PNG和演示视频
python3 tools/host_video.py --load --source /绝对路径/input.mp4 --max-frames 0 --no-display
```

默认视频为 `yolo26_pc/samples/pexels-3796613.mp4`，来源与SHA记录在视频程序中。3默认三帧；4默认30帧，可改视频、帧数（0为全片）及是否显示。每次只提交一帧；等待时窗口保持响应，收到结果才更新对应画面。

窗口内按 **B** 标记当前已显示的错框帧，**Q/Esc** 停止。等待下一帧时标记的是仍在显示的上一帧。标题含运行编号和源帧号；源帧从0开始，板端会话帧号从1开始。

终端打印输出目录 `yolo26_pc/outputs/host-video/run-*/`：

| 文件 | 内容 |
|---|---|
| `source-NNNNNN.png` / `detected-NNNNNN.png` | 原帧 / 检测结果，PNG可用eog打开 |
| `candidates-NNNNNN.bin` / `result-NNNNNN.json` | 原始候选 / 映射、哈希及计时 |
| `demo.mp4` | 只包含已完成帧的演示录像，播放帧率不代表推理速度 |
| `report.json` / `diagnostics.jsonl` | 原始运行记录 / PCIe事件、板端快照及异常 |
| `video-summary.json` / `video-page-1/2/3.txt` | 离线检查与便于截图的三页报告 |
| `golden-NNNNNN.bin/json` | 独立数值参考及来源哈希 |

视频结束后自动在host CPU计算golden，不访问板子。通过条件、缓存复用和结论边界见[视频数值验收](VALIDATION.md#视频检查的判据)。主动停止及失败保留已完成帧，不将未处理帧计入成功。

**5只分析旧结果**：未完成的golden可续算，不需要先选4；新运行通过指针定位，不按host系统日期猜测。5可填错框帧号和说明，最后用eog查看PNG。

```bash
# 离线分析指定旧视频结果并标记源帧1；不加载或访问板子
python3 tools/video_report.py /实际结果目录 --flag 1 --note "错框位置说明"
```

反馈问题时提供三页报告截图及错框PNG，保留运行编号和源帧号；完整数据留在host。UART文字需另开终端会话日志，JSONL不会自动采集串口。

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

主菜单8进入排障：1看后台进度、2启动完整BUS查验、3看已有BUS三页、4重载正式固件。只有需要逐节点定位时才启动查验；优先复用校验通过的完整旧BUS证据，不足时才重新采集和普通复跑。后台查验普通SSH断线后继续。新优化优先同三帧独立golden与计时，不默认重采全部节点。

排障子菜单3对旧规则报告自动离线重分析一次，另存 `bus-rtl-analysis.json`；不修改原 `summary.json`，缺原始数据明确报错，不自动重跑板子。

```bash
# 指定已有full-check目录离线重分析；不访问板子
python3 tools/full_check.py --reanalyze --report "实际结果目录"

# 查看已保存分析第二页，第三页将2改为3
python3 tools/full_check.py --report "实际结果目录" --page 2

# 读取诊断和程序前缀；不复位、不提交推理
python3 tools/collect_board_diag.py --firmware demo-pcie --samples 1 --pbload /rv/fpga/bin/pbload --tool-cwd /rv/fpga/bin

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

菜单9等价于清理入口。只删除安装缓存、源码字节码、白名单旧构建产物，以及无帧、未被引用且访问板子前失败的试运行。固件、工具链、环境、BUS原始证据、有效或中断帧、性能/精度记录、板端/PCIe故障、当前 `n550-board` 和 `host-verify` 构建均保留。

结果JSON可能跨目录引用张量，不能整目录删除outputs；无法核实引用时保留，有任务持锁时拒绝清理。结束标志 `HOST CLEANUP COMPLETED`，记录位于 `yolo26_pc/outputs/cleanup/host-cleanup-*.json`；不按文件日期判旧。

## 6. 构建与发布核对

以下在配置现有ESWIN工具链的Linux/WSL开发机执行，不访问板子：

```bash
# 从当前源码构建正式板端程序
bash yolo26_riscv/build_board.sh

# 检查已准备的交付固件、冻结模型、节点资料及BIN/ELF一致性
python3 tools/check_board_release.py
```

构建输出位于 `yolo26_riscv/build/n550-board/`，不会自动替换 `firmware/`。交付目录须同步本次BIN、ELF、符号、反汇编和校验清单后再核对；原QEMU及标量实验不是板端固件。
