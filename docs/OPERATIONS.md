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
| 3 | 默认三帧流水演示，只保存MP4和汇总 |
| 4 | 自选视频连续运行，默认30帧，0为全片 |
| 5 | 新版简要报告；兼容旧详细数值报告和PNG |
| 6 | 性能三页；普通模式完整节点计时标记未采集 |
| 7 | 已有帧的检测效果对照 |
| 8 | 排障子菜单：1进度、2完整BUS查验、3旧BUS报告、4重载固件、5首块诊断、6原输入普通复跑 |
| 9 | 清理临时材料 |
| 10 | 全链路自动验收和实板配置选择，后台运行 |
| 11 | 最新验收三页报告 |
| 0 | 退出 |

主菜单连续编号1–11，排障子菜单0返回主菜单。旧独立采集、首差异和SVG转换不再占菜单；原有命名命令仍可使用。除10、11和兼容别名17外的旧两位序号不再接受；17仅作为5的离线别名，兼容未修改的golden校验脚本提示，避免改变其源码哈希而影响旧缓存。不要把查看报告误当作重新运行板子。

## 2. 视频运行与结果

### 更新后的自动验收

菜单 **1更新 → 10自动验收**，无需先跑排障。后台任务提取更新前正式固件作临时基线，同三帧运行两轮；读取板上容量，比较卷积分块、输入复用、算子和图候选；最后连续处理默认视频前30帧，并与原golden比较显示结果。只有两轮模型周期均下降、显示一致的候选才被接纳，最终30帧通过后保存绑定固件SHA的配置。无收益候选不启用。SSH普通断线后任务继续；核心仍需保持运行。

- **8→1**：查看进度和日志尾部，不重新启动任务。
- **11**：查看最新三页结果；未结束时明确显示running。
- 结束标志：`OPTIMIZATION PASSED`或`OPTIMIZATION FAILED`。
- 通过后回到 **3/4** 演示；失败保留现场和异常材料，不自动复位或重试写入。
- 结果在`yolo26_pc/outputs/optimization/run-*/`：report.json、terminal.log和video/demo.mp4。latest.json指向本次；selected.json只在通过后更新。成功时删除本次临时旧固件和逐帧证据，候选计时/差异保留在汇总中。

```bash
# 从已有统一入口启动后台自动验收
bash tools/host.sh optimize

# 查看三页结果；不会访问板子
bash tools/host.sh optimization-report
```

验收默认不要求图形会话，它重绘并比较图像像素、校验MP4帧数。实际远端显示速度需正常演示观察，不能从后台结果推断。前三帧用于选型，30帧是本次验收范围，不代表所有输入。

### 日常视频演示

```bash
# 检查host并加载固件一次，流水处理默认视频前三帧
bash tools/host.sh video

# 自选视频，处理完整文件；完成一帧立即显示并加入MP4
python3 tools/host_video.py --load --source /绝对路径/input.mp4 --max-frames 0

# 无图形会话时处理全片，只保存MP4及汇总
python3 tools/host_video.py --load --source /绝对路径/input.mp4 --max-frames 0 --no-display

# 仅排障需要时：FP32串行路径，保留PNG、BIN、完整profile并计算golden
python3 tools/host_video.py --load --detailed --max-frames 3
```

默认视频为`yolo26_pc/samples/pexels-3796613.mp4`。菜单4可选视频、帧数（0为全片）和显示开关。文件视频按序处理、不丢帧；当前推理与下一帧准备/上传重叠。窗口持续处理事件，收到新结果时更新；Q/Esc停止。

正常输出位于`yolo26_pc/outputs/host-video/run-*/`，仅demo.mp4和report.json，结束标志为`VIDEO COMPLETED`。该标志表示功能完成，**普通模式没有golden比较**，不是数值验收通过。录像按源视频FPS播放，不代表推理FPS。失败/中断保留相关输入、原帧、候选（若已取得）和近期通信事件；host停止不表示已中断板端计算。

菜单5/6读取新版简要报告；完整节点profile标记NOT COLLECTED。逐帧PNG、错框标记、菜单7检测对照需要已有旧详细结果或显式--detailed运行。旧报告仍可查看、续算golden，不强制重跑板子。

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
# 从当前源码构建正式板端程序
bash yolo26_riscv/build_board.sh

# 检查已准备的交付固件、冻结模型、节点资料及BIN/ELF一致性
python3 tools/check_board_release.py
```

构建输出位于 `yolo26_riscv/build/n550-board/`，不会自动替换 `firmware/`。交付目录须同步本次BIN、ELF、符号、反汇编和校验清单后再核对；原QEMU及标量实验不是板端固件。
