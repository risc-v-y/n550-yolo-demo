# PCIe 自动逐帧演示

host 为物理连接板卡的 Linux 实验室电脑，Python 3.10。链路为：**host 采集/预处理 → PCIe 写DDR → 板端推理 → PCIe 读结果 → host 后处理/显示/录像**。开发机远程查看 host 桌面。

## 使用步骤

1. 克隆本仓库，在仓库根目录执行 `sha256sum -c firmware/SHA256SUMS`，使用 `firmware/demo-pcie.elf`。如需重新构建，执行 `bash yolo26_riscv/build_board.sh`，新固件位于 `yolo26_riscv/build/n550-board/demo-pcie.elf`。模型图、权重和自测均在ELF内。由同事通过现有工具加载ELF并启动一次，保持程序运行。不要运行另一个仓库会复位和重写权重的 `run_yolo26n.sh`。`firmware/demo.elf` 仍用于UART，两个固件二选一。
2. 按 README 准备 `.venv-host` 和 `requirements-board.txt`。host 必须已有可执行的 `pbcopy`、`pbload` 和配套驱动、访问权限。我们直接调用命令，不安装内核驱动、不控制复位。非PATH内的命令通过 `--pbcopy /绝对路径/pbcopy --pbload /绝对路径/pbload` 指定。
3. 在 host 仓库根目录执行：

```bash
# 握手并读取固件接口，尚不运行图像推理
.venv-host/bin/python yolo26_pc/pcie_backend.py

# 先验单图，原图、候选框、绘框图和差异报告自动保存
.venv-host/bin/python yolo26_pc/pcie_backend.py --image yolo26_pc/samples/bus.jpg --reference reference/bus-qemu-fp16.bin
.venv-host/bin/python yolo26_pc/pcie_backend.py --image yolo26_pc/samples/zidane.jpg --reference reference/zidane-qemu-fp16.bin

# 自动循环视频；先收两帧验证收发与录像
.venv-host/bin/python yolo26_pc/live_demo.py --backend pcie --source yolo26_pc/samples/pexels-3796613.mp4 --max-results 2 --save-frames

# 连续演示：去掉 --max-results 2。host 本地摄像头用 --camera 0 替换 --source 参数
```

等待期间左侧继续预览，右侧每收到一帧结果才更新。Q/Esc或关闭窗口退出。无检测时正常显示原图。视频来源循环播放，推理选取当前最新帧；首版每次只有一帧在途，不保证处理素材的每一帧。

单图结果在 `yolo26_pc/outputs/board-pcie/<时间>/`；视频在 `yolo26_pc/outputs/live-demo/<时间>/`，含 `raw.mp4`、`demo.mp4`、`report.json` 和 `diagnostics.jsonl`。`--save-frames` 额外保存原图、候选框、绘框图和逐帧JSON。录像沿用现有20 FPS录制方式，报告中的实际结果帧率才是推理链路速度。

## 传输与同步约定

- 已确认：CPU运行期间可访问DDR，`pbcopy`成功返回表示写入已到DDR；地址换算为 `PCIe地址=CPU地址−0x80000000`。平台说明目前无额外地址对齐/长度限制。
- 写命令：`pbcopy -d 文件:PCIe地址`；读命令：`pbload -d 输出文件:PCIe地址:字节数`。以参数数组调用，不经Shell。文件是平台工具的暂存接口，由host自动创建和清理。
- 每帧输入保持RGB、CHW、416×416、FP32，共2,076,672字节；输出 `[300,6]` FP32，共7,200字节。保持全部334节点和现有FP16 AMU/RVV实现，不改变模型和后处理。
- 仅一个host控制程序；本机文件锁防止重复启动。其他PCIe工具、另一账户或其他机器不能同时改写这些DDR区域。

固定接口位于现有32 MiB预留DDR内部，不使用另一个仓库的输入、权重和结果区：

| 区域 | CPU地址 | PCIe地址 | 字节数/写入方 |
|---|---|---|---|
| 固件描述 | `0x81F00000` | `0x01F00000` | 64，板端 |
| 命令正文 | `0x81F00040` | `0x01F00040` | 64，host |
| 提交序号 | `0x81F00080` | `0x01F00080` | 64，host，仅首个uint32有效 |
| 处理状态 | `0x81F000C0` | `0x01F000C0` | 64，板端 |

输入、输出和816字节诊断区的CPU地址由固件描述发布；host验证尺寸、范围及64字节对齐。链接脚本禁止模型BSS侵入接口区，主栈仍保留在DDR顶端256 KiB。板端使用不同缓存行保存host写区和板端写区，避免缓存维护覆盖另一方数据。

接口为小端、版本1，描述和状态magic为 `Y26P`；描述、命令、状态均为前60字节CRC32存于最后4字节，布局见 `board/pcie.h`。

1. 板端初始化接口、自测，flush输入区后发布READY。host提交OPEN，以随机64位session建立会话；帧号为64位，命令序号为非零32位。
2. host写完输入，再写带长度/CRC的RUN正文，逐字节读回确认正文，最后写提交序号。正文未验证成功不提交，输入错误由板端CRC拦截。
3. 板端invalidate提交区及正文；相同提交序号不重复执行。核对会话、长度、输入CRC后运行模型。
4. 板端clean结果，flush即将交还host的输入缓存行，完成fence后发布DONE、帧号、结果长度和CRC。host读结果并核对CRC/有限值/类别，以及读取前后状态一致；完成后才提交下一帧。中间张量有限值检查可用 `--check-intermediates` 开启。

状态：0 BOOT、1 READY、2 BUSY、3 DONE、4 ERROR。新增错误：`0x40`命令CRC/格式、`0x41`会话、`0x42`输入CRC、`0x43`模型执行失败；模型详细错误仍在 `board_diag`。输入/输出可在模型arena内复用内存，host不得在BUSY时写输入，也不得在下载旧结果前开始写下一帧。

## 失败与复验

`--infer-timeout` 默认900秒，覆盖单帧传输和推理；`--tool-timeout` 默认30秒限制一次工具调用；`--poll-interval` 默认0.1秒。轮询开销和实际吞吐待板上测量，不预先承诺5 FPS。

工具非零退出立即报错，退出0仍检查实际读回长度、命令读回、输入/结果CRC与帧号，不只相信日志中的success。状态CRC不匹配可能是发布中读到半条记录，在总超时内重读。命令、退出码、工具输出尾部、各阶段耗时和失败信息均写入日志；PCIe暂不提供UART的逐节点事件流，完成/失败时读取固定诊断快照。

取消/超时停止host收发，不复位板子，也不能中断已经开始的推理。同一个失败会话不能再提交帧；重新连接会先等旧命令结束。若内核驱动调用本身不可中断，或板端始终BUSY，由同事检查/恢复平台后再启动。板端trap尽力发布ERROR和诊断；无法返回的总线/AMU指令只能靠host超时及同事调试器定位。

本机回归命令：`HOST_PYTHON=/path/to/python bash yolo26_riscv/test_board_pcie.sh`。测试使用真实板端协议C代码、模拟DDR和可执行的模拟pbcopy/pbload，推理是测试替身。它能检查协议、错误分支和视频软件流程，不能替代真实PCIe、DCache、RVV/AMU及模型数值验收。
