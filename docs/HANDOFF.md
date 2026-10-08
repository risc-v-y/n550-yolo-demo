# Codex 项目上下文交接

更新：2026-10-08。用于另一设备的新会话恢复项目背景；不是完整聊天记录。先读本文件，再读README和相关源码，以实际文件与测试记录为准。

## 位置与版本状态

- 原开发机目录：`D:\Document\work\YOLO移植`；新设备可使用其他路径。
- 私有仓库：`https://github.com/risc-v-y/n550-yolo-demo`。本次板端验证需要的源码、文档和预编译ELF已纳入main；以 `git log -1` 核对实际提交。
- **克隆仓库即可取得本次PCIe实现、同步自测和交接文档。** 配套固件在 `firmware/`，先运行 `sha256sum -c firmware/SHA256SUMS`。旧跨设备补充包仅用于保存其他本机材料，不是板端验证的前置条件。
- 旧Release `v0.2.0-host-diagnostics` 不包含本次PCIe功能或32轮同步自测。配套ESWIN工具链仍可从旧Release获取；虚拟环境与运行输出不在Git内。
- 仓库中还有未跟踪的早期开发材料。不要 `git add .`、`git clean` 或恢复所有修改；用户要求GitHub只提交板端验证需要的材料。若要发布，先明确待提交清单，保持仓库私有。

## 用户目标与约束

- 把YOLO26移植到N550，保证演示效果。当前首要任务是功能跑通；算子优化和量化评估暂停，保留现有RVV+AMU FP16实现。
- 所有算子独立实现，不参考Qwen3；不另维护纯RVV中间产品。不新增NMS、跟踪等任务。
- 固定416×416、batch=1、COCO80、全部334节点（含框解码/Top-K）。最终期望至少5帧/秒，但当前QEMU和软件闭环不按5 FPS验收，不能把模拟速度称为实板速度。
- 软件负责模型与host收发/显示。FPGA硬件、DDR初始化、J-Link/JTAG调试、程序加载与启动由软件/FPGA同事负责。
- 用户要求：每次回复以“主人”开头，中文简洁；写代码前说明方案并等批准；每批修改最多三个文件；需求不清晰先问。
- 最近PCIe实现方案已经获得批准并完成本机验证。用户随后要求更新README，已完成；本文件是跨设备交接补充。

## 当前链路与平台事实

host = 物理连接S2C板卡的Linux实验室电脑，Python3.10。开发机通过ETX登录服务器，再SSH到host，在其图形会话中看演示。网络转发未获准实施，勿自行开通。

当前优先链路：**host读视频/预处理 → PCIe写DDR → N550推理 → PCIe读结果 → host后处理/显示/录像**。开发机摄像头跨机器转发尚未接入；host本地摄像头可用现有入口。

- RV64、RVV、ESWIN自定义AMU `xewmatrix1p0`。用户确认支持FP16 AMU和标准 `cbo.clean/flush/inval`，DCache行64字节。
- RVV/AMU经DDR访问与标量DCache需软件交接。CBO维护缓存，fence排序，AMU完成等待使用读取 `xmfflags` 的现有N550约定；实板完成语义仍待验证。
- 本demo独占CPU地址 `0x80000000–0x81FFFFFF` 的32 MiB。其他仓库与本项目同平台，PCIe地址为CPU地址减 `0x80000000`，但两项目软件缓冲区地址不能混用。
- 用户确认CPU运行期间可通过PCIe访问DDR；`pbcopy`成功返回表示数据已到DDR、CPU可读；目前无额外对齐/长度限制。
- 写：`pbcopy -d 文件:PCIe地址`；读：`pbload -d 文件:PCIe地址:字节数`。截图中的外层Shell脚本用 `$? == 0` 判断成功；工具下层可能是内核驱动，没有详细错误码约定。用户提供的截图为根目录 `写命令.png`、`读命令.png`。
- 外层写脚本会复位、重载程序/权重，不能逐帧调用。我们直接调用底层命令、校验返回码和数据，暂存文件由host自动管理。
- 保留UART：DW_apb_uart，基址 `0x20100000`，APB32、寄存器步长4，pclk/sclk10MHz，115200 8N1，无流控，小数分频已确认。PCIe模型不初始化UART。
- CLP映射曾确认256 MiB：`0x10000000–0x1FFFFFFF` → NoC `0xF0000000–0xFFFFFFFF`，当前demo未使用该路径。

## 已完成的软件

1. 原图/视频/QEMU参考、模型C图、标量与RVV/AMU算子、UART收发和诊断既有实现保留。完整输入为RGB CHW FP32（概念shape `[1,3,416,416]`），共2,076,672字节；等比例缩放补边后除以255。结果 `[300,6]` FP32，共7,200字节，字段为 `x1,y1,x2,y2,score,class`；host过滤 `score>0.25`、还原裁剪坐标并绘框。
2. 新增 `board/sync_selftest.c`：RVV直接写FP16→AMU计算/写FP32→RVV读并加标记；32轮相同缓冲区复用，切换四种矩阵尺寸，精确整数参考逐位检查结果及哨兵。失败记录轮次、缓冲区、元素和原始位值。
3. 新增PCIe固件：单host、单帧在途，模型/权重常驻。固定256字节接口位于CPU `0x81F00000`，四个独立缓存行分别为描述、命令、提交序号、状态。输入/输出/诊断地址动态发布；CRC、session、帧号、提交序号及缓存交接见 `docs/PCIE.md`。
4. host已加入 `PCIeBackend`，直接调用 `pbcopy/pbload`，包含本机单进程锁、有限等待、取消、长度/CRC/帧号/类别检查。错误会话不能继续写下一帧；取消不复位板卡，也不中断已开始推理。
5. `live_demo.py --backend pcie` 接入已有异步预览与检测画面、录像和逐帧保存。原pc/qemu/uart入口保留。README现已以PCIe为主要操作路径，UART集中在第6节。

## 文件导航（本次新增文件也必须同步）

| 目的 | 文件 |
|---|---|
| 用户统一入口 | `README.md` |
| PCIe接口/限制 | `docs/PCIE.md` |
| 平台交接/诊断/证据 | `docs/BOARD.md`、`docs/DIAGNOSTICS.md`、`docs/VALIDATION.md` |
| 板端PCIe协议 | `yolo26_riscv/board/pcie.h`、`pcie.c` |
| 板端接入与链接 | `yolo26_riscv/board/main.c`、`ddr.ld`、`yolo26_riscv/build_board.sh` |
| 同步自测 | `yolo26_riscv/board/sync_selftest.c`、`selftest.c`、`yolo26_riscv/test_board_sync.sh` |
| host PCIe与显示 | `yolo26_pc/pcie_backend.py`、`live_demo.py` |
| PCIe联调测试 | `yolo26_riscv/board/tests/pcie_host.c`、`yolo26_riscv/test_board_pcie.sh`、`yolo26_pc/test_pcie_protocol.py` |
| 预编译固件及哈希 | `firmware/` |
| 固定模型与基准 | `model/`、`reference/`、`yolo26_pc/samples/` |

根目录 `bringup.md` 来自另一个仓库，是平台/操作参考，不是本项目权威接口，且不应未经确认提交到本仓库。它描述另一套量化模型和PCIe批处理，其结果槽为4,864字节，与我们7,200字节不同。

## 验证状态

- ESWIN GCC14.1.1构建 `selftest.elf`、UART `demo.elf`、PCIe `demo-pcie.elf` 均通过；输出在 `yolo26_riscv/build/n550-board/`，有map、symbols、disasm和sha256。
- 新同步自测：QEMU的VLEN128/256/512各覆盖正常32轮、结果故障、边界故障，共9组通过。不能证明真实非一致DCache行为。
- PCIe：11项本机联调通过，运行实际板端协议C代码、模拟DDR及模拟可执行pbcopy/pbload，模型为测试替身。覆盖连续帧、重复/旧提交、CRC、非法类别/NaN、会话/帧号、模型错误、短读、工具失败、取消、超时重连及单进程锁。
- 视频联调实际运行无窗口CLI两帧，检查原图、检测图、候选框和可解码录像；并非实际YOLO/实板吞吐验证。
- 原UART协议7项回归通过。host代码在Windows Python3.10通过语法和CLI帮助检查；Linux联调使用Python3.12。
- **尚未在真实板卡执行本次代码。** 下一步是同事加载新固件，依README做PCIe握手、bus/zidane与历史参考比较、两帧视频，再持续运行并记录实际耗时。

## 在新设备继续

1. 克隆main，在项目根目录开新Codex会话；仓库内已有板端验证所需代码、模型资产、文档和固件。
2. 先 `git status --short` 确认版本与修改，再运行 `sha256sum -c firmware/SHA256SUMS`。读取本文件、README、PCIE、VALIDATION。
3. 依任务配置环境：host依赖为 `yolo26_pc/requirements-board.txt`；构建需Linux/WSL及配套ESWIN工具链，可通过 `TOOLCHAIN_PREFIX` 指定。原机有Ubuntu-24.04 WSL及项目内 `toolchain/`，这些不会自动出现在新设备。
4. 复验命令：`bash yolo26_riscv/build_board.sh`；`HOST_PYTHON=/path/to/python bash yolo26_riscv/test_board_pcie.sh`；`bash yolo26_riscv/test_board_protocol.sh`。同步自测另用 `test_board_sync.sh`，需支持AMU/Zicbom的定制QEMU。
5. 后续提交GitHub时只暂存板端交付相关文件；更新版本说明和交接记录。使用 `firmware/` 下的本次ELF，不能拿旧Release的ELF代替。

可粘贴给新会话：

> 请先阅读 docs/HANDOFF.md、README.md、docs/PCIE.md 和 docs/VALIDATION.md，再检查 git status。恢复本项目背景，说明已完成、实板待验证项和当前未提交修改。每次回复以“主人”开头；修改代码前先给方案等确认，每批最多三个文件。不要重做已完成的移植、恢复优化或把本机模拟测试当成实板通过。
