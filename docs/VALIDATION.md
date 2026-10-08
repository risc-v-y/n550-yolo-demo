# 本次交付验证记录

2026-10-08 交付文件核对：

- PCIe/UART源码、操作文档和预编译固件已进入仓库；软件同事从 `firmware/` 取本次固件。仓库根目录按 `firmware/SHA256SUMS` 核对三个ELF、PCIe BIN、符号/反汇编/map、编译器记录与模型常量，共9项哈希通过。此项只验证文件一致性，不证明实板运行通过。
- PCIe现场加载文件改为 `firmware/demo-pcie.bin`。从对应ELF唯一有文件内容的 `PT_LOAD` 段生成，入口及段起址均为CPU `0x80000000`，文件长度9,755,804字节，PCIe写入偏移 `0x0`；BSS与邮箱是未初始化区域，不在BIN内。构建脚本此后使用工具链的 `objcopy -O binary` 生成BIN。尚未用真实S2C加载该文件，也未验证板端启动。

2026-09-30 PCIe自动逐帧收发：

- 新增独立 `demo-pcie.elf`，模型图与权重不变；固定接口地址 `0x81F00000`，描述/命令/提交/状态各64字节，输入/输出地址由固件发布。三个固件均已交叉构建，尺寸和符号以 `pcie-size.txt`、`pcie-symbols.txt` 为准。
- `test_board_pcie.sh` 的11项回归通过：实际板端协议C代码配合模拟DDR及可执行的模拟pbcopy/pbload；包括连续三帧、重复/旧提交不重算、输入CRC、输出CRC、错误会话/帧号、状态半读、非法类别/NaN、模型错误、工具失败/短读、取消、超时重连、单控制进程锁。
- 同一回归实际调用 `live_demo.py --backend pcie --no-display` 完成两帧，检查帧号、原图、绘框图、候选框和可解码的 `demo.mp4`。使用合成视频与测试替身推理，不是YOLO检测效果或板端吞吐证明。
- Linux回归使用Python3.12与固定host依赖；Windows Python3.10的语法检查及两个CLI帮助入口通过。真实host Python3.10/Linux GUI和实际PCIe工具仍待现场验收。
- 原UART协议7项回归全部通过；保留的UART模型固件与独立自测均重新构建成功。
- 日志：`yolo26_riscv/build/n550-board/pcie-tests.log`。实板的PCIe传输、缓存可见性、模型单图数值与连续视频仍未验证；本地功能回归不能替代上板验收。旧GitHub v0.2.0固件不含本功能；本次固件以仓库 `firmware/` 为准。

2026-09-30 RVV/AMU依赖与缓冲区复用自测：

- 用户确认支持标准CBO和FP16 AMU指令；未改用其他项目的整缓存清理方式。
- 两个板端ELF构建通过（`-Wall -Wextra -Werror`），demo的text+data+bss为17,909,764字节，另预留256 KiB主栈；包含新增自测。
- `bash yolo26_riscv/test_board_sync.sh` 九组回归通过：VLEN 128/256/512各运行32轮正常用例，以及第13轮RVV结果破坏、第7轮AMU输出边界破坏。故障均匹配预期返回码、轮次、缓冲区和元素位置。
- 用例直接执行RVV写FP16→AMU读/计算/写FP32→RVV读/加标记；反复切换四种矩阵尺寸并复用相同缓冲区，整数参考逐位比较，检查全部边界和行填充。
- 本地构建记录在 `yolo26_riscv/build/n550-board/`：`sync-tests.log`、`selftest.disasm`、两个ELF及`sha256.txt`。旧v0.2.0 Release不包含本次新增自测。
- QEMU仅验证计算、控制流和失败定位；实板DCache一致性、fence及读取xmfflags的完成语义仍待开启DCache后运行新版自测确认。尚未执行实板测试。

2026-09-24 诊断版本新增验证：

- `demo.elf`、`selftest.elf` 编译通过，静态占用（demo 的 text+data+bss）17,904,408 字节，另保留 256 KiB 主栈；包含独立 4 KiB 异常栈。
- 实际 C 协议解析器与 Python 客户端共七项测试通过：含诊断快照、CRC 计数、错误节点/有符号返回码、逐节点形状、丢 ACK 后重试不重复推理。
- Windows 上独立 Python 3.10.21 环境安装 host 依赖成功；bus/zidane 后处理与原基准逐字节一致；视频哈希、CLI、串口打开失败的单图/视频入口报告及完整 traceback 保存通过。Linux 原生协议测试使用 Python 3.12。
- 板端真实 trap、AMU 卡住时的诊断可见性、实验室 Python 3.10/Linux GUI 及真实 UART 仍待同事验收；没有将宿主仿真当作实板验证。

以下为首次交付基线记录，尺寸和测试数量以对应版本为准。

日期：2026-09-21。此记录区分本机验证与板端验收，不将编译成功视为硬件运行通过。

| 检查 | 结果 |
|---|---|
| ESWIN GCC 14.1.1编译完整板端demo.elf | 通过，`-Wall -Wextra -Werror`，RVV+AMU+Zicbom |
| 独立UART/cache/RVV/AMU selftest.elf编译 | 通过；尚未在真实板上执行 |
| DDR静态占用 | 17,896,860字节（text+data+bss），另保留256 KiB栈；32 MiB链接检查通过 |
| 反汇编 | 包含cbo.clean/flush/inval与mfmacc.s.h；无QEMU semihosting调用/ebreak |
| 真实C协议解析器＋Python客户端 | 四组测试通过，宿主管道仿真，不代表UART电气或寄存器验证 |
| 分片、CRC与历史响应 | 通过：短读/短写、坏头重同步、响应CRC重试、跳过旧响应 |
| 完整FP32帧和重复RUN | 2,076,672字节上传及7200字节回读通过；丢弃RUN应答后重试，模拟推理只调用一次 |
| 帧号/偏移/取消 | 错误被拒绝；取消及时退出 |
| PC后处理回归 | bus 6个框、zidane 2个框，与固定QEMU FP16后处理基准逐字节一致 |
| 模型资产 | 固定图/常量SHA256与长度校验通过 |
| Python串口、演示CLI | 语法及help入口通过，pyserial 3.5固定 |
| 独立交付目录 | 从Git暂存内容导出后，模型哈希、两个ELF构建、四组协议测试及bus/zidane后处理复验通过 |
| 独立目录PC视频模式 | PyTorch实际处理两帧，无界面运行、生成录像与报告；不是实板测试 |

尚待同事上板验收：真实UART寄存器/波特率与连续传输、DCache开启下的RVV/AMU可见性、AMU能力和完整334节点数值、异常时调试信息、持续视频运行。

编译保留可读写可执行的统一裸机DDR段，因此链接器报告RWX段提示；本轮没有操作系统页权限配置。板级启动与JTAG下载的缓存一致性由调试交接条件保证。

复验命令：

```bash
python3 tools/check_assets.py
bash yolo26_riscv/build_board.sh
bash yolo26_riscv/test_board_protocol.sh
```

在已安装host依赖的Python环境执行 `python yolo26_pc/check_board_reference.py`。
