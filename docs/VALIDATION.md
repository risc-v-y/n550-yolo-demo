# 本次交付验证记录

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
