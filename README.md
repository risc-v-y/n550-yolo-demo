# N550 YOLO26 UART demo

交付目标：开发机采集/预处理 → UART → N550 RVV+AMU FP16 推理 → UART → 开发机后处理/显示。固定416×416、batch=1、COCO80、334节点，输出300×6 FP32候选框。

**当前属于待上板验证的软件交付。编译与本机协议测试通过不等于 UART、缓存或 AMU 硬件已验收。** JTAG连接、下载及调试由软件/FPGA同事负责。

## 先跑起来

1. 从私有仓库 Release 获取 `selftest.elf` 和 `demo.elf`。权重已包含在 `demo.elf`，不用另传 `.pt` 或通过串口加载权重。
2. 按 [板端条件与验收](docs/BOARD.md) 确认 DDR、启动状态和 UART 接线；先由同事下载运行 `selftest.elf`，串口应显示 `UART OK`、`CACHE/RVV/AMU TEST PASS`。
3. 关闭串口终端，由同事重新下载并启动 `demo.elf`；该固件只传二进制协议，不打印文字日志。
4. Windows安装Python 3.12后，在仓库根目录执行（板端模式不需要PyTorch）：

```powershell
py -3.12 -m venv yolo26_pc/.venv-board
& ./yolo26_pc/.venv-board/Scripts/python.exe -m pip install -r yolo26_pc/requirements-board.txt
& ./yolo26_pc/.venv-board/Scripts/python.exe yolo26_pc/uart_backend.py --port COM3 --ping 100
& ./yolo26_pc/.venv-board/Scripts/python.exe yolo26_pc/uart_backend.py --port COM3 --image yolo26_pc/samples/bus.jpg --output yolo26_pc/outputs/board-bus
```

把COM3替换为实际端口。输出目录必须尚不存在，以免覆盖之前的结果。

单图通过后运行摄像头或视频：

```powershell
& ./yolo26_pc/.venv-board/Scripts/python.exe yolo26_pc/live_demo.py --backend uart --serial-port COM3 --camera 0
& ./yolo26_pc/.venv-board/Scripts/python.exe yolo26_pc/live_demo.py --backend uart --serial-port COM3 --source yolo26_pc/samples/pexels-3796613.mp4 --max-results 2
```

左侧持续预览，右侧仅更新已完成推理的对应原图。Q/Esc退出，录像与报告保存在 `yolo26_pc/outputs/live-demo/<时间>/`。按需增加 `--save-frames` 保存逐帧PNG/NPY诊断文件。

115200、8N1传一帧FP32输入的纯数据时间约180秒，分包应答会增加耗时。首版用于功能调试，不验收5 FPS。`--infer-timeout 900` 是RUN命令每次等待上限，不包含上传时间；串口断开明确报错，退出时当前帧可能仍在板端计算。

## 从源码构建

在Linux/WSL解压Release的工具链至仓库 `toolchain/` 下，然后：

```bash
mkdir -p toolchain
tar -xzf eswin-riscv-toolchain-linux-x86_64.tar.gz -C toolchain
bash yolo26_riscv/build_board.sh
# 可选：TOOLCHAIN_PREFIX=/path/to/bin/riscv64-unknown-elf bash yolo26_riscv/build_board.sh
```

输出位于 `yolo26_riscv/build/n550-board/`，包括ELF、map、反汇编、符号表、编译器信息与哈希。必须使用支持 `xewmatrix1p0` 的配套ESWIN工具链；普通RISC-V GCC不能替代。RAM默认独占 `0x80000000–0x81FFFFFF`，栈保留256 KiB。

`model/` 中的常量和生成图是本次固定部署资产，与源码一起提交。更换模型时必须同时重新生成图、常量和数值参考。

## 验证与资料

- [板端约束、验收与故障定位](docs/BOARD.md)
- [UART二进制协议](docs/UART_PROTOCOL.md)
- [软件验证记录](docs/VALIDATION.md)
- `reference/`：bus/zidane的历史QEMU FP16候选框与后处理基准，用于上板核对，不代表本次板端结果。
- 本地PyTorch模式：在仓库根目录执行以下命令。

```powershell
& ./yolo26_pc/setup.ps1
& ./yolo26_pc/.venv/Scripts/python.exe yolo26_pc/live_demo.py --backend pc --source yolo26_pc/samples/pexels-3796613.mp4 --seconds 30
```

本仓库交付UART板端与PC模式；界面保留的QEMU入口需要原开发环境，本交付包不包含QEMU运行环境。

附带视频来源：[Pexels 3796613](https://www.pexels.com/video/people-walking-on-the-street-3796613/)。文件 `yolo26_pc/samples/pexels-3796613.mp4` 的SHA256为 `fcd2af324d05ae09da2570ffd693da084f6afdb209b078b419c30741742b43c5`。

Linux协议测试需要宿主GCC、Python3：`bash yolo26_riscv/test_board_protocol.sh`。该测试编译真实板端解析器，通过管道注入完整帧及异常，不依赖FPGA。
