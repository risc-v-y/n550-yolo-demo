# Linux 串口查找与通信验证

在**物理连接开发板的实验室电脑**运行。兼容 Python 3.10 及以上，仅用标准库；不需要安装 PyTorch、OpenCV 或 pyserial。网络链路、J-Link 和 PCIe 加载不在本工具范围内。

以下命令在仓库根目录执行；探测脚本与 UART 后端均位于 `yolo26_pc/`。

## 1. 查找候选串口

```bash
python3 yolo26_pc/serial_probe.py
python3 yolo26_pc/serial_probe.py --json > serial-ports.json
```

默认只读取 Linux sysfs、设备属性和可见的进程信息，**不打开串口、不发送数据、不改串口配置**。输出包括：

- 设备名，例如 `/dev/ttyUSB0`、`/dev/ttyACM0`、`/dev/ttyS0`。
- USB 厂商/产品 ID、型号、序列号，以及 `/dev/serial/by-id/` 等稳定路径（如果存在）。
- 当前用户读写权限、设备属主与属组编号。
- 可见的占用进程 PID/名称，以及串口锁文件。

无占用记录不证明串口空闲：普通用户可能看不到其他用户进程；检查与打开之间也存在时间差。锁文件可能残留，工具会报告并拒绝探测，不自动删除锁或终止进程。没有候选设备时检查 USB 连接和驱动；列出的 UART 也不一定连接了开发板。可在无任务运行时比较插拔 USB 串口前后的列表，并结合适配器序列号定位。

## 2. 指定一个串口验证

请同事先启动 **`demo.elf`**，让固件完成自检并进入协议等待状态。`selftest.elf` 只输出自检文字，不响应本工具的 HELLO。

关闭同一串口的 PuTTY、minicom 和其他 demo，然后执行：

```bash
python3 yolo26_pc/serial_probe.py --probe /dev/ttyUSB0
# 也可使用第一步查到的 /dev/serial/by-id/... 路径
python3 yolo26_pc/serial_probe.py --probe /dev/ttyUSB0 --json > serial-probe.json
```

默认 115200、8N1、无流控；`--baud` 可指定 Linux 支持的标准波特率，但必须与板端一致。只探测指定设备，不向所有串口轮发数据。工具设置独占打开保护，结束时恢复原串口参数并关闭设备；该保护不能赶走已经打开设备的程序，所以仍需先关闭终端。

探测只发送现有协议的 HELLO，复用 `uart_backend.py` 校验版本、CRC、请求编号和 INFO 内容；每次等待 3 秒，最多 3 次尝试。**HELLO 会重置固件的帧接收状态，应在没有图像任务运行时使用。** 不发送图片，不启动推理，不加载模型。

结果含义：

| 结果 | 含义与下一步 |
|---|---|
| `status: passed` | 收到合法 INFO，输入 2,076,672 字节、输出 7200 字节、334 节点、64 字节 cacheline，且 UART 参数有效；串口与目标固件的基础通信通过 |
| 无读写权限 | 请管理员为当前用户配置该串口的访问权限 |
| 存在占用或锁文件 | 关闭相关程序，由同事确认锁状态后再试 |
| 超时 | 检查设备是否选对、固件是否在等待、接线与波特率；不能仅凭超时断定串口损坏 |
| INFO 不兼容/协议错误 | 保存报告，核对固件版本及串口参数 |

退出码：成功或正常列举为 0，探测失败为 1，参数错误为 2，Ctrl+C 取消为 130。基础握手通过不代表全模型推理或持续传输已经通过。

通信失败时保留 `serial-ports.json`、指定串口的 `serial-probe.json` 及固件哈希，结合 [诊断说明](DIAGNOSTICS.md) 定位。当前验证范围与实板待验项目见 [交付与验收状态](VALIDATION.md)。
