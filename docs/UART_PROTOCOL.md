# UART协议 v1

物理串口采用115200、8N1，无硬件/软件流控；每个方向只有一个未应答请求。PC是请求方，板端仅响应，不依赖启动时的一次性欢迎消息。

## 包格式

全部整数和浮点采用小端。32字节头对应Python `<4sHHIIIIII>`：

| 字段 | 字节数 | 含义 |
|---|---:|---|
| magic | 4 | ASCII `Y26B` |
| version/kind | 2+2 | 版本1、消息类型 |
| seq/frame/offset | 4+4+4 | 请求号、帧号、当前数据块字节偏移 |
| length | 4 | 后续payload长度，0–1024 |
| payload_crc | 4 | payload的CRC32 |
| header_crc | 4 | 前28字节头的CRC32 |

CRC使用IEEE反射多项式0xEDB88320，初值及末尾异或均为0xFFFFFFFF，与 `zlib.crc32` 一致。空payload CRC为0。响应回显seq/frame/offset，头和payload各自校验。

## 消息及顺序

| 类型值 | 请求→响应 | 数据与状态 |
|---:|---|---|
| 1→2 | HELLO→INFO | 请求空；响应6个uint32：输入字节数、输出字节数、节点数、实际baud、DLF位宽、cacheline字节数。重置帧接收状态 |
| 10→11 | PING→PONG | 回显0–1024字节 |
| 3→6 | BEGIN→ACK | 空payload，开始指定帧并清接收计数 |
| 4→6 | DATA→ACK | 从offset=0连续上传，后续offset必须等于已接收长度 |
| 5→6 | RUN→ACK | 仅完整帧可执行；推理完成才应答 |
| 7→8 | GET→RESULT | 空请求、指定offset；响应剩余数据最多1024字节 |
| 12→6 | STOP→ACK | 放弃帧状态，不退出固件、不硬件复位 |
| 任意→9 | ERROR | 4字节错误码：1无活动帧/帧号不匹配；2命令/状态/偏移/长度错误；3模型执行失败 |

输入固定RGB/NCHW FP32 `[1,3,416,416]`，2,076,672字节；PC执行缩放补边和除255。输出固定300×6 FP32，7200字节，字段 `x1,y1,x2,y2,score,class`，坐标在416输入图上。首版不做量化传输，不添加NMS。

## 重试与恢复

PC按精确字节数读取，允许任意分片。通常每次请求超时3秒，RUN每次默认900秒，最多3次尝试；重试必须保留原请求号和内容。收到ERROR立即失败，不将板端逻辑错误当链路超时重试。

板端记住最后一个请求及应答；完全相同的重复请求重发缓存应答，不重复拷贝或推理。CRC/头格式错误丢弃并等待下一包，PC在超时后重发。旧seq响应被PC跳过。新帧只有上一帧完成后才发送，取消后重新HELLO；板端计算期间不接收取消命令，须等其完成或由同事重新启动固件。

协议只处理帧流和诊断，不承载权重、任意内存访问或调试器命令。

## 诊断扩展（2026-09）

原 v1 包头及 INFO 24 字节保持不变。类型 13 `DIAG_GET`（空请求）返回类型 14 `DIAG`；类型 15 `DIAG_CONFIG`（4 字节 uint32 位掩码）返回 ACK。配置 bit0 开启事件、bit1 开启逐节点、bit2 开启中间有限值检查，其他位拒绝。新 HELLO 将选项清零；旧 host 不配置时不会收到额外消息。

开启事件后，同一请求的最终 ACK/ERROR 之前可出现 DIAG，回显当前请求 seq/frame/offset。host 消费这些消息后继续等待最终应答，总超时不因收到诊断消息重置。重复 RUN 只重发应答及当前快照，不重复推理；重发 DIAG_GET 可能返回该请求缓存的快照。DIAG_GET 不改变模型帧状态；同步 RUN 执行中不能并发查询。

DIAG 固定 816 字节，为 102 个 little-endian uint64。前 28 项依次是：

```
version stage frame node op phase error detail
received rx_errors rx_timeouts tx_errors crc_errors protocol_errors retries
amu_flags tensor element observed expected
trap_cause trap_pc trap_value node_start node_cycles inputs outputs options
```

随后 6 组各 7 项：`tensor_id,dtype,rank,shape[4]`，前四组输入、后两组输出；最后 32 项为 x0–x31。无有效节点/张量时为 UINT64_MAX，detail 按 int64 二补码解释。诊断版本当前为 1；具体枚举以 `board/diagnostics.h`、`board_diagnostics.py` 为准。CPU trap 不保证能发出 UART 消息，内存记录的限制见 [诊断说明](DIAGNOSTICS.md)。
