# 性能与检测效果评估

先定位耗时和检测问题，再决定修改。以下程序不会自动调整正式模型、后处理参数或量化方案。

## 性能

在host执行统一入口：

```bash
# 打开菜单；1更新；15加载新增计时固件并运行三帧；18选择该运行，逐页截图
bash tools/host.sh
```

已有视频结果可直接选18，不重新推理或计算golden。固件在DDR保存整帧及334节点计数，完成后一次读取。输出：`performance-summary.json`、`performance-page-1/2/3.txt`、全部节点含六阶段周期的 `node-timing.csv`。

当前实现、缓存交接与全算子状态统一见 [IMPLEMENTATION](IMPLEMENTATION.md)。本页只维护实测、计时与验收，不重复内核细节。

| 计时版本/实现 | host运行 | 平均模型周期 | 每帧记录耗时 |
|---|---|---:|---:|
| v1 原始逐块准备 | run-t5240cg4 | 3,817,107,181 | 110.4461秒 |
| v2 启动权重预打包 | run-ls19m5ja | 2,862,430,017 | 86.0835秒 |
| v3 标量融合窗口/FP16并复用 | run-llwcyfdf | 2,150,645,371 | 68.1029秒 |
| v4 RVV窗口提取/整数FP16转换 | 待同三帧现场运行 | 未测 | 未测 |

前三版均为3个普通帧、无无效计时记录，数据由现场菜单18提供。v3比v2模型周期减少24.9%、每帧记录耗时减少18秒；比v1累计减少43.7%周期、38.3%记录耗时。计时报告本身不是数值验收报告，各新运行仍须确认VIDEO CHECK PASSED。

v3模型周期分布：input_pack=1,268,354,376（59.0%），input_im2col=244,184,400（11.4%），other_compute=520,756,478（24.2%）。前两项合计相比v2减少31.4%。节点278/274是RVV深度卷积，合计约7.7%模型周期；不把other_compute全部视为可删开销，也不把matrix_wait当作AMU总执行时间。

执行 **1更新 → 15同一三帧 → 确认VIDEO CHECK PASSED → 18新运行三页**，无需节点采集。v4匹配同一父目录下完整、同帧号/输入哈希、计数有效的v3运行，优先 `run-llwcyfdf`；没有合格v3才向前匹配v2/v1。保留三份基线。基线版本/路径/报告SHA256写入summary，缺匹配项明确提示。周期比包含计时开销，需结合Host耗时评估。

| 板端字段 | 计时边界 |
|---|---|
| service_cycles | 接受RUN请求至输出处理完成，尚未做计时表CRC/clean和DONE状态发布；含输入CRC、模型、DONE日志及输出处理 |
| model_cycles | 计时表初始化之后至最后节点结束；含RUN日志、输入有限值扫描、图执行；DONE日志之前冻结 |
| graph_cycles | 输入扫描结束至最后节点结束，包含期间诊断、日志和计时记账 |
| input_check_cycles | 输入有限值扫描，不含输入CRC |
| input_crc_cycles | 输入缓存交接及CRC检查 |
| output_cycles | 模型返回后至输入缓冲区释放、输出CRC/clean及诊断提交结束 |
| 每节点cache | 输入缓存clean及CACHE阶段标记 |
| 每节点compute | `accel_model_node`调用，包含内部AMU等待、数据打包及同步 |
| weight_prepare | 帧内权重转换，预打包版本应为0；不含启动准备 |
| input_im2col | 生成FP32窗口矩阵；v3/v4只用于原RVV分组卷积，AMU卷积应为0 |
| input_pack | v2仅转换/打包；v3含标量窗口及clean；v4含RVV索引元数据准备、gather、整数RNE、补零与scratch所有权交接，每空间块一次 |
| matrix_submit | 矩阵配置/加载/乘加/存储提交及内部缓存维护，不能视为纯指令发射 |
| matrix_wait | 提交后读取xmfflags排空的剩余等待及标志发布；不是AMU总运行时间 |
| bias_dequant | 卷积最后的独立FP32偏置加法；此版本无量化 |
| other_compute | compute减去六阶段之和，含循环、诊断和未归入六阶段的维护 |
| 每节点check | 计算后的有限值检查分支；普通模式仅最终输出扫描 |
| graph_overhead_cycles | 图总周期减去全部节点cache/compute/check之和，含进度日志、诊断提交、记账等 |

计时表写入、读计数器、发布和Host读取有开销。带逐节点trace、中间有限值扫描或节点暂停采集的运行不进入性能排行。启动自测、权重加载不在模型周期内；Host第一帧等待仍可能包含会话初始化。不同版本input_pack边界不同，比较输入准备应合并input_im2col、input_pack与内部维护，不能只看某一项下降。

18询问的是**当前FPGA上N550实际运行频率MHz**，未知直接回车，仅报周期。不能用RTL目标频率或QEMU计数推算板端秒数。Host的PCIe等待包含pbload轮询和工具启动；`elapsed_seconds`含录像保存，但未完整计入最终窗口显示和JSON落盘，不能称为采集到可见延迟。离线golden耗时单列，不算推理帧耗时。

计时表uint64头固定16项：`magic version bytes frame nodes completed options state model_cycles graph_cycles input_check_cycles service_cycles input_crc_cycles output_cycles node_cycles session`。v1共8152字节；v2/v3/v4均24320字节，布局不变。每节点为cache/compute/check及六阶段周期，随后全局六组cycles/calls、五项权重统计，末项uint64承载此前全部字节的CRC32。五项为weight_pack_bytes、weight_pack_values、weight_pack_cycles、prepared_layers、runtime_weight_conversions。Host兼容四版，核验CRC、会话/帧号、完成数量、阶段总账及权重不变量；离线重建核验SHA256。

开发验证：标量打包对独立IEEE binary16；RVV逐位对标量，覆盖随机FP32位型、所有正常FP16相邻值中点附近、不同frm、94层形状、边界/跨行/尾块及复用、异常值。VLEN128/512均验证11种矩阵形状各两组输入及4种真实卷积/偏置/多批量。完整BUS使用正式启动/PCIe路径，7200字节与原QEMU参考逐位一致，5710次输入准备均走RVV；实际C计时表由Python3.8解码。QEMU不证明物理缓存或速度，仍需现场同三帧RTL golden和计时。

## 检测效果

菜单19选择旧运行和已完成帧，默认首/中/末帧。复用该帧候选输出及坐标映射，核对原图能重现提交的输入哈希；不访问PCIe。固定比较四项：

| 名称 | 置信度 | 同类别去重 |
|---|---|---|
| baseline | >0.25 | 无 |
| confidence035 | >0.35 | 无 |
| nms050 | >0.25 | IoU>0.5 |
| confidence035_nms050 | >0.35 | IoU>0.5 |

去重按分数从高到低保留，分数相同按原候选编号决定；不同类别互不抑制，显示仍保留原模型顺序。这些是候选对照参数，未确定为最佳值。提高阈值可能漏掉远处行人，NMS可能误删拥挤场景中真实重叠目标。

输出在 `yolo26_pc/outputs/accuracy-review/review-*/`：

- `comparison-NNNNNN.png`：四格对照；另存每项原尺寸PNG。
- `accuracy-summary.json`：每项完整框、保留/删除候选编号、NMS保留框及IoU。
- `accuracy-report.txt`：便于截图的短报告。
- `labels-template.json`：人工完整标注模板；原图SHA绑定，坐标为原图像素xyxy、class_id为COCO80编号。

模板初始 `reviewed=false, objects=[]`，表示**尚未标注**。完整标注该帧所有目标后置 `reviewed=true`；确实没有目标的已审核空帧可以保留空列表。再次选择19、同一运行和帧号，填写标签文件路径，才会统计IoU≥0.5、同类别一对一匹配的TP/FP/FN、precision、recall、F1；这不是mAP。不完整标签不能评估漏检或误检。

当前原始PyTorch视频首帧对照：基线11框，单独NMS后10框；阈值0.35后8框。NMS去掉一辆车的重复框，但没有修复将行人腿部识别为自行车的问题；提高阈值同时删掉了一个真实远处行人。尚无完整人工标签，不能宣称整体准确率提高。后续先评估代表性画面，再决定是否微调训练或更换模型，并同步评估计算量。
