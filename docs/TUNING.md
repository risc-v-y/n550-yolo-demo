# 性能与检测效果评估

先定位耗时和检测问题，再决定修改。以下程序不会自动调整正式模型、后处理参数或量化方案。

## 性能

在host执行统一入口：

```bash
# 打开菜单；1更新；15加载新增计时固件并运行三帧；18选择该运行，逐页截图
bash tools/host.sh
```

已有视频结果可直接选18，不重新推理或计算golden。固件在DDR保存整帧及334节点计数，完成后一次读取。输出：`performance-summary.json`、`performance-page-1/2/3.txt`、全部节点含六阶段周期的 `node-timing.csv`。

权重准备：94层AMU卷积权重在READY前转换、打包一次，逐帧读取。普通DDR预留5MiB，实际5,033,984字节、2,391,664个有效FP16值。原实现每帧重复转换36,955,840次，预打包后为0；启动准备周期单列。

融合输入准备：对每批≤32个输出位置，直接从NCHW FP32特征图生成FP16 `[ceil(K/32),32位置,32K]` 输入块，边界补零；同批输入在全部M输出通道分块之间复用。沿用原 `amu_half`，保持每个输出K≤32分块的顺序、逐指令舍入及最后独立FP32偏置相加。1×1卷积同样按空间块准备；不同输出块的执行先后允许调整，不改变单个输出的累加。RVV分组/深度卷积保留原路径。原512KiB CLP工作区使用union，前最多256KiB存FP16块，整体布局不变。

已测基线：`run-t5240cg4`模型3,817,107,181周期、记录每帧110.45秒；权重预打包版 `run-ls19m5ja` 为2,862,430,017周期、86.08秒。模型周期下降25.0%；该版窗口提取及输入打包合计约77%，本轮融合的实板收益尚未测量。

执行 **1更新 → 15同一三帧 → 确认VIDEO CHECK PASSED → 18新运行三页**，无需节点采集。18匹配同一父目录下状态completed、全部帧号和输入哈希相同且计数有效的早期版本运行；融合版优先比较v2的 `run-ls19m5ja`，没有合格v2才比较v1。保留这两份基线。基线版本/路径/报告SHA256写入summary，缺匹配项明确提示。周期比包含计时开销，需结合实际Host耗时评估。

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
| input_im2col | 生成FP32窗口矩阵；v3只用于原RVV分组卷积路径，AMU卷积应为0 |
| input_pack | v2为FP32矩阵转FP16；v3包括直接窗口提取、FP16打包、尾部补零及缓存clean发布，每空间块一次 |
| matrix_submit | 矩阵配置/加载/乘加/存储提交及内部缓存维护，不能视为纯指令发射 |
| matrix_wait | 提交后读取xmfflags排空的剩余等待及标志发布；不是AMU总运行时间 |
| bias_dequant | 卷积最后的独立FP32偏置加法；此版本无量化 |
| other_compute | compute减去六阶段之和，含循环、诊断和未归入六阶段的维护 |
| 每节点check | 计算后的有限值检查分支；普通模式仅最终输出扫描 |
| graph_overhead_cycles | 图总周期减去全部节点cache/compute/check之和，含进度日志、诊断提交、记账等 |

计时表写入、读计数器、发布和Host读取有开销。带逐节点trace、中间有限值扫描或节点暂停采集的运行不进入性能排行。启动自测、权重加载不在模型周期内；Host第一帧等待仍可能包含会话初始化。v2/v3的input_pack计时边界不同，比较输入准备时应合并input_im2col、input_pack与内部提交维护，不能只看单项下降。

18询问的是**当前FPGA上N550实际运行频率MHz**，未知直接回车，仅报周期。不能用RTL目标频率或QEMU计数推算板端秒数。Host的PCIe等待包含pbload轮询和工具启动；`elapsed_seconds`含录像保存，但未完整计入最终窗口显示和JSON落盘，不能称为采集到可见延迟。离线golden耗时单列，不算推理帧耗时。

计时表uint64头固定16项：`magic version bytes frame nodes completed options state model_cycles graph_cycles input_check_cycles service_cycles input_crc_cycles output_cycles node_cycles session`。v1共8152字节，随后334组cache/compute/check，末项CRC32存于uint64。v2/v3共24320字节，每节点在三项后追加六阶段周期，再附全局六组cycles/calls、五项权重统计，末项为此前全部字节的CRC32。v3仅改变输入准备方式和阶段含义，布局不变。六阶段顺序与上表相同。五项为weight_pack_bytes、weight_pack_values、weight_pack_cycles、prepared_layers、runtime_weight_conversions。Host兼容三版，核验CRC、会话/帧号、完成数量、阶段总账及权重不变量；离线重建核验SHA256。

开发验证：全部94层的权重及窗口输入打包与独立IEEE binary16转换核对，覆盖边界、跨行、尾块复用、负零和半精度舍入边界。QEMU对11种矩阵形状各复用两组输入，加4种真实卷积窗口/偏置/多批量用例，与旧路径逐位比较。BUS整图使用正式启动和PCIe路径，验证每空间块只打包一次及最终输出；QEMU不验证物理缓存或实际速度，现场仍需同三帧RTL golden和计时。

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
