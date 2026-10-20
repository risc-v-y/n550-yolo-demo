# 性能与检测效果评估

先定位耗时和检测问题，再决定修改。以下程序不会自动调整正式模型、后处理参数或量化方案。

## 性能

操作步骤只在[OPERATIONS](OPERATIONS.md#2-视频运行与结果)维护；菜单18离线读取已保存结果，不访问板子。新版本先确认[数值验收](VALIDATION.md#视频检查的判据)通过，再比较性能。

固件在DDR保存整帧及334节点计数，完成后一次读取。输出：`performance-summary.json`、`performance-page-1/2/3.txt`、全部节点含六阶段周期的 `node-timing.csv`。

当前实现、缓存交接与全算子状态统一见 [IMPLEMENTATION](IMPLEMENTATION.md)。本页只维护实测、计时边界及检测效果评估，不重复内核细节或数值测试。

| 计时版本/实现 | host运行 | 平均模型周期 | 每帧记录耗时 |
|---|---|---:|---:|
| v1 原始逐块准备 | run-t5240cg4 | 3,817,107,181 | 110.4461秒 |
| v2 启动权重预打包 | run-ls19m5ja | 2,862,430,017 | 86.0835秒 |
| v3 标量融合窗口/FP16并复用 | run-llwcyfdf | 2,150,645,371 | 68.1029秒 |
| v4 RVV窗口提取/整数FP16转换 | run-sup5jvd9 | 1,137,382,966 | 43.2521秒 |
| v5 RVV直接3×3深度卷积 | run-2i3ciq3m | 884,603,103 | 37.0487秒 |
| v6 合并RVV/输入准备/选择与CRC | 待同三帧现场运行 | 未测 | 未测 |

前五版均为3个普通帧、无无效计时记录，数据由现场菜单18提供。v3比v2模型周期减少24.9%、每帧记录耗时减少18秒；比v1累计减少43.7%周期、38.3%记录耗时。计时报告本身不是数值验收报告，各新运行仍须确认VIDEO CHECK PASSED。历次优化原理见[IMPLEMENTATION](IMPLEMENTATION.md#已实施优化的原理)。

v3模型周期分布：input_pack=1,268,354,376（59.0%），input_im2col=244,184,400（11.4%），other_compute=520,756,478（24.2%）。前两项合计相比v2减少31.4%。节点278/274是RVV深度卷积，合计约7.7%模型周期；不把other_compute全部视为可删开销，也不把matrix_wait当作AMU总执行时间。

v4匹配同一父目录下完整、同帧号/输入哈希、计数有效的v3运行，优先 `run-llwcyfdf`；没有合格v3才向前匹配v2/v1。保留三份基线。基线版本/路径/报告SHA256写入summary，缺匹配项明确提示。周期比包含计时开销，需结合Host耗时评估。

### v4现场结果：run-sup5jvd9

来源为用户提供的菜单18三页文本，3帧均有有效计数，无trace/finite/capture排除帧；报告自动选中v3 `run-llwcyfdf`。本次提供的是性能页，未包含同次VIDEO CHECK结果，**数值通过状态仍待核对旧报告，不要求重跑**。

| 对比项 | v3 | v4 | 降幅 |
|---|---:|---:|---:|
| model_cycles | 2,150,645,371 | 1,137,382,966 | 47.1%（1.891倍） |
| input_pack | 1,268,354,376 | 255,910,412 | 79.8% |
| input_im2col | 244,184,400 | 244,017,709 | 0.07% |
| other_compute | 520,756,478 | 520,308,378 | 0.09% |
| 每帧记录耗时 | 68.1029秒 | 43.2521秒 | 36.5% |
| run_roundtrip_seconds | 56.0150秒 | 31.2231秒 | 44.3% |

相对v1累计模型周期下降70.2%，记录耗时下降60.8%。本轮模型少1,013,262,405周期，其中input_pack少1,012,443,964周期；其余主要阶段基本不变，下降与本次输入准备优化对应。other_compute占比升至45.7%是分母变小，绝对周期并未增加。

v4的input_im2col占模型21.5%，input_pack占22.5%；节点278/274当时为原FP32深度卷积，合计14.6%模型周期。该分布促成v5直接深度卷积实现，现场结果见下一节；随后按完整节点表评估Add、Permute和Nearest，不将通用other_compute视为单一开销。

本次热点（节点总周期，包含节点cache）：

| 节点 | 操作 | 平均周期 |
|---:|---|---:|
| 278 | 深度卷积 | 92,271,919 |
| 274 | 深度卷积 | 73,812,251 |
| 12 | Add | 37,286,158 |
| 284 | 深度卷积 | 36,801,486 |
| 314 | Permute | 30,977,720 |
| 155 | Nearest | 28,133,018 |
| 0 | 普通卷积 | 25,087,616 |
| 2 | 普通卷积 | 23,237,520 |
| 288 | 深度卷积 | 23,000,503 |
| 16 | 普通卷积 | 20,123,886 |

其余阶段：service=1,224,149,968、graph=1,131,299,461、matrix_submit=23,382,057、matrix_wait=2,812,270、bias_dequant=48,580,575、boundary_cache=34,210,111周期。启动权重准备74,873,508周期，94层/5,033,984字节/2,391,664个有效值；帧内转换仍为0。

Host平均upload=4.7873、download=1.8878、profile_read=1.8885、prepare=0.0176、postprocess=0.0037秒；elapsed最大47.6653秒。pbload全程116次/109.456秒，pbcopy 11次/10.379秒，均无失败；累计工具时间与板端计算重叠，不能与模型时间直接相加。只覆盖本次3帧，不解除历史长视频PCIe故障的待验状态。

### v5验收与比较

v5计时版号5，输入路径 `rvv_window_fp16_reuse_dw3x3`；布局保持24320字节并兼容旧版。其菜单18优先匹配同输入、计数有效的v4 `run-sup5jvd9`，缺合格v4才向前匹配。保留该基线；操作只在[OPERATIONS](OPERATIONS.md#2-视频运行与结果)维护。

现场运行 `run-2i3ciq3m`：用户提供菜单18三页，3帧计数全部有效，无trace/finite/capture排除帧，自动匹配上述v4基线。**同次VIDEO CHECK结果尚未提供，数值状态可从菜单17离线确认，不要求重跑板子。**

| 对比项 | v4 | v5 | 降幅 |
|---|---:|---:|---:|
| model_cycles | 1,137,382,966 | 884,603,103 | 22.2%（1.286倍） |
| graph_cycles | 1,131,299,461 | 878,519,703 | 22.3% |
| service_cycles | 1,224,149,968 | 969,554,678 | 20.8% |
| 每帧记录耗时 | 43.2521秒 | 37.0487秒 | 14.3%（少6.2034秒） |
| run_roundtrip_seconds | 31.2231秒 | 24.9601秒 | 20.1% |

报告填写core=40MHz，按该频率换算模型22.115078秒、板端服务24.238867秒，分别约0.0452和0.0413帧/秒；不等同于host展示帧率。相对v1累计模型周期下降76.8%、每帧记录耗时下降66.5%。

固定图8层走直接路径，整帧input_im2col实测为0。input_pack=255,815,802、matrix_submit=23,384,863、matrix_wait=2,811,595、bias_dequant=44,131,740、other_compute=516,093,552、boundary_cache=34,210,151周期；帧内weight_prepare和权重转换次数均为0。启动94层准备81,413,039周期，5,033,984字节、2,391,664个有效权重值。

直接深度卷积含bias和内部缓存交接，整体记入节点compute/other_compute；bias_dequant只剩AMU卷积的独立偏置阶段。因边界变化，**不能把im2col/bias减少全部称作节省，也不能把other_compute增长直接称作退化**。本轮减少252,779,863模型周期；八个深度卷积的具体降幅仍需完整node-timing.csv，前十热点列表不足以计算各层收益。

当前热点（节点总周期，包含节点cache）：

| 节点 | 操作 | 平均周期 |
|---:|---|---:|
| 12 | Add | 36,938,876 |
| 314 | Permute | 30,983,363 |
| 155 | Nearest | 28,133,043 |
| 0 | 普通卷积 | 25,046,573 |
| 2 | 普通卷积 | 23,214,037 |
| 16 | 普通卷积 | 20,124,095 |
| 8 | 普通卷积 | 20,079,736 |
| 26 | Add | 18,479,582 |
| 83 | Pool | 17,760,610 |
| 84 | Pool | 17,760,145 |

深度卷积退出前十热点。单节点12占graph约4.2%，不能从它外推全部Add/Sub收益；input_pack仍占模型28.9%。依据这些热点及用户合并优化授权，v6将明确保持数值规则的内核修改统一交付，原理只在[IMPLEMENTATION](IMPLEMENTATION.md#v6合并优化的原理)维护。

Host平均prepare=0.0165、postprocess=0.0061、upload=4.7671、download=1.9052、profile_read=1.9109秒；elapsed最大40.9181秒。pbload全程97次/92.819秒，pbcopy 11次/10.478秒，均无失败；累计工具时间与计算重叠，不能相加为额外串行成本。本次只有三帧，不解除旧长视频工具退出-11的待验状态。重启host后本次运行恢复正常，也不据此将此前BAR资源异常与旧退出-11归为同一原因。

### v6验收与比较

本轮标识 `rvv_bulk_v6`、计时版号6，24320字节布局和所有原字段保持不变。包含RVV连续/周期寻址、整缓冲区交接、转置/复制/上采样/池化、四行有序BMM、Max/堆Top-K、FP16打包分支和CRC查表。**本轮没有实板耗时数据，不声称每个候选或整帧已经提速。**

下一次只需同一视频前三帧，确认独立golden后查看18，自动优先按相同输入匹配v5 `run-2i3ciq3m`，缺合格v5再向前匹配。原始结果及该基线保留，不要求逐节点重采实板。

阶段边界沿用v5：DW含bias/cache记入other_compute，普通卷积的独立bias仍计入bias_dequant；input_pack涵盖1×1直接读取、内部窗口及边缘掩码路径。CRC优化作用于input_crc/service，不能全部归为模型周期收益。比较model/service/节点总周期与host记录耗时，不将QEMU执行速度换算为实板收益。

### 计时字段与解释


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
| input_im2col | 生成FP32窗口矩阵；v3/v4只用于原RVV分组卷积；v5/v6固定图全部为0，通用回退仍可计时 |
| input_pack | v2仅转换/打包；v3含标量窗口及clean；v4/v5含RVV索引/gather、整数RNE、补零与scratch交接；v6含直接/内部/边缘取数分支，每空间块一次 |
| matrix_submit | 矩阵配置/加载/乘加/存储提交及内部缓存维护，不能视为纯指令发射 |
| matrix_wait | 提交后读取xmfflags排空的剩余等待及标志发布；不是AMU总运行时间 |
| bias_dequant | 独立偏置遍历的周期；v5/v6不含直接深度卷积内部的bias，此版本无量化 |
| other_compute | compute减去六阶段之和；v5/v6还包含完整直接深度卷积，不能当作纯软件开销 |
| 每节点check | 计算后的有限值检查分支；普通模式仅最终输出扫描 |
| graph_overhead_cycles | 图总周期减去全部节点cache/compute/check之和，含进度日志、诊断提交、记账等 |

计时表写入、读计数器、发布和Host读取有开销。带逐节点trace、中间有限值扫描或节点暂停采集的运行不进入性能排行。启动自测、权重加载不在模型周期内；Host第一帧等待仍可能包含会话初始化。不同版本input_pack边界不同，比较输入准备应合并input_im2col、input_pack与内部维护，不能只看某一项下降。

18询问的是**当前FPGA上N550实际运行频率MHz**，未知直接回车，仅报周期。不能用RTL目标频率或QEMU计数推算板端秒数。Host的PCIe等待包含pbload轮询和工具启动；`elapsed_seconds`含录像保存，但未完整计入最终窗口显示和JSON落盘，不能称为采集到可见延迟。离线golden耗时单列，不算推理帧耗时。

计时表uint64头固定16项：`magic version bytes frame nodes completed options state model_cycles graph_cycles input_check_cycles service_cycles input_crc_cycles output_cycles node_cycles session`。v1共8152字节；v2至v6均24320字节，布局不变。每节点为cache/compute/check及六阶段周期，随后全局六组cycles/calls、五项权重统计，末项uint64承载此前全部字节的CRC32。五项为weight_pack_bytes、weight_pack_values、weight_pack_cycles、prepared_layers、runtime_weight_conversions。Host兼容六版，核验CRC、会话/帧号、完成数量、阶段总账及权重不变量；离线重建核验SHA256。

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
