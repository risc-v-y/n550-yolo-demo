"""Generate the current 334-node audit from frozen assets; Python 3.8 stdlib."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = '2026-10-21 / FP16 three-row cache / whole-layer bias / first-block probe'
# Maintained source review, not an automatic proof of implementation/speed.
RULES = {
    'CONV': ('局部特征加权求和及通道组合', '按K顺序乘加，最后加bias', '见分组规则', 'accel_model.c;amu.c;amu_input_rvv.c', '输入准备；深度卷积专用路径'),
    'SILU': ('非线性特征变换', 'x*sigmoid(x)', 'RVV m2原FP32指数近似、乘法与除法；整缓冲区交接', 'rvv_model_ops.c;rvv_access.h', '测量指数/除法成本，不改变近似和运算顺序'),
    'ADD': ('残差/特征或坐标相加', '广播后逐元素a+b', '同形状RVV m2连续相加；其他广播回退', 'accel_model.c;rvv_model_ops.c;rvv_access.h', '先测连续读取路径'),
    'SUB': ('特征或坐标相减', '广播后逐元素a-b', '同形状RVV m2连续相减；其他广播回退', 'accel_model.c;rvv_model_ops.c;rvv_access.h', '先测连续读取路径'),
    'MUL': ('特征加权或坐标缩放', '广播后逐元素a*b', 'RVV m2标量或连续尾部周期广播；不生成逐lane索引', 'accel_model.c;rvv_model_ops.c;rvv_access.h', '保留逐元素FP32乘法'),
    'COPY': ('保持数据顺序并改变逻辑形状', '输出与输入共享同一地址', '同址直接返回；异址RVV复制', 'accel_model.c', '当前图同址别名已实现'),
    'SLICE': ('取指定轴的连续区间', 'Y=X[axis,start:start+width]', 'FP32用RVV m2分段复制，整段交接；其他类型标量', 'accel_model.c;basic_rvv.c;rvv_access.h;scalar_model.c', '零拷贝子视图需另外修改生命期和guard'),
    'CONCAT': ('合并不同来源的特征', '沿axis依次拼接', 'RVV m2分段复制到输出；整输出交接一次', 'accel_model.c;basic_rvv.c;rvv_access.h', '生产者直接写目标布局需另评估生命期'),
    'POOL': ('扩大感受范围并保留强特征', '窗口内最大值', 'RVV沿空间位置并行，跨通道复用窗口索引；无效lane取负无穷', 'basic_rvv.c;rvv_access.h', '先测空间并行路径，保留原ky/kx最大值语义'),
    'NEAREST': ('将特征图放大', '按最近输入位置复制', '宽度2倍用RVV寄存器重复后连续写入；输出行复用', 'basic_rvv.c;rvv_access.h', '与Concat融合需另评估布局'),
    'PERMUTE': ('调整轴和内存排列', '按给定轴顺序重排', '固定图均为末两轴转置：RVV跨步读、连续写；其他排列回退', 'accel_model.c;rvv_model_ops.c;rvv_access.h', '更大转置分块需按实板访存测量'),
    'EXPAND': ('广播轴重复或同形状恒等', '长度1的轴重复至目标形状', '8个同形状RVV整段复制；2个INT64末轴RVV重复，仍物化', 'accel_model.c;rvv_model_ops.c;rvv_access.h;scalar_model.c', '零拷贝或与Gather融合需另评估生命期'),
    'BMM': ('Attention相关性与内容汇总', '[B,M,K]*[B,K,N]', 'RVV m2沿N并行，每4行复用B加载，K顺序独立乘法和加法', 'rvv_model_ops.c;rvv_access.h', '实板检查分块、寄存器与访存成本'),
    'SOFTMAX': ('归一化Attention权重', 'exp(x-max)/sum(exp(x-max))', 'RVV m2原指数与归一化；逐行交接，标量求和顺序不变', 'rvv_model_ops.c;rvv_access.h', '归约或近似更改需另定数值方案'),
    'SIGMOID': ('生成类别概率', '1/(1+exp(-x))', 'RVV m2原FP32指数近似和除法；整缓冲区交接', 'rvv_model_ops.c;rvv_access.h', '测量原近似成本'),
    'MAX': ('取每个位置最高类别分数', '沿axis求最大值及首个索引', 'RVV沿位置并行，严格大于才更新；保持同分首索引', 'accel_model.c;rvv_model_ops.c;rvv_access.h', '按实测评估跨步读取成本'),
    'TOPK': ('选择最高分候选', '高分优先，同分小索引优先', '标量最差候选小堆，最后堆排序；NaN回退原插入路径', 'accel_model.c;scalar_model.c', '保持原值位模式及完整候选排序'),
    'GATHER': ('按候选索引选择分数/坐标', '按axis和索引张量读取', '先检查索引；相同索引的行用RVV整行复制；非重复索引逐项读取', 'accel_model.c;rvv_model_ops.c;rvv_access.h;scalar_model.c', '与Expand融合需另评估图和布局'),
    'REMAINDER': ('从扁平索引提取类别', 'X mod 常数', '标量INT64', 'scalar_model.c', '保持整数与负数规则'),
    'FLOORDIV': ('从扁平索引提取候选位置', 'floor(X/常数)', '标量INT64', 'scalar_model.c', '保持整数与负数规则'),
    'CAST': ('将类别索引转换为输出类型', '逐元素类型转换', '标量C', 'scalar_model.c', '保持类型、范围及元素顺序'),
}
SOURCES = ('accel_model.c', 'amu.c', 'amu.h', 'amu_input_rvv.c', 'amu_half_rvv.h',
           'rvv_model_ops.c', 'rvv_model_ops.h', 'rvv_access.h', 'basic_rvv.c', 'scalar_model.c',
           'board/coherent_rvv.h', 'board/cache.h', 'board/main.c',
           'board/diagnostics.c', 'board/diagnostics.h', 'board/model_io.c', 'build_board.sh')


def tensor_text(t):
    dtype = {1:'FP32', 2:'INT64'}[t['dtype']]
    region = 'weights' if t['constant'] else 'arena'
    return 'id={} {} {} @{}+{} ({}B)'.format(t['id'], dtype,
           json.dumps(t['shape'], separators=(',', ':')), region, t['offset'], t['bytes'])


def rows_for(manifest):
    rows = []
    for node in manifest['nodes']:
        kind = node['kind']; purpose, formula, impl, files, todo = RULES[kind]
        inputs = [manifest['tensors'][i] for i in node['inputs'] if i >= 0]
        outputs = [manifest['tensors'][i] for i in node['outputs'] if i >= 0]
        engine = 'RVV' if outputs[0]['dtype'] == 1 else 'scalar C'
        status = '候选未实施，按实测排序'
        if kind in ('SILU','ADD','SUB','MUL','SLICE','CONCAT','POOL','NEAREST','PERMUTE',
                    'EXPAND','BMM','SOFTMAX','SIGMOID','MAX','TOPK','GATHER'):
            status = '当前已实施；开发回归与实板状态见VALIDATION'
            if kind in ('EXPAND','GATHER'): engine = 'RVV'
        constraint = '保持既有元素对应、FP32计算/整数语义及缓存交接'
        macs = 0; dims = ''; packed = ''; block_bytes = 0
        p = node['params']
        if kind == 'CONV':
            w = inputs[1]['shape']; out = outputs[0]['shape']
            m, k, n = w[0], w[1]*w[2]*w[3], out[2]*out[3]
            macs = out[0]*m*k*n
            dims = 'batch={} groups={} M/group={} N={} K={}'.format(out[0], p[4], m//p[4], n, k)
            if p[4] == 1:
                engine = 'AMU FP16 / RVV input'
                if w[2:] == [3,3] and p[0] <= 2 and p[1] <= 2:
                    impl = '启动FP16权重预打包；三行FP16环形缓存复用RNE转换；RVV半字gather生成N32输入；AMU K32顺序累加；整层RVV独立加bias'
                else:
                    impl = '启动FP16权重预打包；1x1跨通道直读/通用窗口masked gather；整数RNE；N32输入复用；AMU K32顺序累加；整层RVV独立加bias'
                packed = '[{},32,32]'.format((k+31)//32)
                block_bytes = ((k+31)//32)*2048
                status = '当前行缓存/整层bias已实施；实板与数值状态见VALIDATION'
                todo = '按最新实测评估1x1布局、输入gather及AMU调度'
                constraint = 'FP16 RNE位模式一致；每输出K顺序/逐指令舍入/bias独立加法不变；CLP输出所有权交接'
            else:
                engine = 'RVV FP32'
                if (w[1] == 1 and p[:4] == [1, 1, 1, 1] and w[2:] == [3, 3] and
                        p[4] == inputs[0]['shape'][1] == out[1]):
                    impl = 'RVV m2按行直接读取3x3窗口；ky/kx九次独立乘加；最后独立加bias；整输出CLP交接，无im2col'
                    status = '当前已实施；开发回归与实板状态见VALIDATION'
                    todo = '先测直接窗口，后续复用不得改变每输出累加顺序'
                else:
                    impl = '按组生成FP32 Kx32 im2col；RVV沿位置并行，按K顺序独立乘加；最后加bias'
                files = 'accel_model.c;rvv_model_ops.c'
                constraint = 'ky/kx顺序不变，不引入FMA；边界补零、尾VL及缓存交接'
        elif kind == 'BMM':
            b, m, k = inputs[0]['shape']; n = inputs[1]['shape'][2]
            dims = 'B={} M={} N={} K={}'.format(b, m, n, k); macs = b*m*n*k
            constraint = 'K顺序、独立乘法/加法、尾VL及数据对应不变'
        elif kind == 'COPY':
            alias = all((t['offset'], t['constant']) == (inputs[0]['offset'], inputs[0]['constant']) for t in outputs)
            engine = 'alias' if alias else 'RVV'
            status = '已实现同址零拷贝' if alias else '仍复制'
        elif kind in ('TOPK', 'REMAINDER', 'FLOORDIV', 'CAST'):
            engine = 'scalar C'
        if kind == 'TOPK': formula += '; axis={} K={}'.format(p[0], p[1])
        if kind in ('POOL', 'NEAREST'): files = 'accel_model.c;'+files
        rows.append(dict(node=node['id'], module=node['module'], operation=kind,
            purpose=purpose, input='; '.join(tensor_text(t) for t in inputs),
            output='; '.join(tensor_text(t) for t in outputs), params=json.dumps(p),
            formula=formula, dimensions=dims, macs=macs, engine=engine,
            current_implementation=impl, source=files, packed_input_shape=packed,
            packed_input_bytes=block_bytes, status=status, candidate=todo, constraint=constraint))
    if len(rows) != 334 or set(RULES) != {r['operation'] for r in rows}:
        raise ValueError('Frozen node/operator coverage mismatch')
    return rows


def artifacts(root):
    manifest = json.loads((root/'reference/nodes/manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['model_files_sha256'].items():
        if hashlib.sha256((root/'model'/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen asset mismatch: '+name)
    rows = rows_for(manifest)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    csv_bytes = stream.getvalue().encode('utf-8')
    hashes = {}
    for name in SOURCES:
        raw = (root/'yolo26_riscv'/name).read_bytes().replace(b'\r\n', b'\n')
        hashes[name] = hashlib.sha256(raw).hexdigest()
    meta = dict(revision=REVISION, nodes=len(rows), operators=dict(Counter(r['operation'] for r in rows)),
                engines=dict(Counter(r['engine'] for r in rows)), source_sha256_lf=hashes,
                model_files_sha256=manifest['model_files_sha256'],
                csv_sha256=hashlib.sha256(csv_bytes).hexdigest(),
                scope='Static current-source review; candidates are not speed or correctness proofs.')
    return {'operator-audit.csv':csv_bytes,
            'operator-audit.json':(json.dumps(meta, ensure_ascii=False, indent=2)+'\n').encode('utf-8')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Fail if the audit or source binding is stale')
    args = parser.parse_args()
    for name, data in artifacts(ROOT).items():
        path = ROOT/'docs'/name
        if args.check:
            if not path.is_file() or path.read_bytes() != data: raise ValueError('Stale audit: '+name)
        else: path.write_bytes(data)
    print('Operator audit: 334 nodes / 21 operations; frozen assets and source hashes bound.')


if __name__ == '__main__': main()
