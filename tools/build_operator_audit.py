"""Generate the current 334-node audit from frozen assets; Python 3.8 stdlib."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = '2026-10-19 / input path v4'
# Maintained source review, not an automatic proof of implementation/speed.
RULES = {
    'CONV': ('局部特征加权求和及通道组合', '按K顺序乘加，最后加bias', '见分组规则', 'accel_model.c;amu.c;amu_input_rvv.c', '输入准备；深度卷积专用路径'),
    'SILU': ('非线性特征变换', 'x*sigmoid(x)', 'RVV FP32指数近似、乘法与除法', 'rvv_model_ops.c', '先测量指数/除法成本，保持既有数值规则'),
    'ADD': ('残差/特征或坐标相加', '广播后逐元素a+b', 'C逐lane计算广播索引，RVV加载和相加', 'accel_model.c', '同形状连续加载快速路径'),
    'SUB': ('特征或坐标相减', '广播后逐元素a-b', 'C逐lane计算广播索引，RVV加载和相减', 'accel_model.c', '同形状连续加载快速路径'),
    'MUL': ('特征加权或坐标缩放', '广播后逐元素a*b', 'C逐lane计算广播索引，RVV加载和相乘', 'accel_model.c', '同形状/标量广播快速路径'),
    'COPY': ('保持数据顺序并改变逻辑形状', '输出与输入共享同一地址', '同址直接返回；异址RVV复制', 'accel_model.c', '当前图同址别名已实现'),
    'SLICE': ('取指定轴的连续区间', 'Y=X[axis,start:start+width]', 'FP32用RVV复制，其他类型标量', 'accel_model.c;basic_rvv.c;scalar_model.c', '连续子视图需同时修改生命期和guard'),
    'CONCAT': ('合并不同来源的特征', '沿axis依次拼接', 'RVV分段复制到输出', 'accel_model.c;basic_rvv.c', '考察生产者直接写目标布局'),
    'POOL': ('扩大感受范围并保留强特征', '窗口内最大值', 'RVV扫描窗口，边界外不参与', 'basic_rvv.c', '连续行专用路径，保持边界规则'),
    'NEAREST': ('将特征图放大', '按最近输入位置复制', 'RVV复制，本图H/W各放大2倍', 'basic_rvv.c', '连续行写入及与Concat融合'),
    'PERMUTE': ('调整轴和内存排列', '按给定轴顺序重排', 'C生成索引，RVV索引读取', 'accel_model.c', '连续分段及消费者布局兼容'),
    'EXPAND': ('广播轴重复或同形状恒等', '长度1的轴重复至目标形状', '当前均物化，FP32用RVV，INT64用标量', 'accel_model.c;scalar_model.c', '8个等形状别名；2个索引展开与Gather融合'),
    'BMM': ('Attention相关性与内容汇总', '[B,M,K]*[B,K,N]', 'RVV沿N并行，K顺序独立乘法和加法', 'rvv_model_ops.c', '多行共享B加载，保留各输出累加顺序'),
    'SOFTMAX': ('归一化Attention权重', 'exp(x-max)/sum(exp(x-max))', 'RVV指数与归一化，最大值/求和按既有顺序', 'rvv_model_ops.c', '归约或近似更改需另定数值方案'),
    'SIGMOID': ('生成类别概率', '1/(1+exp(-x))', 'RVV FP32指数近似和除法', 'rvv_model_ops.c', '测量后评估近似成本'),
    'MAX': ('取每个位置最高类别分数', '沿axis求最大值及首个索引', '标量顺序扫描', 'scalar_model.c', '保留同分首索引语义'),
    'TOPK': ('选择最高分候选', '高分优先，同分小索引优先', '标量维护长度300的有序数组', 'scalar_model.c', '候选堆或选择算法需保持完整排序规则'),
    'GATHER': ('按候选索引选择分数/坐标', '按axis和索引张量读取', 'FP32用RVV索引加载，INT64用标量', 'accel_model.c;scalar_model.c', '与Expand合并，按整行搬运'),
    'REMAINDER': ('从扁平索引提取类别', 'X mod 常数', '标量INT64', 'scalar_model.c', '保持整数与负数规则'),
    'FLOORDIV': ('从扁平索引提取候选位置', 'floor(X/常数)', '标量INT64', 'scalar_model.c', '保持整数与负数规则'),
    'CAST': ('将类别索引转换为输出类型', '逐元素类型转换', '标量C', 'scalar_model.c', '保持类型、范围及元素顺序'),
}
SOURCES = ('accel_model.c', 'amu.c', 'amu.h', 'amu_input_rvv.c', 'amu_half_rvv.h',
           'rvv_model_ops.c', 'basic_rvv.c', 'scalar_model.c',
           'board/coherent_rvv.h', 'board/cache.h', 'board/main.c',
           'board/diagnostics.c', 'board/diagnostics.h', 'build_board.sh')


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
                impl = '启动FP16权重预打包；RVV gather+整数RNE直接输入打包；N32块供全部M32块复用；AMU K32顺序累加；RVV加bias'
                packed = '[{},32,32]'.format((k+31)//32)
                block_bytes = ((k+31)//32)*2048
                status = '权重复用/标量融合已有实测；RVV输入版实板待验收'
                constraint = 'FP16 RNE位模式一致；每输出K顺序/逐指令舍入/bias独立加法不变；CLP输出所有权交接'
            else:
                engine = 'RVV FP32'
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
        elif kind in ('MAX', 'TOPK', 'REMAINDER', 'FLOORDIV', 'CAST'):
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
