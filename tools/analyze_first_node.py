"""Offline first-convolution analysis from saved capture files (Python 3.8+).

No PCIe access. Reconstructed FP16 operands and candidate accumulators are not
observations of the board's packed buffers or AMU intermediate results.
"""
import argparse
import array
import json
import math
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'yolo26_pc'))
from node_compare import compare_tensor, digest, indices, ordered, replay_convolution


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def inside(root,name):
    path=(root/name).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ValueError('Asset path escapes repository') from None
    return path


def verified(path,sha,size=None):
    data=path.read_bytes()
    if not sha or digest(data)!=sha or (size is not None and len(data)!=size):
        raise ValueError('Saved data hash/length mismatch: '+str(path))
    return data


def word(data,index):
    return struct.unpack_from('<I',data,index*4)[0]


def value(data,index):
    return struct.unpack_from('<f',data,index*4)[0]


def bits(value):
    return '{:08x}'.format(struct.unpack('<I',struct.pack('<f',value))[0])


def extrema(actual,reference,shape):
    if len(actual)!=len(reference) or len(actual)!=math.prod(shape)*4:
        raise ValueError('Output shape/length mismatch')
    av=array.array('f'); av.frombytes(actual)
    rv=array.array('f'); rv.frombytes(reference)
    aw=array.array('I'); aw.frombytes(actual)
    rw=array.array('I'); rw.frombytes(reference)
    if sys.byteorder!='little':
        for data in (av,rv,aw,rw): data.byteswap()
    max_ulp=max_abs=None
    for index,(a,r,ab,rb) in enumerate(zip(av,rv,aw,rw)):
        if not math.isfinite(a) or not math.isfinite(r):
            raise ValueError('Nonfinite output at element '+str(index))
        sample={'element':index,'index':indices(index,shape),'actual':a,'reference':r,
                'actual_bits':f'{ab:08x}','reference_bits':f'{rb:08x}',
                'absolute_difference':abs(a-r),'ulp':abs(ordered(ab)-ordered(rb))}
        if max_ulp is None or sample['ulp']>max_ulp['ulp']: max_ulp=sample
        if max_abs is None or sample['absolute_difference']>max_abs['absolute_difference']: max_abs=sample
    if max_ulp is None:
        raise ValueError('Empty output')
    return max_ulp,max_abs


def dot_terms(node,tensors,inputs,element):
    x,w,b=(inputs[tid] for tid in node['inputs'])
    xs,ws,ys=(tensors[tid]['shape'] for tid in (node['inputs'][0],node['inputs'][1],node['outputs'][0]))
    if len(xs)!=4 or len(ws)!=4 or len(ys)!=4:
        raise ValueError('Expected NCHW/OIHW convolution shapes')
    n,o,y,col=indices(element,ys)
    _,ic,ih,iw=xs; oc,wic,kh,kw=ws
    sh,sw,ph,pw,groups=node['params'][:5]
    if groups!=1 or wic!=ic or ic*kh*kw!=27 or min(sh,sw)<=0:
        raise ValueError('This first-layer analysis requires ordinary K=27 convolution')
    if ys!=[xs[0],oc,(ih+2*ph-kh)//sh+1,(iw+2*pw-kw)//sw+1]:
        raise ValueError('Convolution output shape mismatch')
    terms=[]
    for channel in range(ic):
        for ky in range(kh):
            for kx in range(kw):
                iy=y*sh+ky-ph; ix=col*sw+kx-pw
                xi=((n*ic+channel)*ih+iy)*iw+ix if 0<=iy<ih and 0<=ix<iw else None
                wi=((o*ic+channel)*kh+ky)*kw+kx
                xv=value(x,xi) if xi is not None else 0.0
                wv=value(w,wi)
                if not math.isfinite(xv) or not math.isfinite(wv):
                    raise ValueError('Nonfinite convolution input')
                xraw=struct.pack('<e',xv); wraw=struct.pack('<e',wv)
                xh=struct.unpack('<e',xraw)[0]; wh=struct.unpack('<e',wraw)[0]
                terms.append({'k':len(terms),'channel':channel,'ky':ky,'kx':kx,
                              'input_index':xi,'input_coordinate':[n,channel,iy,ix],
                              'weight_index':wi,'padding':xi is None,
                              'input_fp32':xv,'weight_fp32':wv,
                              'input_bits':bits(xv),'weight_bits':bits(wv),
                              'input_half':int.from_bytes(xraw,'little'),
                              'weight_half':int.from_bytes(wraw,'little'),
                              'input_fp16':xh,'weight_fp16':wh,'product':xh*wh})
    return terms,value(b,o)


def analyze(directory,root=ROOT):
    directory=Path(directory).resolve(); root=Path(root).resolve()
    summary_raw=(directory/'summary.json').read_bytes()
    first_raw=(directory/'first_difference.json').read_bytes()
    summary=json.loads(summary_raw); first=json.loads(first_raw)
    manifest=read_json(root/'reference/nodes/manifest.json')
    node=manifest['nodes'][0]; tensors=manifest['tensors']
    if (first['node']!=0 or first['kind']!='CONV' or node['kind']!='CONV' or
            first['module']!=node['module'] or first['tensor']!=node['outputs'][0] or
            len(node['inputs'])!=3 or first['shape']!=tensors[first['tensor']]['shape']):
        raise ValueError('Saved difference is not the frozen first convolution')
    model=read_json(root/'model/manifest.json')
    if {r['name']:r['sha256'] for r in model['files']}!=manifest['model_files_sha256']:
        raise ValueError('Frozen model and node reference manifests differ')
    assets={r['name']:verified(inside(root/'model',r['name']),r['sha256'],r['bytes']) for r in model['files']}
    constants=assets['constants.bin']
    sample=summary['sample']
    references={r['id']:r['sha256'] for r in manifest['samples'][sample]['records']}
    prepared=read_json(root/'reference/prepared/manifest.json')['samples'][sample]
    original=verified(inside(root,prepared['input']),prepared['input_sha256'])
    if digest(original)!=references[0] or digest(original)!=summary.get('input_sha256'):
        raise ValueError('Capture input does not match frozen sample')
    checks={r['tensor']:r for r in first['inputs']}
    if set(checks)!=set(node['inputs']):
        raise ValueError('Missing saved convolution inputs')
    inputs={}; input_checks=[]
    for tid in node['inputs']:
        t=tensors[tid]; check=checks[tid]
        data=verified(directory/f'first-input-{tid:04d}.bin',check['sha256_actual'],t['bytes'])
        if t['constant']:
            expected=constants[t['offset']:t['offset']+t['bytes']]
        elif tid==0:
            expected=original
        else:
            raise ValueError('Unexpected first-layer input')
        if len(expected)!=t['bytes'] or digest(expected)!=check['sha256_reference']:
            raise ValueError('Saved input reference hash mismatch')
        inputs[tid]=data
        input_checks.append({'id':tid,'shape':t['shape'],'match':data==expected,'sha256':digest(data)})
    tid=first['tensor']; shape=tensors[tid]['shape']
    actual=verified(directory/f'tensors/{tid:04d}.bin',first['metrics']['sha256_actual'],tensors[tid]['bytes'])
    reference=verified(directory/'first-reference.bin',references[tid],len(actual))
    if digest(reference)!=first['metrics']['sha256_reference']:
        raise ValueError('Saved output reference hash mismatch')
    maximum,absolute=extrema(actual,reference,shape)
    metrics=compare_tensor(actual,reference,1,shape)
    terms,bias=dot_terms(node,tensors,inputs,maximum['element'])
    replay=replay_convolution(node,tensors,inputs,actual,reference,[maximum])['results'][0]
    before=replay['first_16_blocks'][0]
    names=(('Q','qemu_fp64_dot_fp32_per_k32','qemu_accumulator'),
           ('S','sequential_fp32','sequential_fp32_accumulator'),
           ('T','pairwise_fp32_k32','tree_fp32_accumulator'))
    rules=[]
    for label,key,accumulator in names:
        r=replay['rules'][key]
        rules.append({'label':label,'before_bias':before[accumulator],'before_bits':bits(before[accumulator]),
                      'after_bias':r['value'],'after_bits':bits(r['value']),
                      'matches_board':r['matches_actual_bits'],'matches_reference':r['matches_reference_bits']})
    identity=digest(summary_raw+first_raw+json.dumps(manifest,sort_keys=True).encode())[:12]
    return {'version':1,'id':identity,'capture_status':summary.get('status','unknown'),'sample':sample,
            'node':0,'kind':'CONV','module':node['module'],'tensor':tid,'shape':shape,
            'metrics':metrics,'inputs':input_checks,'max_ulp':maximum,'max_absolute':absolute,
            'focus':'max_ulp','bias':bias,'bias_bits':bits(bias),'terms':terms,'rules':rules,
            'dot_fp64':sum(t['product'] for t in terms),
            'note':'FP16 operands and pre-bias accumulators are reconstructed, NOT board measurements. '
                   'Matching candidates do not prove the FPGA accumulation rule.'}


def sample_lines(label,sample):
    return [f"{label}: element={sample['element']} index={sample['index']} ULP={sample['ulp']}",
            f" A={sample['actual']:.9g} [{sample['actual_bits']}] R={sample['reference']:.9g} [{sample['reference_bits']}]",
            f" Absolute difference={sample['absolute_difference']:.9g}"]


def render(report,page):
    if page not in (1,2,3,4): raise ValueError('Choose page 1-4')
    state=report['capture_status']; metrics=report['metrics']
    lines=[f'FIRST CONV ANALYSIS - PAGE {page}/4',
           f"ID={report['id']} sample={report['sample']} capture={state}"]
    if page==1:
        lines += [f"Node=0 CONV tensor={report['tensor']} shape={report['shape']}",
                  'Saved files verified against capture hashes and frozen assets.']
        for item in report['inputs']:
            lines.append(f"Input t{item['id']} shape={item['shape']} full-match={'YES' if item['match'] else 'NO'}")
        lines += [f"Different elements={metrics['different_elements']}/{metrics['count']}",
                  f"Abs max/mean/RMSE={metrics['max_absolute_difference']:.6g}/{metrics['mean_absolute_difference']:.6g}/{metrics['rmse']:.6g}"]
        lines += sample_lines('Max ULP',report['max_ulp'])+sample_lines('Max abs',report['max_absolute'])
        lines += ['Pages 2-4 focus on the maximum-ULP element.',
                  'Large ULP alone is not a detection-accuracy measurement.',
                  'No final-model accuracy conclusion from this partial capture.' if state!='completed'
                  else 'First-layer analysis alone does not establish model accuracy.']
    elif page==2:
        lines += sample_lines('Focus',report['max_ulp'])
        lines += [f"Dot K=27; FP32 bias={report['bias']:.9g} [{report['bias_bits']}]",
                  f"FP64 sum of reconstructed FP16 products={report['dot_fp64']:.17g}",
                  'Q=FP64 dot then FP32; S=sequential FP32; T=pairwise FP32 tree',
                  'Rule     Before bias [bits]           After bias [bits]']
        for rule in report['rules']:
            lines += [f" {rule['label']} {rule['before_bias']:.9g} [{rule['before_bits']}] -> {rule['after_bias']:.9g} [{rule['after_bits']}]",
                      f"   Matches board={rule['matches_board']} reference={rule['matches_reference']}"]
        lines += ['Board pre-bias output was NOT captured; the above are candidates.',
                  'FP16 values use round-to-nearest-even reconstruction.',
                  'No claim of rounding-only error or correct AMU hardware.']
    else:
        focus=report['max_ulp']
        lines += [f"Focus element={focus['element']} NCHW={focus['index']}; K order=(c,ky,kx)",
                  'X=input W=weight; 32/16 columns are raw FP32/FP16 word bits.',
                  ' k (c,y,x) X32      W32      X16  W16  X16*W16']
        start=0 if page==3 else 14
        for t in report['terms'][start:start+14]:
            lines.append(f"{t['k']:2} ({t['channel']},{t['ky']},{t['kx']}) {t['input_bits']} {t['weight_bits']} "
                         f"{t['input_half']:04x} {t['weight_half']:04x} {t['product']:.9g}")
        lines += ['Operands reconstructed from saved inputs; NOT measured AMU buffers.',
                  'Full decimal values/coordinates: first-node-analysis.json on host.']
    if len(lines)>22 or any(len(line)>80 for line in lines):
        raise ValueError('Analysis page exceeds 80x22 screenshot budget')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--page',type=int,choices=(1,2,3,4),default=1)
    args=parser.parse_args()
    try:
        report=analyze(args.directory)
        output=render(report,args.page)
        target=args.directory/'first-node-analysis.json'
        temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        temporary.replace(target)
        print(output,end='')
    except (OSError,ValueError,KeyError,IndexError,TypeError,OverflowError) as exc:
        parser.exit(1,'Analysis error: '+str(exc)+'\n')


if __name__=='__main__': main()
