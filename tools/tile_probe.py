"""Python 3.8+ first-convolution tile acquisition and offline comparison.

The probe is a separate firmware, never a detector. Real PCIe transfers reuse
the checked mailbox/session protocol. Analysis uses only the standard library.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import textwrap
import time
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'yolo26_pc'))
from pcie_backend import PCIeBackend
from node_compare import ReferenceArchive
from analyze_first_node import bits, dot_terms, inside, read_json, verified, value

SIZE=25024
STARTS=(0,15328,18848)
POINTS=(28,29,30,15357,624567)
ANALYSIS_VERSION=2


def fp32_rne(total):
    """Round an exact rational directly to FP32, without binary64 double rounding."""
    total=Fraction(total)
    if not total: return 0.0
    sign=0x80000000 if total<0 else 0
    n,d=abs(total.numerator),total.denominator
    exponent=n.bit_length()-d.bit_length()
    below=(n<(d<<exponent)) if exponent>=0 else ((n<<-exponent)<d)
    if below:
        exponent-=1
    shift=23-max(exponent,-126)
    numerator,denominator=(n<<shift,d) if shift>=0 else (n,d<<-shift)
    mantissa,residue=divmod(numerator,denominator)
    mantissa+=2*residue>denominator or (2*residue==denominator and mantissa&1)
    if exponent<-126:
        word=sign|mantissa  # A rounded carry naturally produces minimum normal.
    else:
        if mantissa>=1<<24:
            mantissa>>=1; exponent+=1
        if exponent>127: raise OverflowError('FP32 replay overflow')
        word=sign|((exponent+127)<<23)|(mantissa&0x7fffff)
    return struct.unpack('<f',struct.pack('<I',word))[0]


def split16(products,bias):
    """Explanation comparator: RNE(sum(up to 16 exact products)+prior FP32).

    This captures the hardware's grouping, not its finite-window PE internals.
    The final short group must not be rounded before adding the prior result.
    """
    if not products: raise ValueError('Empty dot product')
    accumulator=0.0
    for first in range(0,len(products),16):
        exact=sum((Fraction(x) for x in products[first:first+16]),Fraction(accumulator))
        accumulator=fp32_rne(exact)
    return accumulator,fp32_rne(Fraction(accumulator)+Fraction(bias))


def decode(raw):
    if len(raw)!=SIZE:
        raise ValueError('Tile probe length mismatch')
    h=struct.unpack_from('<16I',raw)
    if h[:4]!=(0x31505459,1,SIZE,3) or h[6]!=0 or h[7]!=7 or any(h[8:]):
        raise ValueError('Tile probe magic/version/completion mismatch')
    records=[]
    for slot,first in enumerate(STARTS):
        at=64+slot*8320
        meta=struct.unpack_from('<16I',raw,at)
        if meta[:7]!=(first,16,32,27,27,32,43264):
            raise ValueError('Unexpected original AMU tile dimensions/strides')
        if meta[7]!=0:
            raise ValueError('Scalar/RVV frm is not RNE; refusing RNE bias comparison')
        r={'meta':meta,'left':struct.unpack_from('<1024H',raw,at+64),
           'right':struct.unpack_from('<1024H',raw,at+2112),
           'before':raw[at+4160:at+6208],'after':raw[at+6208:at+8256],
           'bias':raw[at+8256:at+8320]}
        for field in ('before','after','bias'):
            if any(not math.isfinite(x) for x, in struct.iter_unpack('<f',r[field])):
                raise ValueError('Nonfinite probe '+field)
        if any((r[field][row*32+k]&0x7c00)==0x7c00 for field,rows in (('left',16),('right',32))
               for row in range(rows) for k in range(27)):
            raise ValueError('Nonfinite packed FP16 lane')
        records.append(r)
    return h,records


class ProbeBackend(PCIeBackend):
    output_bytes=SIZE
    node_count=1
    required_capability=2
    firmware_name='tile-probe.bin'

    def validate_result(self,result):
        decode(result)


def assets():
    manifest=read_json(ROOT/'reference/nodes/manifest.json')
    model=read_json(ROOT/'model/manifest.json')
    if {r['name']:r['sha256'] for r in model['files']}!=manifest['model_files_sha256']:
        raise ValueError('Model/reference manifest mismatch')
    blobs={r['name']:verified(inside(ROOT/'model',r['name']),r['sha256'],r['bytes']) for r in model['files']}
    prepared=read_json(ROOT/'reference/prepared/manifest.json')['samples']['bus']
    data=verified(inside(ROOT,prepared['input']),prepared['input_sha256'],2076672)
    tensors=manifest['tensors']; constants=blobs['constants.bin']
    inputs={0:data}
    for tid in (1,2):
        t=tensors[tid]; inputs[tid]=constants[t['offset']:t['offset']+t['bytes']]
    archive=ReferenceArchive(ROOT,manifest,'bus')
    try:
        if archive.take(0)!=data: raise ValueError('Reference input mismatch')
        reference=archive.take(3)
    finally:
        archive.close()
    return manifest,inputs,constants,reference


def analyze(raw,fixture,old=None):
    manifest,inputs,constants,reference=fixture
    header,records=decode(raw)
    if header[4:6]!=(zlib.crc32(inputs[0]),zlib.crc32(constants)):
        raise ValueError('Board input/weights CRC differs from verified frozen assets')
    result={'sha256':hashlib.sha256(raw).hexdigest(),'tiles':[],'points':[],
            'packing_mismatches':[],'old_capture_supplied':old is not None}
    node=manifest['nodes'][0]; tensors=manifest['tensors']
    for r in records:
        first=r['meta'][0]
        counts={'first':first,'weight_bad':0,'input_bad':0,'bias_bad':0,
                'pre_vs_exact_RNE':0,'post_vs_actual_pre_plus_bias':0,'post_vs_qemu':0,'post_vs_old':0,
                'pre_vs_split16':0,'post_vs_split16':0,
                'meta':r['meta'],'inactive_left_nonzero':0,'inactive_right_nonzero':0}
        # Each packed lane is checked once, including all 16*27 and 32*27 active operands.
        for oc in range(16):
            terms,bias=dot_terms(node,tensors,inputs,oc*43264+first)
            if bits(value(r['bias'],oc))!=bits(bias): counts['bias_bad']+=1
            for k,t in enumerate(terms):
                if r['left'][oc*32+k]!=t['weight_half']:
                    counts['weight_bad']+=1
                    result['packing_mismatches'].append(['weight',first,oc,k,r['left'][oc*32+k],t['weight_half']])
        for col in range(32):
            terms,_=dot_terms(node,tensors,inputs,first+col)
            for k,t in enumerate(terms):
                if r['right'][col*32+k]!=t['input_half']:
                    counts['input_bad']+=1
                    result['packing_mismatches'].append(['input',first,col,k,r['right'][col*32+k],t['input_half']])
        counts['inactive_left_nonzero']=sum(x!=0 for i,x in enumerate(r['left']) if i//32>=16 or i%32>=27)
        counts['inactive_right_nonzero']=sum(x!=0 for i,x in enumerate(r['right']) if i%32>=27)
        for oc in range(16):
            bias=value(r['bias'],oc)
            for col in range(32):
                index=oc*32+col; element=oc*43264+first+col
                # FP16 products are exactly representable in binary64 here;
                # fsum gives the exact small first-layer sum before FP32 RNE.
                products=[struct.unpack('<e',struct.pack('<H',r['left'][oc*32+k]))[0]*
                          struct.unpack('<e',struct.pack('<H',r['right'][col*32+k]))[0] for k in range(27)]
                exact=math.fsum(products)
                split_pre,split_post=split16(products,bias)
                pre=value(r['before'],index); post=value(r['after'],index)
                ref=value(reference,element)
                counts['pre_vs_exact_RNE']+=bits(pre)!=bits(exact)
                counts['post_vs_actual_pre_plus_bias']+=bits(post)!=bits(pre+bias)
                counts['post_vs_qemu']+=bits(post)!=bits(ref)
                counts['pre_vs_split16']+=bits(pre)!=bits(split_pre)
                counts['post_vs_split16']+=bits(post)!=bits(split_post)
                if old is not None: counts['post_vs_old']+=bits(post)!=bits(value(old,element))
                if element in POINTS:
                    rational=sum((Fraction(x) for x in products),Fraction())
                    if Fraction(exact)!=rational: raise ValueError('Dot replay not exact in binary64')
                    result['points'].append({'element':element,'pre':bits(pre),'pre_RNE':bits(exact),
                        'pre_split16':bits(split_pre),'post_split16':bits(split_post),
                        'post':bits(post),'bias':bits(bias),'pre_plus_bias':bits(pre+bias),'qemu':bits(ref),
                        'old':bits(value(old,element)) if old is not None else None,
                        'exact_dot':str(rational),'exact_dot_float':exact})
        result['tiles'].append(counts)
    result['packing_mismatches']=result['packing_mismatches'][:10]
    result['points'].sort(key=lambda p:POINTS.index(p['element']))
    return result


def old_capture(directory,fixture):
    if directory is None: return None
    first=read_json(directory/'first_difference.json')
    summary=read_json(directory/'summary.json')
    if (summary['sample']!='bus' or first['node']!=0 or first['tensor']!=3 or
            summary['input_sha256']!=hashlib.sha256(fixture[1][0]).hexdigest()):
        raise ValueError('Selected capture is not the same BUS first convolution')
    return verified(directory/'tensors/0003.bin',first['metrics']['sha256_actual'],692224*4)


def saved_inputs(directory):
    original=(directory/'summary.json').read_bytes()
    report=json.loads(original)
    if (not isinstance(report,dict) or report.get('status')!='completed' or
            not isinstance(report.get('runs'),list) or len(report['runs'])!=2 or
            any(not isinstance(r,dict) or not r.get('sha256') for r in report['runs'])):
        raise ValueError('Need a completed capture with two recorded raw SHA256 hashes')
    raw=[verified(directory/('probe-{}.bin'.format(i+1)),r['sha256'],SIZE)
         for i,r in enumerate(report['runs'])]
    return original,report,raw


def report_id(report):
    body={k:v for k,v in report.items() if k!='id'}
    return hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()[:12]


def reanalyze(directory):
    """Read saved capture only. Commit the new sidecar after both runs succeed."""
    directory=Path(directory)
    started=time.monotonic()
    original,source,raw=saved_inputs(directory)
    fixture=assets()
    previous=Path(source['old_capture']) if source.get('old_capture') else None
    old=old_capture(previous,fixture)
    report=dict(source)
    report.update(analysis_version=ANALYSIS_VERSION,source_report_id=source.get('id'),
        source_report_sha256=hashlib.sha256(original).hexdigest(),
        source_runs_sha256=[r['sha256'] for r in source['runs']],
        model_files_sha256=fixture[0].get('model_files_sha256'),
        input_sha256=hashlib.sha256(fixture[1][0]).hexdigest(),
        constants_sha256=hashlib.sha256(fixture[2]).hexdigest(),
        runs=[analyze(blob,fixture,old) for blob in raw],stable=raw[0]==raw[1],
        analysis_seconds=time.monotonic()-started)
    report['id']=report_id(report)
    fd,name=tempfile.mkstemp(prefix='.analysis-v2-',suffix='.tmp',dir=str(directory))
    temporary=Path(name)
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(report,indent=2)+'\n')
        temporary.replace(directory/'analysis-v2.json')
    finally:
        temporary.unlink(missing_ok=True)
    return report


def load_report(directory):
    directory=Path(directory)
    path=directory/'analysis-v2.json'
    if not path.exists(): return read_json(directory/'summary.json')
    original,source,raw=saved_inputs(directory)
    report=read_json(path)
    hashes=[r['sha256'] for r in source['runs']]
    if (report.get('analysis_version')!=ANALYSIS_VERSION or
            report.get('source_report_sha256')!=hashlib.sha256(original).hexdigest() or
            report.get('source_report_id')!=source.get('id') or
            report.get('source_runs_sha256')!=hashes or
            [r.get('sha256') for r in report.get('runs',[])]!=hashes or
            report.get('id')!=report_id(report)):
        raise ValueError('Analysis source/hash/version mismatch; reanalyze the saved capture')
    return report


def split_pages(report,page):
    r=report['runs'][0]
    if page==1:
        print('Each tile: 512 outputs, M16/N32/K27. old = full capture.')
        print('once = exact FP16 dot, one FP32 RNE; bad = packed W/X/bias')
        for index,t in enumerate(r['tiles']):
            second=report['runs'][1]['tiles'][index]
            print('tile {}: bad W/X/bias={}/{}/{}'.format(t['first'],t['weight_bad'],t['input_bad'],t['bias_bad']))
            print('  once/RVV/QEMU/old={}/{}/{}/{}'.format(t['pre_vs_exact_RNE'],
                t['post_vs_actual_pre_plus_bias'],t['post_vs_qemu'],
                t['post_vs_old'] if r['old_capture_supplied'] else '-'))
            print('  split16 pre/post: run1={}/{} run2={}/{}'.format(
                t['pre_vs_split16'],t['post_vs_split16'],second['pre_vs_split16'],second['post_vs_split16']))
        matches=all(t[k]==0 for run in report['runs'] for t in run['tiles']
                    for k in ('pre_vs_split16','post_vs_split16'))
        print('All 1536 pre/post outputs match split16 in BOTH runs:',matches)
        if not report['stable']:
            totals=[sum(t[k] for t in report['runs'][1]['tiles']) for k in
                    ('weight_bad','input_bad','bias_bad','pre_vs_exact_RNE','post_vs_actual_pre_plus_bias','post_vs_qemu')]
            print('RUN2 bad W/X/bias; once/RVV/QEMU:', '/'.join(map(str,totals)))
        print('No tolerance. Match explains this input/3 tiles only.')
    elif page==2:
        print('element: measured-pre / split16-pre / once-pre')
        print('  post: measured / split16 / QEMU / old-full')
        for p in r['points']:
            print('{}: {} / {} / {}'.format(p['element'],p['pre'],p['pre_split16'],p['pre_RNE']))
            print('  post: {} / {} / {} / {}'.format(p['post'],p['post_split16'],p['qemu'],p['old'] or '-'))
            if not report['stable']:
                second=next(x for x in report['runs'][1]['points'] if x['element']==p['element'])
                print('  RUN2 measured pre/post: {} / {}'.format(second['pre'],second['post']))
        print('FP32 bit patterns. Full two-run details in analysis-v2.json.')
    else:
        for t in r['tiles']:
            m=t['meta']
            print('tile {} frm={} xmcsr={:x} xmfflags={:x}'.format(t['first'],*m[7:10]))
            print('  W/X/out DDR: '+' '.join('0x{:x}'.format(m[i]|m[i+1]<<32) for i in (10,12,14)))
            print('  nonzero inactive W/X lanes: {}/{}'.format(t['inactive_left_nonzero'],t['inactive_right_nonzero']))
        if r['packing_mismatches']:
            print('First packing diff [buffer,tile,row,k,got,expected]:')
            print(' ',r['packing_mismatches'][0])
        else:
            print('First packing differences: none')
        print('Source report:',report.get('source_report_id','new acquisition'))
        print('RNE every <=16 products + previous FP32, then RNE bias.')
        print('split16 is an explanation comparator, not a bit-exact PE model.')
        print('Source hashes verified; original summary/bin files preserved.')


def pages(report,page):
    print('TILE PROBE {} page {}/3'.format(report['id'],page))
    print('Status:',report['status'],'  repeated bytes identical:',report.get('stable'))
    if report['status']!='completed':
        lines=textwrap.wrap('ERROR: '+str(report.get('error')),width=78)
        if len(lines)>16: lines=lines[:8]+['... full error in summary.json / terminal.log ...']+lines[-7:]
        print('\n'.join(lines)); return
    if report.get('analysis_version')==ANALYSIS_VERSION:
        split_pages(report,page); return
    r=report['runs'][0]
    if page==1:
        print('Measured: packed FP16 buffers, AMU stored FP32, RVV post-bias')
        print('Only 3 original tiles; NOT a full 334-node inference.')
        print('Each tile: M=16 N=32 K=27, output row stride=43264')
        for t in r['tiles']:
            print('tile {}: bad W/X/bias={}/{}/{}'.format(t['first'],t['weight_bad'],t['input_bad'],t['bias_bad']))
            print('  differing pre/RVV-add/QEMU={}/{}/{} (of 512)'.format(
                t['pre_vs_exact_RNE'],t['post_vs_actual_pre_plus_bias'],t['post_vs_qemu']))
            print('  differs from old full capture:',t['post_vs_old'] if r['old_capture_supplied'] else 'not supplied')
        print('pre comparator = exact FP16 dot, once rounded to FP32 RNE.')
        print('A mismatch alone does NOT prove an illegal rounding rule.')
        if not report.get('stable'):
            totals=[sum(t[k] for t in report['runs'][1]['tiles']) for k in
                    ('weight_bad','input_bad','pre_vs_exact_RNE','post_vs_actual_pre_plus_bias','post_vs_qemu')]
            print('RUN2 differing W/X/pre/add/QEMU:', '/'.join(map(str,totals)))
    elif page==2:
        print('element: measured-pre / exact-dot-RNE')
        print('         measured-post / pre+bias / QEMU / old-full')
        for p in r['points']:
            print('{}: {} / {}'.format(p['element'],p['pre'],p['pre_RNE']))
            print('  {} / {} / {} / {}'.format(p['post'],p['pre_plus_bias'],p['qemu'],p['old'] or '-'))
            if not report.get('stable'):
                second=next(x for x in report['runs'][1]['points'] if x['element']==p['element'])
                print('  RUN2 measured pre/post: {} / {}'.format(second['pre'],second['post']))
        print('All values above are FP32 bit patterns, not rounded decimals.')
    else:
        for t in r['tiles']:
            m=t['meta']
            print('tile {} frm={} xmcsr={:x} xmfflags={:x}'.format(t['first'],*m[7:10]))
            print('  W/X/out DDR: '+' '.join('0x{:x}'.format(m[i]|m[i+1]<<32) for i in (10,12,14)))
            print('  nonzero inactive W/X lanes: {}/{}'.format(t['inactive_left_nonzero'],t['inactive_right_nonzero']))
        if r['packing_mismatches']:
            print('First packing diff [buffer,tile,row,k,got,expected]:')
            print(' ',r['packing_mismatches'][0])
        else:
            print('First packing differences: none')
        print('Skipped tiles change execution history; compare old-full above.')
        print('Matching packing narrows faults to AMU load/compute/store path;')
        print('it does not observe internal AMU registers or prove final accuracy.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--capture',type=Path,help='Optional previous full-node capture')
    p.add_argument('--offline',type=Path,help='Analyze two probe-N.bin files without board I/O')
    p.add_argument('--report',type=Path,help='Print a saved summary only')
    p.add_argument('--reanalyze',action='store_true',help='Recompute saved runs into analysis-v2.json; use with --report')
    p.add_argument('--page',type=int,choices=(1,2,3),default=1)
    p.add_argument('--load',action='store_true',help='Hold reset, load diagnostic image, release reset')
    p.add_argument('--pbcopy',default='/rv/fpga/bin/pbcopy')
    p.add_argument('--pbload',default='/rv/fpga/bin/pbload')
    p.add_argument('--reset-dir',default='/rv/fpga/bin')
    args=p.parse_args()
    if args.reanalyze and not args.report: p.error('--reanalyze requires --report')
    if args.report and (args.load or args.offline or args.output):
        p.error('--report/--reanalyze cannot be combined with --load, --offline or --output')
    if args.report:
        try:
            report=reanalyze(args.report) if args.reanalyze else load_report(args.report)
        except (OSError,ValueError,KeyError,OverflowError) as exc:
            pages({'id':'unavailable','status':'failed','error':str(exc)},args.page)
            return 1
        pages(report,args.page); return 0
    if not args.output: p.error('--output is required')
    if args.offline and args.load: p.error('--offline cannot load hardware')
    args.output.mkdir(parents=True,exist_ok=True)
    report={'status':'running','mode':'offline' if args.offline else 'physical board','runs':[],
            'analysis_version':ANALYSIS_VERSION}
    def diagnostic(event):
        with (args.output/'diagnostics.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(event)+'\n')
    started=time.monotonic()
    try:
        fixture=assets(); old=old_capture(args.capture,fixture)
        report['old_capture']=str(args.capture) if args.capture else None
        if args.load:
            subprocess.run(['bash',str(ROOT/'tools/load_board.sh'),'--tile-probe','--reset-dir',args.reset_dir,
                            '--pbcopy',args.pbcopy],cwd=ROOT,check=True)
        backend=None
        try:
            if not args.offline:
                # Boot includes CLP selftest; wait before attempting metadata access.
                time.sleep(2)
                backend=ProbeBackend(args.pbcopy,args.pbload,infer_timeout=300,
                                     progress=lambda s:print(s,flush=True),diagnostic=diagnostic)
                report['firmware_info']=backend.info
            for i in (1,2):
                raw=(args.offline/('probe-{}.bin'.format(i))).read_bytes() if args.offline else backend.infer_bytes(fixture[1][0])
                (args.output/('probe-{}.bin'.format(i))).write_bytes(raw)
                report['runs'].append(analyze(raw,fixture,old))
        finally:
            if backend: backend.close()
        report['stable']=report['runs'][0]['sha256']==report['runs'][1]['sha256']
        report['status']='completed'
    except (Exception,KeyboardInterrupt) as exc:
        report.update(status='failed',error=str(exc) or 'Interrupted')
    report['elapsed_seconds']=time.monotonic()-started
    report['id']=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()[:12]
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    pages(report,1)
    print('Report:',args.output,flush=True)
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':
    sys.exit(main())
