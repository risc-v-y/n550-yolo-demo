"""Show a saved node capture as three 80-column, screenshot-sized pages.

Python standard library only. Reads host files; never accesses the board.
The latest-run pointer follows completion order, not the host's calendar.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import textwrap

ROOT=Path(__file__).resolve().parent.parent
LATEST=ROOT/'yolo26_pc/outputs/node-capture/latest.json'


def remember_report(directory):
    LATEST.parent.mkdir(parents=True,exist_ok=True)
    temporary=LATEST.with_suffix('.tmp')
    temporary.write_text(json.dumps({'directory':str(directory.resolve())})+'\n',encoding='utf-8')
    temporary.replace(LATEST)


def flag(value):
    if value is True or value=='True': return 'YES'
    if value is False or value=='False': return 'NO'
    return 'unknown'


def number(value):
    return f'{value:.6g}' if isinstance(value,(float,int)) else 'unknown'


def short(value, width=80):
    text=str(value).encode('ascii','backslashreplace').decode('ascii')
    text=' '.join(text.split())
    return text if len(text)<=width else text[:width-3]+'...'


def load_report(directory=None):
    if directory is None:
        if not LATEST.exists():
            raise ValueError('No latest capture pointer. Supply the saved result directory.')
        directory=Path(json.loads(LATEST.read_text(encoding='utf-8'))['directory'])
    directory=Path(directory).resolve()
    raw=(directory/'summary.json').read_bytes()
    summary=json.loads(raw)
    first_path=directory/'first_difference.json'
    first=json.loads(first_path.read_text(encoding='utf-8')) if first_path.exists() else None
    rows=[]
    if (directory/'nodes.csv').exists():
        with (directory/'nodes.csv').open(encoding='utf-8',newline='') as stream:
            rows=list(csv.DictReader(stream))
    return directory,summary,first,rows,hashlib.sha256(raw).hexdigest()[:12]


def input_descriptors(first):
    if 'input_descriptors' in first:
        return first['input_descriptors']
    # Older capture reports predate embedded input shapes. Only use a matching
    # frozen node/module/output when displaying their graph metadata.
    path=ROOT/'reference/nodes/manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text(encoding='utf-8'))
        node=manifest['nodes'][first['node']]
        if node['module']==first.get('module') and first['tensor'] in node['outputs']:
            return [manifest['tensors'][tid] for tid in node['inputs']]
    return [{'id':item['tensor']} for item in first.get('inputs',[])]


def shape(value):
    return 'unknown' if value is None else ('x'.join(map(str,value)) or 'scalar')


def failure_lines(summary, page):
    diagnostic=summary.get('last_diagnostic') or {}
    status=summary.get('last_status') or {}
    if page==2:
        lines=[f"Exception: {summary.get('exception','unknown')}",
               f"Board state={status.get('state','?')} seq={status.get('sequence','?')} frame={status.get('frame','?')}",
               f"PCIe error={status.get('error','?')} detail={status.get('detail','?')}",
               f"Last node={diagnostic.get('node','?')} phase={diagnostic.get('phase_name','?')}",
               f"Board error={diagnostic.get('error','?')} detail={diagnostic.get('detail','?')}",
               'Error text (continued on page 3 if long):']
    else:
        lines=['Error text, continuation:']
    error=' '.join(str(summary.get('error','No error text recorded.')).split())
    wrapped=textwrap.wrap(error.encode('ascii','backslashreplace').decode('ascii'),80)
    start=0 if page==2 else 13
    capacity=13 if page==2 else 16
    lines+=wrapped[start:start+capacity] or ['(none)']
    if page==3:
        if len(wrapped)>start+capacity:
            lines.append('Further error text remains in summary.json on host.')
        lines.append('Capture incomplete; no numerical conclusion.')
    return lines


def render_page(directory=None, page=1, view='status'):
    directory,summary,first,rows,identity=load_report(directory)
    title='YOLO NODE REPORT' if view=='status' else 'YOLO NODE DIFFERENCE'
    if view=='difference' and summary.get('status')!='completed':
        title+=' (PARTIAL CAPTURE)'
    lines=[f'{title} - PAGE {page}/3',
           f"ID={identity} sample={summary.get('sample','?')} status={summary.get('status','?')}"]
    completed={r['node'] for r in rows}
    differences={r['node'] for r in rows if r.get('byte_identical')=='False'}
    if page==1:
        reference=summary.get('final_reference') or {}
        normal=summary.get('normal_repeat') or {}
        last=summary.get('last_diagnostic') or {}
        lines += [f"Nodes captured: {summary.get('nodes_captured',len(completed))}/334",
                  f"Tensors captured: {summary.get('tensors_captured',len(rows))}/335",
                  f"Different nodes: {summary.get('different_nodes',len(differences))}",
                  f"Last captured node: {max(map(int,completed)) if completed else 'none'}",
                  f"Elapsed seconds: {number(summary.get('elapsed_seconds'))}",
                  f"Input SHA256: {summary.get('input_sha256','unknown')}",
                  f"Final bytes match QEMU: {flag(reference.get('byte_identical'))}",
                  f"Candidate class ranks match: {flag(reference.get('class_sequence_match'))}",
                  f"Rank-wise max coord/score diff: {number(reference.get('max_coordinate_difference'))} / {number(reference.get('max_score_difference'))}",
                  f"Displayed detections: {summary.get('detections','unknown')}",
                  f"Ordinary repeat == capture: {flag(normal.get('byte_identical_to_capture'))}"]
        if first:
            lines += [f"First difference: node={first['node']} op={first['kind']} tensor={first['tensor']}",
                      f"First-node input full bytes == reference: {flag(first.get('same_input_full_bytes_verified'))}"]
        else:
            lines.append('First difference: none recorded (see capture status above)')
        if summary.get('status')=='failed':
            lines += [short(f"ERROR: {summary.get('error',summary.get('exception','unknown'))}"),
                      f"Last board node={last.get('node','?')} error={last.get('error','?')}; details: page 2"]
        lines.append('Cause not established; stable results do not prove rounding.')
    elif summary.get('status')=='failed' and view=='status':
        lines+=failure_lines(summary,page)
    elif not first:
        lines+=['No first differing tensor was recorded.',
                'See page 1 for completeness and ordinary-repeat status.',
                'No numerical replay is needed/available for this report.']
    elif page==2:
        metrics=first['metrics']
        lines += [f"Node={first['node']} op={first['kind']} tensor={first['tensor']}",
                  short(f"Module: {first.get('module','unknown')}"),
                  f"Output shape: {shape(first.get('shape'))}"]
        checks={item['tensor']:item for item in first.get('inputs',[])}
        for item in input_descriptors(first)[:4]:
            tid=item['id']; dtype={1:'FP32',2:'I64'}.get(item.get('dtype'),'?')
            exact=flag(checks.get(tid,{}).get('byte_identical_to_reference'))
            lines.append(f"Input t{tid} {dtype} {shape(item.get('shape'))} full-match={exact}")
        lines += [f"All input full bytes match: {flag(first.get('same_input_full_bytes_verified'))}",
                  f"Different elements: {metrics['different_elements']}/{metrics['count']}",
                  f"Abs max/mean/RMSE: {number(metrics['max_absolute_difference'])} / {number(metrics['mean_absolute_difference'])} / {number(metrics['rmse'])}",
                  f"Max ULP={metrics['max_ulp_difference']} nonfinite A/R={metrics['nonfinite_actual']}/{metrics['nonfinite_reference']}"]
        samples=metrics.get('samples',[])
        selected=samples if len(samples)<=3 else samples[:2]+samples[-1:]
        for sample in selected:
            lines.append(f"Element {sample['element']} index={sample.get('index','?')} ULP={sample.get('ulp','n/a')}")
            if 'actual_bits' in sample:
                lines.append(f" A={number(sample['actual'])} [{sample['actual_bits']}] R={number(sample['reference'])} [{sample['reference_bits']}]")
            else:
                lines.append(f" A={sample['actual']} R={sample['reference']}")
        lines.append('A=board R=QEMU; hex is raw word bits. Metrics use ALL elements.')
    else:
        replay=first.get('replay') or {}
        results=replay.get('results',[])
        lines += [f"First node={first['node']} op={first['kind']} tensor={first['tensor']}",
                  f"Replay: {replay.get('status','not available')}; sampled outputs={len(results)}"]
        rules=[('Q','qemu_fp64_dot_fp32_per_k32'),('S','sequential_fp32'),('T','pairwise_fp32_k32')]
        if results:
            lines += ['Q=FP64 dot, FP32 per K<=32; S=sequential FP32',
                      'T=pairwise FP32 tree in K=32 blocks',
                      'Rule matches: board / reference (selected elements only)']
            for label,key in rules:
                matches=[sum(bool(r['rules'][key].get(field)) for r in results)
                         for field in ('matches_actual_bits','matches_reference_bits')]
                lines.append(f" {label}: {matches[0]}/{len(results)} / {matches[1]}/{len(results)}")
            lines.append('Element    Board    Reference Q        S        T  (FP32 hex)')
            selected=results if len(results)<=4 else results[:3]+results[-1:]
            for result in selected:
                words=[int.from_bytes(bytes.fromhex(result[key]),'little')
                       for key in ('actual_bits_le','reference_bits_le')]
                words += [struct.unpack('<I',struct.pack('<f',result['rules'][key]['value']))[0] for _,key in rules]
                lines.append(f"{result['element']:<10} "+' '.join(f'{word:08x}' for word in words))
            lines += [f"First sampled dot: K={results[0]['k']} bias={number(results[0]['bias'])}",
                      'Replays use captured actual inputs, rounded to FP16.']
        else:
            lines+=textwrap.wrap(short(replay.get('reason','No replay reason recorded.'),240),80)
        lines.append('Candidate matches do not establish the FPGA accumulation rule.')
    # Keep one page readable in an 80x24 terminal, leaving a shell prompt row.
    if len(lines)>22:
        raise ValueError('Report page exceeds the screenshot line budget')
    return '\n'.join(short(line) for line in lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',type=Path,help='Saved result directory; default: latest completed report')
    parser.add_argument('--page',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--view',choices=('status','difference'),default='status',
                        help='difference: show saved first-node metrics/replay even after a later capture failure')
    args=parser.parse_args()
    try:
        print(render_page(args.directory,args.page,args.view),end='')
    except (OSError,ValueError,KeyError,IndexError,TypeError) as exc:
        parser.exit(1,f'Report error: {exc}\n')


if __name__=='__main__':
    main()
