"""Python 3.8+ BUS-only collection, local replay and screenshot report."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from full_replay import Replay,ROOT,sha,source_sha
from full_evidence import audit,reusable,read_json
from node_capture import verify_reference_manifest
from png_output import check_renderer
import tile_probe

BUS_ONLY_VERSION = 2


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def probe_gate(directory):
    """Compare native first-layer predictions to both saved physical probes."""
    if directory is None:
        return {'status':'missing','error':'No saved two-run physical tile probe; numerical acceptance remains incomplete'}
    original,source,raws=tile_probe.saved_inputs(directory)
    if source.get('mode')!='physical board': raise ValueError('Probe is not recorded as physical board data')
    fixture=tile_probe.assets(); manifest,inputs,_,_=fixture
    replay=Replay(16)
    try: native=replay.run(manifest['nodes'][0],manifest['tensors'],inputs)[3]
    finally: replay.close()
    runs=[]
    for raw in raws:
        result=tile_probe.analyze(raw,fixture)
        _,records=tile_probe.decode(raw)
        mismatch=0
        for record in records:
            first=record['meta'][0]
            for oc in range(16):
                for col in range(32):
                    pos=4*(oc*43264+first+col); i=4*(oc*32+col)
                    mismatch+=native[pos:pos+4]!=record['after'][i:i+4]
        result['native_post_mismatches']=mismatch
        runs.append(result)
    keys=('weight_bad','input_bad','bias_bad','pre_vs_split16','post_vs_split16')
    passed=raws[0]==raws[1] and all(not r['native_post_mismatches'] and
                  all(not t[k] for t in r['tiles'] for k in keys) for r in runs)
    return {'status':'passed' if passed else 'unexplained','source_report_id':source.get('id'),
            'source_report_sha256':sha(original),
            'directory':str(directory),'raw_sha256':[sha(r) for r in raws],'runs':runs}


def command(args,log):
    print('RUN: '+' '.join(map(str,args)),flush=True)
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen([str(a) for a in args],cwd=ROOT,stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,
                                 encoding='utf-8',errors='replace',bufsize=1)
        try:
            for line in process.stdout:
                stream.write(line); stream.flush(); print(line,end='',flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
        finally: process.stdout.close()


def load(args,output):
    status=command(['bash',ROOT/'tools/load_board.sh','--reset-dir',args.reset_dir,'--pbcopy',args.pbcopy],output/'load.log')
    if status: raise RuntimeError('Firmware load/reset failed; no further board operations (see load.log)')


def find_reuse(sample,manifest,prepared,exclude):
    base=ROOT/'yolo26_pc/outputs'
    patterns=('full-check/*/*/attempt-*/summary.json','node-capture/*/summary.json',
              'host-menu/run-*/capture/summary.json')
    paths=[p.parent for pattern in patterns for p in base.glob(pattern)]
    for directory in sorted(paths,key=lambda p:p.stat().st_mtime,reverse=True):
        if exclude in directory.parents: continue
        if reusable(directory,sample,manifest,prepared): return directory
    return None


def capture(args,sample,output):
    """Only a tagged, recoverable read failure permits one whole-frame retry."""
    for attempt in (1,2):
        directory=output/('attempt-'+str(attempt))
        load(args,output)
        rc=command([sys.executable,'-u',ROOT/'yolo26_pc/node_capture.py','--sample',sample,
                    '--pbcopy',args.pbcopy,'--pbload',args.pbload,'--node-timeout',args.node_timeout,
                    '--tool-timeout',args.tool_timeout,'--output',directory],output/('capture-'+str(attempt)+'.log'))
        report=read_json(directory/'summary.json') if (directory/'summary.json').exists() else {}
        if not rc and report.get('status')=='completed': return directory,False
        if attempt==1 and report.get('retryable_capture_failure') is True:
            print('Recoverable PCIe read failure: retry the whole frame once.',flush=True)
            continue
        return directory,True


def new_report():
    return {'version':BUS_ONLY_VERSION,'sample':'bus','status':'running','created':datetime.now().isoformat(),
            'samples':{},'errors':[],'tile_gate':{'status':'not_run'},
            'scope':'One fixed bus input only; matches do not prove equivalence for all inputs'}


def assets(report):
    manifest=read_json(ROOT/'reference/nodes/manifest.json'); verify_reference_manifest(manifest)
    prepared=read_json(ROOT/'reference/prepared/manifest.json')['samples']['bus']
    # Verify packaged binary, source hashes and renderer before any reset.
    native=Replay(); native.close(); check_renderer()
    report['auditor']=read_json(ROOT/'firmware/host-verify.json')
    report['reference_manifest_sha256']=source_sha(ROOT/'reference/nodes/manifest.json')
    report['reference_normalization']='CRLF-to-LF'
    return manifest,prepared


def finish(report,started):
    report['elapsed_seconds']=time.monotonic()-started
    report['id']=sha(json.dumps(report,sort_keys=True).encode())[:12]


def bus_capture(directory):
    """Find raw BUS evidence, including an unfinished legacy parent job."""
    candidates=[directory]
    # Prefer a completed retry over an incomplete earlier attempt.
    candidates+=sorted(directory.glob('bus/attempt-*'),reverse=True)
    for name in ('summary.json','checkpoint.json','bus/analysis.json'):
        path=directory/name
        if not path.is_file(): continue
        record=read_json(path)
        result=record if name=='bus/analysis.json' else record.get('samples',{}).get('bus',{})
        if result.get('directory'): candidates.append(Path(result['directory']))
    for candidate in candidates:
        path=candidate/'summary.json'
        if path.is_file():
            record=read_json(path)
            if record.get('sample')=='bus' and record.get('status')=='completed': return candidate
    # audit() will enumerate the missing evidence; a derived report cannot pass.
    return directory/'bus/not-collected'


def saved_report(directory):
    """Display an existing BUS analysis without replaying nodes or touching the board."""
    for name in ('bus-analysis.json','summary.json','checkpoint.json','bus/analysis.json'):
        path=directory/name
        if not path.is_file(): continue
        raw=path.read_bytes(); source=json.loads(raw)
        if name=='bus-analysis.json' and source.get('status')=='failed':
            return dict(source,mode='saved analysis',source={'path':str(path),'sha256':sha(raw)})
        result=source if name=='bus/analysis.json' else source.get('samples',{}).get('bus')
        if not result or not isinstance(result.get('rows'),list): continue
        gate=source.get('tile_gate')
        if gate is None:
            gate=read_json(directory/'tile-gate.json') if (directory/'tile-gate.json').is_file() else {'status':'missing'}
        complete=(result.get('passed') is True and result.get('nodes_captured')==334 and
                  result.get('outputs_captured')==335 and result.get('inputs_verified_nodes')==334 and
                  len(result['rows'])==334 and result.get('final',{}).get('normal_repeat_exact') is True and
                  result.get('final',{}).get('valid') is True)
        return {'version':BUS_ONLY_VERSION,'sample':'bus','id':sha(raw+b'bus-saved-view')[:12],
                'status':'passed' if complete and gate.get('status')=='passed' else 'not_passed',
                'samples':{'bus':result},'tile_gate':gate,'errors':result.get('errors',[]),
                'elapsed_seconds':source.get('elapsed_seconds',0),'mode':'saved analysis',
                'source':{'path':str(path),'id':source.get('id'),'sha256':sha(raw)}}
    raise LookupError('No saved BUS node analysis in '+str(directory))


def reanalyze(args):
    """Re-audit raw BUS data without modifying any original reports or board state."""
    directory=args.report; started=time.monotonic(); report=new_report()
    report['mode']='offline reanalysis'
    try:
        source_files={}
        for name in ('summary.json','checkpoint.json','tile-gate.json'):
            path=directory/name
            if path.is_file(): source_files[name]=sha(path.read_bytes())
        report['source']={'directory':str(directory),'files_sha256':source_files}
        manifest,prepared=assets(report)
        gate=read_json(directory/'tile-gate.json') if (directory/'tile-gate.json').is_file() else {}
        probe=Path(gate['directory']) if gate.get('directory') else args.probe
        try: report['tile_gate']=probe_gate(probe)
        except Exception as exc: report['tile_gate']={'status':'failed','error':str(exc)}
        raw=bus_capture(directory)
        if (raw/'summary.json').is_file():
            report['source']['capture_summary_sha256']=sha((raw/'summary.json').read_bytes())
        report['samples']['bus']=audit(raw,'bus',manifest,prepared)
        report['status']='passed' if (report['tile_gate']['status']=='passed' and
                                     report['samples']['bus']['passed']) else 'not_passed'
    except Exception as exc:
        report.update(status='failed'); report['errors'].append(type(exc).__name__+': '+str(exc))
    finish(report,started)
    save(directory/'bus-analysis.json',report)
    return report


def run(args):
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists(): raise ValueError('Output already has a final report; use a new directory')
    started=time.monotonic()
    report=new_report()
    def progress(sample,node,status):
        save(output/'progress.json',{'sample':sample,'node':node,'status':status,
                                    'elapsed_seconds':time.monotonic()-started})
        if node%16==0 or status in ('unexplained','input_unverified','missing'):
            print('AUDIT {} node={:03d} {}'.format(sample,node,status),flush=True)
    try:
        manifest,prepared=assets(report)
        try: report['tile_gate']=probe_gate(args.probe)
        except Exception as exc: report['tile_gate']={'status':'failed','error':str(exc)}
        save(output/'tile-gate.json',report['tile_gate'])
        directory=None; sample_output=output/'bus'; sample_output.mkdir(exist_ok=True)
        if not args.fresh: directory=find_reuse('bus',manifest,prepared,output)
        if directory is not None:
            print('Reuse fully verified BUS capture: '+str(directory),flush=True)
        else:
            try:
                directory,failed=capture(args,'bus',sample_output)
                if failed:
                    failure=read_json(directory/'summary.json') if (directory/'summary.json').is_file() else {}
                    report['errors'].append('bus: capture/normal repeat failed; '+failure.get('error','see capture log'))
            except Exception as exc:
                report['errors'].append('bus: '+str(exc))
        if directory is None:
            directory=sample_output/'not-collected'
            report['errors'].append('bus: no complete collection; board actions stopped')
        result=audit(directory,'bus',manifest,prepared,progress=progress)
        report['samples']['bus']=result
        save(sample_output/'analysis.json',result)
        save(output/'checkpoint.json',report)
        report['status']='passed' if (not report['errors'] and report['tile_gate']['status']=='passed'
                           and all(r['passed'] for r in report['samples'].values())) else 'not_passed'
    except BaseException as exc:
        report.update(status='failed'); report['errors'].append(type(exc).__name__+': '+str(exc))
        (output/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        finish(report,started)
        save(output/'summary.json',report)
        print('FULL CHECK FINISHED: '+report['status']+' ID='+report['id'],flush=True)
        print('Full evidence remains on host: '+str(output),flush=True)
        print(render(report,1),end='',flush=True)
    return 0 if report['status']=='passed' else 1


def render(report,page):
    lines=['FULL MODEL CHECK ID={} PAGE={}/3'.format(report['id'],page),'STATUS: '+report['status']]
    if report.get('mode')=='saved analysis': lines.append('Saved analysis; no new inference or numerical replay.')
    samples={k:v for k,v in report.get('samples',{}).items() if k=='bus'}
    if page==1:
        lines+=['Coverage: nodes / valid outputs / verified-input nodes']
        for sample in ('bus',):
            r=samples.get(sample,{})
            lines+=['{}: {}/334 {}/335 {}/334'.format(sample,r.get('nodes_captured',0),r.get('outputs_captured',0),r.get('inputs_verified_nodes',0))]
            c=r.get('counts',{})
            lines+=['  exact={} split16={} unexplained={}'.format(c.get('exact',0),c.get('split16_match',0),c.get('unexplained',0)),
                    '  input-unverified={} missing={}'.format(c.get('input_unverified',0),c.get('missing',0 if r else 334))]
        lines+=['Saved physical tile gate: '+report.get('tile_gate',{}).get('status','missing'),
                'Elapsed seconds: {:.1f}'.format(report.get('elapsed_seconds',0)),
                'No tolerance; every element compared. Match covers this input only.']
    elif page==2:
        lines+=['Exceptions (full details in host analysis.json):']
        tile=report.get('tile_gate',{})
        if tile.get('status')!='passed':
            lines+=['Tile gate: '+tile.get('status','missing'),tile.get('error','See tile-gate.json')[:78]]
        for sample,r in samples.items():
            bad=[n for n in r['rows'] if n['status'] in ('unexplained','input_unverified','missing')]
            lines+=['{}: {} exceptional nodes'.format(sample,len(bad))]
            for n in bad[:4]:
                lines+=['  node={} {} {}'.format(n['node'],n['kind'],n['status'])]
                d=next((o.get('difference') for o in n['outputs'] if o.get('difference')),None)
                if d: lines+=['    differing={} maxULP={}'.format(d['different_elements'],d['max_ulp_difference'])]
            lines.extend('  '+e[:70] for e in r.get('errors',[])[:1])
        lines.extend(e[:78] for e in report.get('errors',[])[:2])
    elif page==3:
        for sample in ('bus',):
            r=samples.get(sample,{}); f=r.get('final',{})
            lines+=['{}: candidate-valid={} normal-repeat={}'.format(sample,f.get('valid',False),f.get('normal_repeat_exact',False)),
                    '  boxes={} QEMU-different-outputs={}'.format(len(f.get('detections',[])),r.get('qemu_different_outputs','?'))]
            q=f.get('qemu',{})
            lines+=['  QEMU final exact={} coord-diff={} score-diff={}'.format(q.get('byte_identical','?'),
                      '{:.5g}'.format(q.get('max_coordinate_difference',0)),'{:.5g}'.format(q.get('max_score_difference',0)))]
            lines+=['  QEMU class sequence match: '+str(q.get('class_sequence_match','?'))]
            lines+=['  capture SHA256: '+f.get('capture_sha256','missing')[:48]]
        lines+=['Drawing: each capture/full-check-annotated.png (host).',
                'Inspect drawings on host; no GUI needed for numeric checks.',
                'Report retains all local differences; screenshot these 3 pages.',
                'Scope: BUS only. Next stage may proceed after this check passes.']
    else: raise ValueError('Page must be 1, 2 or 3')
    # Deliberately bounded terminal pages; full error text is retained in JSON.
    return '\n'.join(line[:80] for line in lines[:22])+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--report',type=Path)
    p.add_argument('--reanalyze',action='store_true',help='Re-audit saved BUS raw evidence offline')
    p.add_argument('--page',type=int,choices=(1,2,3),default=1)
    p.add_argument('--probe',type=Path)
    p.add_argument('--fresh',action='store_true',help='Recapture BUS even if old evidence is complete')
    p.add_argument('--reset-dir',default='/rv/fpga/bin')
    p.add_argument('--pbcopy',default='/rv/fpga/bin/pbcopy'); p.add_argument('--pbload',default='/rv/fpga/bin/pbload')
    p.add_argument('--node-timeout',type=float,default=300); p.add_argument('--tool-timeout',type=float,default=30)
    args=p.parse_args()
    if args.probe is None:
        pointer=ROOT/'yolo26_pc/outputs/host-menu/last-probe.json'
        if pointer.is_file(): args.probe=Path(read_json(pointer)['directory'])
    if args.reanalyze and not args.report: p.error('--reanalyze requires --report')
    if args.report:
        if args.output or args.fresh: p.error('--report cannot be combined with --output/--fresh')
        if args.reanalyze: report=reanalyze(args)
        else:
            try: report=saved_report(args.report)
            except LookupError as exc: p.error(str(exc)+'; --reanalyze can analyze existing raw captures offline')
        print(render(report,args.page),end=''); return 0
    if sys.platform!='linux': p.error('Run collection on the Linux x86_64 host')
    if not args.output: p.error('--output is required for collection')
    if min(args.node_timeout,args.tool_timeout)<=0: p.error('Timeouts must be positive')
    return run(args)


if __name__=='__main__': raise SystemExit(main())
