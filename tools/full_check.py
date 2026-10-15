"""Python 3.8+ single-start BUS/ZIDANE collection, local replay and screenshot report."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from full_replay import Replay,ROOT,sha
from full_evidence import audit,reusable,read_json
from node_capture import verify_reference_manifest
import tile_probe


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


def run(args):
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists(): raise ValueError('Output already has a final report; use a new directory')
    started=time.monotonic()
    report={'version':1,'status':'running','created':datetime.now().isoformat(),'samples':{},'errors':[],
            'scope':'Two fixed inputs only; split16 matches do not prove RTL equivalence for all inputs'}
    def progress(sample,node,status):
        save(output/'progress.json',{'sample':sample,'node':node,'status':status,
                                    'elapsed_seconds':time.monotonic()-started})
        if node%16==0 or status in ('unexplained','input_unverified','missing'):
            print('AUDIT {} node={:03d} {}'.format(sample,node,status),flush=True)
    try:
        manifest=read_json(ROOT/'reference/nodes/manifest.json'); verify_reference_manifest(manifest)
        prepared=read_json(ROOT/'reference/prepared/manifest.json')['samples']
        # Verify packaged binary, source hashes and native selftest before any reset.
        native=Replay(); native.close()
        report['auditor']=read_json(ROOT/'firmware/host-verify.json')
        report['reference_manifest_sha256']=sha((ROOT/'reference/nodes/manifest.json').read_bytes())
        try: report['tile_gate']=probe_gate(args.probe)
        except Exception as exc: report['tile_gate']={'status':'failed','error':str(exc)}
        save(output/'tile-gate.json',report['tile_gate'])
        stop_board=False
        for sample in ('bus','zidane'):
            directory=None; sample_output=output/sample; sample_output.mkdir(exist_ok=True)
            if not args.fresh: directory=find_reuse(sample,manifest,prepared[sample],output)
            if directory is not None:
                print('Reuse fully verified capture: '+str(directory),flush=True)
            elif not stop_board:
                try:
                    directory,stop_board=capture(args,sample,sample_output)
                    if stop_board:
                        failure=read_json(directory/'summary.json') if (directory/'summary.json').is_file() else {}
                        report['errors'].append(sample+': capture/normal repeat failed; '+failure.get('error','see capture log'))
                except Exception as exc:
                    stop_board=True; report['errors'].append(sample+': '+str(exc))
            if directory is None:
                directory=sample_output/'not-collected'
                report['errors'].append(sample+': no complete collection; board actions stopped')
            result=audit(directory,sample,manifest,prepared[sample],progress=progress)
            report['samples'][sample]=result
            save(sample_output/'analysis.json',result)
            save(output/'checkpoint.json',report)
        report['status']='passed' if (not report['errors'] and report['tile_gate']['status']=='passed'
                           and all(r['passed'] for r in report['samples'].values())) else 'not_passed'
    except BaseException as exc:
        report.update(status='failed'); report['errors'].append(type(exc).__name__+': '+str(exc))
        (output/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['id']=sha(json.dumps(report,sort_keys=True).encode())[:12]
        save(output/'summary.json',report)
        print('FULL CHECK FINISHED: '+report['status']+' ID='+report['id'],flush=True)
        print('Full evidence remains on host: '+str(output),flush=True)
        print(render(report,1),end='',flush=True)
    return 0 if report['status']=='passed' else 1


def render(report,page):
    lines=['FULL MODEL CHECK ID={} PAGE={}/3'.format(report['id'],page),'STATUS: '+report['status']]
    samples=report.get('samples',{})
    if page==1:
        lines+=['Coverage: nodes / valid outputs / verified-input nodes']
        for sample in ('bus','zidane'):
            r=samples.get(sample,{})
            lines+=['{}: {}/334 {}/335 {}/334'.format(sample,r.get('nodes_captured',0),r.get('outputs_captured',0),r.get('inputs_verified_nodes',0))]
            c=r.get('counts',{})
            lines+=['  exact={} split16={} unexplained={}'.format(c.get('exact',0),c.get('split16_match',0),c.get('unexplained',0)),
                    '  input-unverified={} missing={}'.format(c.get('input_unverified',0),c.get('missing',0 if r else 334))]
        lines+=['Saved physical tile gate: '+report.get('tile_gate',{}).get('status','missing'),
                'Elapsed seconds: {:.1f}'.format(report.get('elapsed_seconds',0)),
                'No tolerance; every element compared. Match covers these inputs only.']
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
        for sample in ('bus','zidane'):
            r=samples.get(sample,{}); f=r.get('final',{})
            lines+=['{}: candidate-valid={} normal-repeat={}'.format(sample,f.get('valid',False),f.get('normal_repeat_exact',False)),
                    '  boxes={} QEMU-different-outputs={}'.format(len(f.get('detections',[])),r.get('qemu_different_outputs','?'))]
            q=f.get('qemu',{})
            lines+=['  QEMU final exact={} coord-diff={} score-diff={}'.format(q.get('byte_identical','?'),
                      '{:.5g}'.format(q.get('max_coordinate_difference',0)),'{:.5g}'.format(q.get('max_score_difference',0)))]
            lines+=['  QEMU class sequence match: '+str(q.get('class_sequence_match','?'))]
            lines+=['  capture SHA256: '+f.get('capture_sha256','missing')[:48]]
        lines+=['Drawing: each capture/full-check-annotated.svg (host).',
                'Inspect drawings on host; no GUI needed for numeric checks.',
                'Report retains all local differences; screenshot these 3 pages.']
    else: raise ValueError('Page must be 1, 2 or 3')
    # Deliberately bounded terminal pages; full error text is retained in JSON.
    return '\n'.join(line[:80] for line in lines[:22])+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--report',type=Path)
    p.add_argument('--page',type=int,choices=(1,2,3),default=1)
    p.add_argument('--probe',type=Path)
    p.add_argument('--fresh',action='store_true',help='Recapture both images even if old evidence is complete')
    p.add_argument('--reset-dir',default='/rv/fpga/bin')
    p.add_argument('--pbcopy',default='/rv/fpga/bin/pbcopy'); p.add_argument('--pbload',default='/rv/fpga/bin/pbload')
    p.add_argument('--node-timeout',type=float,default=300); p.add_argument('--tool-timeout',type=float,default=30)
    args=p.parse_args()
    if args.report:
        print(render(read_json(args.report/'summary.json'),args.page),end=''); return 0
    if sys.platform!='linux': p.error('Run collection on the Linux x86_64 host')
    if not args.output: p.error('--output is required for collection')
    if min(args.node_timeout,args.tool_timeout)<=0: p.error('Timeouts must be positive')
    if args.probe is None:
        pointer=ROOT/'yolo26_pc/outputs/host-menu/last-probe.json'
        if pointer.is_file(): args.probe=Path(read_json(pointer)['directory'])
    return run(args)


if __name__=='__main__': raise SystemExit(main())
