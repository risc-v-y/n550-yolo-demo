"""Replay one preserved first-layer input (or BUS). Python 3.8, stdlib only."""
import argparse
import hashlib
import json
import math
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
import collect_board_diag as diag
from board_diagnostics import DiagnosticLog
from pcie_backend import (BASE,MAILBOX,READY,DONE,BUSY,INPUT_BYTES,PCIeBackend,PBTools,
                          INFO,STATUS,STATUS_KEYS,MAGIC as PCIE_MAGIC,checked,
                          MODEL_DDR_BASE,MODEL_DDR_END)
from board_diagnostics import decode_snapshot

MAGIC=0x314b435036325959
WIRE=struct.Struct('<32Q')
FIELDS=('magic version bytes generation session frame node stage first columns column inner '
        'row vl events error input cache output cache_bytes row0 row1 row2 converted '
        'next_column next_inner row_visits elapsed started updated k checksum').split()
STAGES=('none begin prepare.begin prepare.done row.check row.fill.begin row.fill.done row.hit '
        'rows.done index.begin index.done gather.begin gather.done store.begin store.done '
        'column.done block.done GUARD.FAIL').split()
REASONS={1:'column out of range',2:'column did not advance',3:'repeated row checks',
         4:'K position did not advance',5:'invalid VL/K increment',6:'column K incomplete',
         7:'block columns incomplete',8:'event bound exceeded',9:'400M-cycle block limit'}


class UnstableProbe(ValueError): pass
class ProbeStalled(RuntimeError): pass


def decode(raw,session,frame):
    if len(raw)!=WIRE.size: raise ValueError('Pack snapshot length mismatch')
    words=WIRE.unpack(raw)
    if not any(words): return None
    result=dict(zip(FIELDS,words))
    if (result['magic'],result['version'],result['bytes'])!=(MAGIC,1,WIRE.size):
        raise ValueError('Pack snapshot magic/version mismatch; load current firmware')
    value=14695981039346656037
    for word in words[:-1]: value=((value^word)*1099511628211)&((1<<64)-1)
    if result['generation']&1 or not result['generation'] or value!=result['checksum']:
        raise UnstableProbe('Live pack snapshot changed during read')
    if (result['session'],result['frame'],result['node'],result['first'])!=(session,frame,0,0):
        raise ValueError('Pack snapshot session/frame/node mismatch')
    if not 1<=result['stage']<len(STAGES) or not 1<=result['columns']<=32 or not 1<=result['k']<=4096:
        raise ValueError('Invalid pack snapshot progress fields')
    if result['next_column']>result['columns'] or result['next_inner']>result['k']:
        raise ValueError('Invalid pack snapshot progress counters')
    result['stage_name']=STAGES[result['stage']]
    result['reason']=REASONS.get(result['error'],'unknown guard reason' if result['error'] else '')
    result['row']=result['row']-(1<<64) if result['row']>=1<<63 else result['row']
    result['sha256']=hashlib.sha256(raw).hexdigest()
    return result


def render(report):
    rows=['PACK PROBE: '+report['status'].upper(),
          'Diagnostic only; excluded from performance ranking.',
          'Run: '+report['id'],
          'Frame: '+report.get('frame_status','not submitted')]
    rows.append('Input: '+report.get('input',{}).get('source','not prepared'))
    sample=report['samples'][-1]['probe'] if report.get('samples') else None
    if sample:
        rows+=['Last step: '+sample['stage_name'],
               'col={}/{} K={}/{} vl={} row={}'.format(sample['column'],sample['columns'],sample['inner'],sample['k'],sample['vl'],sample['row']),
               'Completed columns={} next-K={} row-checks={}'.format(sample['next_column'],sample['next_inner'],sample['row_visits']),
               'Converted={} events={} generation={}'.format(sample['converted'],sample['events'],sample['generation']),
               'Block elapsed cycles={}'.format(sample['elapsed']),
               'input=0x{:x} cache=0x{:x} bytes={}'.format(sample['input'],sample['cache'],sample['cache_bytes']),
               'Guard: '+(sample['reason'] or 'none')]
    else: rows.append('No verified first-block snapshot; inspect UART/startup error.')
    if report.get('last_board_diagnostic'):
        d=report['last_board_diagnostic']
        rows.append('Board: node={} phase={} error=0x{:x}'.format(d['node'],d['phase_name'],d['error']))
    rows+=['Verified samples={} unstable reads={}'.format(len(report.get('samples',[])),report.get('unstable_reads',0))]
    if report.get('error'): rows.append('Error: '+report['error'])
    if report.get('snapshot_error'): rows.append('Final snapshot: '+report['snapshot_error'])
    rows+=['UART: last [PACK] begin without done narrows the interval.',
           'Instrumentation adds calls/synchronization and may change behavior.',
           'Passing this probe alone does not prove the original stall fixed.',
           'Files: report.json, terminal details in diagnostics.jsonl.']
    return '\n'.join(textwrap.shorten(row,width=80,placeholder='...') for row in rows)+'\n'


def preserved_input(args,report,log):
    """Read the original in-flight first-layer input before any reset/write."""
    link=PBTools(args.pbcopy,args.pbload,timeout=args.tool_timeout,diagnostic=log.emit)
    try:
        deadline=time.monotonic()+args.tool_timeout*6
        info=INFO.unpack(checked(link.read(MAILBOX,64,deadline)))
        if info[:3]!=(PCIE_MAGIC,1,256) or info[7]!=INPUT_BYTES or info[9] not in (816,832):
            raise ValueError('Cannot identify current firmware before preserving input; no reset performed')
        state=dict(zip(STATUS_KEYS,STATUS.unpack(checked(link.read(MAILBOX+192,64,deadline)))))
        if state['magic']!=PCIE_MAGIC or state['version']!=1: raise ValueError('Invalid current PCIe state')
        if state['state'] not in (READY,DONE,BUSY):
            raise ValueError('Core is booting, failed or capture-paused; no reset performed')
        if state['state']!=BUSY or not state['frame']: return None
        diag_address=info[6]
        if diag_address%64 or not BASE<=diag_address<=MAILBOX-info[9]: raise ValueError('Invalid live diagnostic address')
        d=decode_snapshot(link.read(diag_address,info[9],deadline)[:816])
        if (d['node'],d['phase'],d['error'],d['frame'])!=(0,5,0,state['frame']):
            raise ValueError('Busy core is outside first-layer input packing; preserve its evidence before resetting')
        request=struct.unpack_from('<QQ11I',checked(link.read(MAILBOX+64,64,deadline)))
        if request[:5]!=(state['session'],state['frame'],2,state['sequence'],INPUT_BYTES):
            raise ValueError('Current input request identity mismatch; no reset performed')
        address=info[4]
        if address%64 or not (BASE<=address<=MAILBOX-INPUT_BYTES or MODEL_DDR_BASE<=address<=MODEL_DDR_END-INPUT_BYTES):
            raise ValueError('Current input address outside approved DDR')
        data=link.read(address,INPUT_BYTES,deadline)
        if len(data)!=INPUT_BYTES or zlib.crc32(data)!=request[5]:
            raise ValueError('Current input CRC differs from its RUN request; original files preserved, no reset performed')
        confirmed=dict(zip(STATUS_KEYS,STATUS.unpack(checked(link.read(MAILBOX+192,64,deadline)))))
        if confirmed!=state: raise ValueError('Current state changed during input preservation; no reset performed')
        report['input']=dict(source='preserved first-layer DDR request',session=state['session'],frame=state['frame'],
                             address=address,crc32=request[5],sha256=hashlib.sha256(data).hexdigest())
        print('Preserved original frame {} input; CRC verified.'.format(state['frame']),flush=True)
        return data
    finally: link.close()


class ProbeBackend(PCIeBackend):
    def __init__(self,address,output,report,stall_timeout,**kwargs):
        self.probe_address=address; self.probe_output=output; self.probe_report=report
        self.probe_next_read=0.; self.probe_last_progress=None; self.probe_progress_at=None
        self.probe_stall_timeout=stall_timeout
        super().__init__(**kwargs)

    def sample_probe(self,deadline,enforce_stall=True):
        for attempt in range(3):
            raw=self.link.read(self.probe_address,WIRE.size,deadline)
            try: sample=decode(raw,self.session,self.frame)
            except UnstableProbe:
                self.probe_report['unstable_reads']+=1
                continue
            if sample is None: return None
            index=len(self.probe_report['samples'])
            path=self.probe_output/('probe-{:03d}.bin'.format(index))
            path.write_bytes(raw)
            self.probe_report['samples'].append(dict(probe=sample,file=path.name))
            self.diagnostic({'type':'pack_probe','message':'PACK step={} col={} K={} seq={}'.format(
                sample['stage_name'],sample['column'],sample['inner'],sample['generation']),'probe':sample})
            progress=(sample['next_column'],sample['next_inner'],sample['converted'])
            now=time.monotonic()
            if progress!=self.probe_last_progress:
                self.probe_last_progress=progress; self.probe_progress_at=now
            elif enforce_stall and sample['stage']!=16 and not sample['error'] and now-self.probe_progress_at>=self.probe_stall_timeout:
                raise ProbeStalled('No completed pack progress for {:.0f}s at {}'.format(self.probe_stall_timeout,sample['stage_name']))
            return sample
        return None  # Bounded live-read races; saved samples remain intact.

    def _pause(self,deadline):
        if (self.frame and self.last_status and self.last_status['state']==BUSY and
                self.last_status['session']==self.session and self.last_status['frame']==self.frame and
                time.monotonic()>=self.probe_next_read):
            self.probe_next_read=time.monotonic()+2
            self.sample_probe(deadline)
        super()._pause(deadline)


def execute(args,output):
    report=dict(version=1,id=output.name,status='starting',samples=[],unstable_reads=0,
                inference_timeout=args.infer_timeout,stall_timeout=args.stall_timeout,
                note='No board reset after error/timeout; core may still be running.')
    backend=None
    log=DiagnosticLog(output)
    started=time.monotonic()
    try:
        firmware,_,prefix=diag.firmware_info(ROOT,'demo-pcie')
        symbols=diag.elf_symbols((ROOT/'firmware/demo-pcie.elf').read_bytes(),{'board_pack_probe':WIRE.size})
        address=symbols['board_pack_probe']
        if address%64 or not BASE<=address<=MAILBOX-WIRE.size: raise ValueError('Invalid pack probe symbol address')
        report.update(firmware=firmware,probe_address=address)
        manifest=json.loads((ROOT/'reference/prepared/manifest.json').read_text())['samples']['bus']
        data=(ROOT/manifest['input']).read_bytes()
        if len(data)!=INPUT_BYTES or hashlib.sha256(data).hexdigest()!=manifest['input_sha256']:
            raise ValueError('Frozen BUS input hash/length mismatch')
        if getattr(args,'input_source','bus')=='current':
            current=preserved_input(args,report,log)
            if current is not None: data=current
        if 'input' not in report:
            report['input']=dict(source='frozen BUS',sha256=hashlib.sha256(data).hexdigest())
        (output/'source-input.bin').write_bytes(data)
        if args.load:
            subprocess.run(['bash',str(ROOT/'tools/load_board.sh'),'--reset-dir',args.reset_dir,
                            '--pbcopy',args.pbcopy],cwd=ROOT,check=True)
        backend=ProbeBackend(address,output,report,args.stall_timeout,pbcopy=args.pbcopy,pbload=args.pbload,
                             infer_timeout=args.infer_timeout,tool_timeout=args.tool_timeout,
                             diagnostic=log.emit,progress=lambda message: print(message,flush=True),trace=True)
        if backend.link.read(BASE,len(prefix),time.monotonic()+args.tool_timeout)!=prefix:
            raise ValueError('Loaded DDR program prefix differs from selected firmware')
        result=backend.infer_bytes(data)
        (output/'candidates.bin').write_bytes(result)
        report['frame_status']='completed (7200 bytes, transport/values checked)'
        report['timing']=backend.last_timing
        report['status']='completed'
    except KeyboardInterrupt:
        report.update(status='stopped',error='Interrupted by user')
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc))
        report['frame_status']='incomplete'
        log.exception(exc)
    finally:
        if backend is not None:
            try:
                deadline=time.monotonic()+args.tool_timeout*4
                if backend.frame: backend.sample_probe(deadline,enforce_stall=False)
                backend._snapshot(deadline)
                report['last_board_diagnostic']=backend.last_diagnostic
            except BaseException as exc: report['snapshot_error']=type(exc).__name__+': '+str(exc)
            finally: backend.close()
        sample=report['samples'][-1]['probe'] if report['samples'] else None
        if report['status']=='completed' and (sample is None or sample['stage']!=16 or sample['error']):
            report.update(status='failed',error='Inference completed without a valid block.done probe')
        report['elapsed_seconds']=time.monotonic()-started
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        page=render(report)
        (output/'pack-report.txt').write_text(page,encoding='utf-8')
        log.close()
        print(page,end='',flush=True)
        print('Output:',output,flush=True)
    return 0 if report['status']=='completed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load',action='store_true')
    parser.add_argument('--pbcopy',default='/rv/fpga/bin/pbcopy')
    parser.add_argument('--pbload',default='/rv/fpga/bin/pbload')
    parser.add_argument('--reset-dir',default='/rv/fpga/bin')
    parser.add_argument('--infer-timeout',type=float,default=120)
    parser.add_argument('--tool-timeout',type=float,default=30)
    parser.add_argument('--stall-timeout',type=float,default=20)
    parser.add_argument('--input-source',choices=('current','bus'),default='current',
                        help='Preserve current first-layer input before reset; idle board uses frozen BUS')
    args=parser.parse_args()
    if sys.platform!='linux': parser.error('Run on the Linux host connected to S2C')
    if any(not math.isfinite(value) or value<=0 for value in (args.infer_timeout,args.tool_timeout,args.stall_timeout)):
        parser.error('Timeouts must be finite and positive')
    from host_control import guard,workspace,board_lock,save_json
    base=ROOT/'yolo26_pc/outputs/pack-probe'; base.mkdir(parents=True,exist_ok=True)
    with guard(workspace()/'control.lock'):
        # Check for another owner before loading; loader/backend acquire the
        # same board lock themselves. No broad process kill or forced reset.
        with guard(board_lock()): pass
        output=Path(tempfile.mkdtemp(prefix='run-',dir=str(base)))
        save_json(base/'latest.json',dict(directory=str(output)))
        return execute(args,output)


if __name__=='__main__': sys.exit(main())
