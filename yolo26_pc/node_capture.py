"""Automatically collect board nodes, compare on the host and bundle small results.

Run on the Linux host connected to PCIe. Full data remain on that host;
terminal report pages can be returned as screenshots. A compact ZIP is optional.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
import zipfile
import zlib

from board_diagnostics import DiagnosticLog
from node_compare import ReferenceArchive, compare_tensor, digest, replay_convolution
from pcie_backend import (BASE, MAILBOX, MAGIC, INFO, STATUS, checked, PBTools,
                          PCIeBackend, READY, BUSY, DONE, ERROR, NODE, OUTPUT_BYTES)
from pcie_prepared import checked_file, detections, annotated_svg, reference_result

ROOT=Path(__file__).resolve().parent.parent
HEADER=struct.Struct('<4I2Q8I')
TENSOR=struct.Struct('<iIIIQ10I')
CSV_KEYS=['node','kind','tensor','bytes','byte_identical','different_elements','max_absolute_difference',
          'mean_absolute_difference','rmse','max_ulp_difference','same_input_reference_crc','node_cycles','amu_flags']


def write_csv(path, rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=CSV_KEYS,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def parse_snapshot(blob, state, backend):
    if len(blob)!=512 or zlib.crc32(blob[:508])!=struct.unpack_from('<I',blob,508)[0]:
        raise ValueError('Node snapshot CRC/length mismatch')
    h=HEADER.unpack_from(blob)
    keys='magic version bytes node session frame sequence op inputs outputs token options reserved reserved2'.split()
    header=dict(zip(keys,h))
    if (h[:3]!=(MAGIC,1,512) or h[4:7]!=(backend.session,backend.frame,backend.sequence) or
            h[7]>=21 or h[8]>4 or h[9]>2 or h[10]!=h[3]+1 or h[11]!=backend.options or
            state['detail']!=h[3] or state['output_bytes']!=512 or
            state['output_crc']!=zlib.crc32(blob)):
        raise ValueError('Node snapshot identity/layout mismatch')
    descriptors=[]
    for slot in range(6):
        raw=TENSOR.unpack_from(blob,64+slot*64)
        if raw[0]<0:
            if raw[0]!=-1:
                raise ValueError('Invalid empty tensor slot')
            continue
        tid,dtype,rank,size,address,constant,count=raw[:7]
        shape=list(raw[7:7+rank])
        if (not 1<=rank<=4 or dtype not in (1,2) or constant not in (0,1) or
                any(d<=0 for d in shape) or math.prod(shape)!=count or
                size!=count*(4 if dtype==1 else 8) or
                (slot<4 and slot>=header['inputs']) or
                (slot>=4 and slot-4>=header['outputs'])):
            raise ValueError('Invalid node tensor descriptor')
        backend.link._range(address,size)
        descriptors.append({'slot':slot,'direction':'input' if slot<4 else 'output',
                            'id':tid,'dtype':dtype,'shape':shape,'bytes':size,'address':address,
                            'constant':constant,'count':count,'crc_before':raw[11],'crc_after':raw[12]})
    tail=struct.unpack_from('<15I',blob,448)
    header.update(tensors=descriptors,amu_flags=tail[0]|tail[1]<<32,
                  node_cycles=tail[2]|tail[3]<<32)
    return header


def wait_capture(backend, deadline):
    while True:
        state=backend._status(deadline)
        if state['state']==ERROR:
            backend._snapshot(deadline)
            raise RuntimeError(f'Board capture error: {state}; diagnostic={backend.last_diagnostic}')
        if state['sequence']==backend.sequence:
            if (state['session'],state['frame'])!=(backend.session,backend.frame):
                raise ValueError('Capture session/frame mismatch')
            if state['state'] in (NODE,DONE):
                return state
        backend._pause(deadline)


def read_node(backend, state, deadline):
    blob=backend.link.read(backend.capture_address,512,deadline)
    node=parse_snapshot(blob,state,backend)
    if backend._status(deadline)!=state:
        raise ValueError('Board resumed/reset during node snapshot read')
    return node


def read_tensor(backend, descriptor, deadline, crc_key='crc_after'):
    data=backend.link.read(descriptor['address'],descriptor['bytes'],deadline)
    if zlib.crc32(data)!=descriptor[crc_key]:
        raise ValueError(f"Tensor {descriptor['id']} CRC mismatch; board data was not stable")
    return data


def recover_paused(pbcopy, pbload, tool_timeout, diagnostic):
    """Abort a stranded capture only after acquiring the ordinary host-owner lock."""
    link=PBTools(pbcopy,pbload,timeout=tool_timeout,diagnostic=diagnostic)
    try:
        deadline=time.monotonic()+max(60,tool_timeout*6)
        values=INFO.unpack(checked(link.read(MAILBOX,64,deadline)))
        if values[:3]!=(MAGIC,1,256):
            raise ValueError('Invalid firmware info during capture recovery')
        backend=PCIeBackend.__new__(PCIeBackend)
        backend.link=link; backend.stop=link.stop; backend.poll_interval=.1
        backend.last_status=None; backend.diagnostic=diagnostic
        while True:
            state=backend._status(deadline)
            doorbell=struct.unpack_from('<I',link.read(MAILBOX+128,64,deadline))[0]
            if state['state'] in (READY,DONE,ERROR):
                return False
            if state['state']==NODE and doorbell==state['sequence']:
                break
            backend._pause(deadline)
        backend.session=state['session']; backend.frame=state['frame']; backend.sequence=state['sequence']
        backend.capture_address=values[11]
        if not values[3]&1 or not BASE<=backend.capture_address<=MAILBOX-512 or backend.capture_address%64:
            raise ValueError('Invalid recovery snapshot address')
        blob=link.read(backend.capture_address,512,deadline)
        backend.options=HEADER.unpack_from(blob)[11]
        node=parse_snapshot(blob,state,backend)
        if backend._status(deadline)!=state:
            raise ValueError('Board changed during capture recovery')
        backend._submit(4,0,node['token'],deadline)
        while True:
            state=backend._status(deadline)
            if state['sequence']==backend.sequence and state['state']==ERROR:
                if state['error']!=0x44:
                    raise RuntimeError(f'Capture recovery failed: {state}')
                return True
            backend._pause(deadline)
    finally:
        link.close()


def verify_reference_manifest(manifest):
    if manifest.get('version')!=1 or len(manifest['nodes'])!=334 or len(manifest['tensors'])!=549:
        raise ValueError('Invalid frozen node reference manifest')
    frozen=json.loads((ROOT/'model/manifest.json').read_text(encoding='utf-8-sig'))
    if manifest['model_files_sha256']!={r['name']:r['sha256'] for r in frozen['files']}:
        raise ValueError('Node reference graph/constants do not match this package')
    for record in frozen['files']:
        data=(ROOT/'model'/record['name']).read_bytes()
        if len(data)!=record['bytes'] or digest(data)!=record['sha256']:
            raise ValueError('Frozen deployment asset hash mismatch')


def collect(backend, archive, manifest, sample, prepared, output, timeout, log):
    descriptors=manifest['tensors']; nodes=manifest['nodes']
    constants=(ROOT/'model/constants.bin').read_bytes()
    expected_live={0:archive.take(0)}
    if expected_live[0]!=prepared:
        raise ValueError('Reference input differs from the transmitted prepared input')
    actual_paths={0:ROOT/sample['input']}
    last_use={}
    for step,node in enumerate(nodes):
        for tid in node['inputs']:
            last_use[tid]=step
    tensor_dir=output/'tensors'; tensor_dir.mkdir()
    backend.frame+=1
    deadline=time.monotonic()+timeout
    previous=backend._status(deadline)
    if previous['state'] not in (READY,DONE) or previous['session']!=backend.session:
        raise ValueError('Board is not ready for capture input')
    backend.link.write(backend.input_address,prepared,deadline)
    backend._submit(2,len(prepared),zlib.crc32(prepared),deadline)
    rows=[]; first=None; input_checks=[]
    try:
        for step,node in enumerate(nodes):
            deadline=time.monotonic()+timeout
            state=wait_capture(backend,deadline)
            if state['state']!=NODE:
                raise ValueError(f'Capture finished early; expected node {step}')
            snapshot=read_node(backend,state,deadline)
            if snapshot['node']!=step or snapshot['op']!=node['op']:
                raise ValueError('Missing, repeated or unexpected capture node')
            observed_inputs=[d['id'] for d in snapshot['tensors'] if d['direction']=='input']
            if observed_inputs!=node['inputs']:
                raise ValueError('Missing, repeated or unexpected node inputs')
            actual_inputs={}; checks=[]
            for d in snapshot['tensors']:
                tid=d['id']
                if not 0<=tid<len(descriptors):
                    raise ValueError('Node tensor ID outside frozen graph')
                t=descriptors[tid]
                if (d['dtype'],d['shape'],d['bytes'],d['constant'])!=(t['dtype'],t['shape'],t['bytes'],t['constant']):
                    raise ValueError('Board tensor metadata differs from frozen graph')
                if d['direction']!='input':
                    continue
                if tid not in node['inputs']:
                    raise ValueError('Unexpected node input ID')
                expected=constants[t['offset']:t['offset']+t['bytes']] if t['constant'] else expected_live[tid]
                checks.append({'tensor':tid,'reference_crc_before_matches':zlib.crc32(expected)==d['crc_before'],
                               'storage_unchanged_after_operation':d['crc_before']==d['crc_after']})
            observed_ids=[d['id'] for d in snapshot['tensors'] if d['direction']=='output']
            if observed_ids!=[tid for tid in node['outputs'] if tid>=0]:
                raise ValueError('Missing or extra node outputs')
            for d in snapshot['tensors']:
                if d['direction']!='output':
                    continue
                tid=d['id']
                expected=archive.take(tid)
                actual=read_tensor(backend,d,deadline)
                path=tensor_dir/f'{tid:04d}.bin'
                path.write_bytes(actual); actual_paths[tid]=path
                metrics=compare_tensor(actual,expected,d['dtype'],d['shape'])
                row={'node':step,'kind':node['kind'],'tensor':tid,'bytes':d['bytes'],
                     'same_input_reference_crc':all(c['reference_crc_before_matches'] for c in checks),
                     'node_cycles':snapshot['node_cycles'],'amu_flags':snapshot['amu_flags'],**metrics}
                rows.append(row)
                if not metrics['byte_identical'] and first is None:
                    # Fetch actual inputs while this node is frozen. A reused
                    # input is reconstructed from its earlier captured output.
                    for item in snapshot['tensors']:
                        if item['direction']!='input':
                            continue
                        source_id=item['id']; t=descriptors[source_id]
                        if item['crc_after']==item['crc_before']:
                            data=read_tensor(backend,item,deadline,'crc_before')
                            source='physical DDR at paused node'
                        elif not t['constant'] and source_id in actual_paths:
                            data=actual_paths[source_id].read_bytes()
                            source='previous full tensor capture; input storage was reused'
                            if zlib.crc32(data)!=item['crc_before']:
                                continue
                        else:
                            continue
                        actual_inputs[source_id]=data
                        ref=constants[t['offset']:t['offset']+t['bytes']] if t['constant'] else expected_live[source_id]
                        input_checks.append({'tensor':source_id,'source':source,
                                             'byte_identical_to_reference':data==ref,
                                             'sha256_actual':digest(data),'sha256_reference':digest(ref)})
                        (output/f'first-input-{source_id:04d}.bin').write_bytes(data)
                    first={'node':step,'kind':node['kind'],'module':node['module'],'tensor':tid,
                           'shape':d['shape'],'metrics':metrics,'inputs':input_checks,
                           'input_descriptors':[{key:item[key] for key in ('id','dtype','shape')}
                                                for item in snapshot['tensors'] if item['direction']=='input'],
                           'same_input_full_bytes_verified':len(input_checks)==len(node['inputs']) and
                                all(c['byte_identical_to_reference'] for c in input_checks),
                           'replay':replay_convolution(node,descriptors,actual_inputs,actual,expected,metrics['samples'])}
                    (output/'first-reference.bin').write_bytes(expected)
                expected_live[tid]=expected
                print(f"node {step:03d} {node['kind']} tensor={tid} exact={metrics['byte_identical']} "
                      f"different={metrics['different_elements']}/{metrics['count']} "
                      f"max_abs={metrics['max_absolute_difference']:.8g}",flush=True)
            if backend._status(deadline)!=state:
                raise ValueError('Board node changed before host acknowledgement')
            backend._submit(3,0,snapshot['token'],deadline)
            for tid in list(expected_live):
                if last_use.get(tid,-1)<=step:
                    del expected_live[tid]
        state=wait_capture(backend,time.monotonic()+timeout)
        if state['state']!=DONE or state['output_bytes']!=OUTPUT_BYTES:
            raise ValueError('No final DONE after all captured nodes')
        result=backend.link.read(backend.output_address,OUTPUT_BYTES,time.monotonic()+timeout)
        if zlib.crc32(result)!=state['output_crc'] or result!=actual_paths[548].read_bytes():
            raise ValueError('Final result differs from the captured last tensor')
        for row in struct.iter_unpack('<6f',result):
            if not all(map(math.isfinite,row)) or row[5]!=int(row[5]) or not 0<=row[5]<80:
                raise ValueError('Invalid final candidate values/classes')
        archive.finish()
        backend._snapshot(time.monotonic()+timeout)
        return rows,first,result
    except BaseException:
        # CTRL-C and host failures release a paused model through authenticated
        # ABORT. A halted core/tool failure may prevent this; report it explicitly.
        try:
            deadline=time.monotonic()+max(30,backend.link.timeout*6)
            while True:
                state=backend._status(deadline)
                doorbell=struct.unpack_from('<I',backend.link.read(MAILBOX+128,64,deadline))[0]
                if state['state'] in (NODE,DONE,ERROR) and state['sequence']==doorbell:
                    break
                backend._pause(deadline)
            if state['state']==NODE:
                backend.sequence=state['sequence']
                snapshot=read_node(backend,state,deadline)
                backend._submit(4,0,snapshot['token'],deadline)
                while True:
                    aborted=backend._status(deadline)
                    if aborted['sequence']==backend.sequence and aborted['state']==ERROR:
                        break
                    backend._pause(deadline)
        except Exception as exc:
            log({'type':'capture_abort_error','message':str(exc)})
        (output/'partial-nodes.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
        write_csv(output/'nodes.csv',rows)
        if first:
            (output/'first_difference.json').write_text(json.dumps(first,indent=2)+'\n',encoding='utf-8')
        raise


def make_return_bundle(output):
    bundle=output/'return.zip'
    names=('summary.json','nodes.csv','first_difference.json','annotated.svg')
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path=output/name
            if path.exists():
                archive.write(path,arcname=name)
    if bundle.stat().st_size>1024*1024:
        raise ValueError('Compact return bundle exceeded 1 MiB; full tensor files were not included')
    return bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample',choices=('bus','zidane'),default='bus')
    parser.add_argument('--pbcopy',default='pbcopy')
    parser.add_argument('--pbload',default='pbload')
    parser.add_argument('--tool-timeout',type=float,default=30)
    parser.add_argument('--node-timeout',type=float,default=300)
    parser.add_argument('--board-trace',action='store_true')
    parser.add_argument('--recover-paused',action='store_true')
    parser.add_argument('--skip-normal-check',action='store_true',help='Skip the final ordinary-mode repeat')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if sys.platform!='linux':
        parser.error('Run on the Linux host connected to S2C PCIe')
    if min(args.tool_timeout,args.node_timeout)<=0:
        parser.error('Timeouts must be positive')
    output=args.output or ROOT/'yolo26_pc/outputs/node-capture'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True,exist_ok=False)
    diagnostics=DiagnosticLog(output)
    def log(event):
        # Keep successful pbload polling in the local log instead of the console.
        if event.get('type')=='pcie_tool' and not event.get('returncode'):
            with diagnostics.lock:
                diagnostics.stream.write(json.dumps(event)+'\n'); diagnostics.stream.flush()
        else:
            diagnostics.emit(event)
    summary={'status':'starting','sample':args.sample,'numerical_cause':'not established',
             'full_tensors_location':'tensors/ on host; excluded from return.zip'}
    backend=archive=None
    started=time.monotonic(); exit_code=1
    try:
        manifest=json.loads((ROOT/'reference/nodes/manifest.json').read_text(encoding='utf-8'))
        verify_reference_manifest(manifest)
        samples=json.loads((ROOT/'reference/prepared/manifest.json').read_text(encoding='utf-8'))['samples']
        sample=samples[args.sample]
        prepared=checked_file(sample['input'],sample['input_sha256'])
        archive=ReferenceArchive(ROOT,manifest,args.sample)
        if args.recover_paused:
            summary['recovered_paused_capture']=recover_paused(args.pbcopy,args.pbload,args.tool_timeout,log)
        backend=PCIeBackend(args.pbcopy,args.pbload,infer_timeout=args.node_timeout,
                            tool_timeout=args.tool_timeout,diagnostic=log,check_finite=True,
                            trace=args.board_trace,capture=True)
        summary.update(board=backend.info,input_sha256=digest(prepared))
        rows,first,raw=collect(backend,archive,manifest,sample,prepared,output,args.node_timeout,log)
        write_csv(output/'nodes.csv',rows)
        (output/'nodes.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
        if first:
            (output/'first_difference.json').write_text(json.dumps(first,indent=2)+'\n',encoding='utf-8')
        found=detections(raw,sample['transform'])
        source=checked_file(sample['source'],sample['source_sha256'])
        (output/'annotated.svg').write_text(annotated_svg(source,sample['transform'],found),encoding='utf-8')
        reference=checked_file(sample['reference'],sample['reference_sha256'],7232)
        summary.update(status='completed',nodes_captured=len({r['node'] for r in rows}),
                       tensors_captured=len(rows),different_nodes=len({r['node'] for r in rows if not r['byte_identical']}),
                       first_difference=None if first is None else {k:first[k] for k in ('node','kind','tensor','shape','same_input_full_bytes_verified')},
                       final_reference=reference_result(raw,reference),detections=len(found),
                       last_diagnostic=backend.last_diagnostic)
        if not args.skip_normal_check:
            backend.close(); backend=None
            backend=PCIeBackend(args.pbcopy,args.pbload,infer_timeout=args.node_timeout,
                                tool_timeout=args.tool_timeout,diagnostic=log,check_finite=True,
                                trace=args.board_trace)
            normal=backend.infer_bytes(prepared)
            (output/'normal-candidates.bin').write_bytes(normal)
            summary['normal_repeat']={'byte_identical_to_capture':normal==raw,
                                      'reference':reference_result(normal,reference),'timing':backend.last_timing}
            print(f'Ordinary-mode repeat exact-to-capture={normal==raw}',flush=True)
        exit_code=0
    except BaseException as exc:
        summary.update(status='failed',error=str(exc),exception=type(exc).__name__)
        if backend is not None:
            summary.update(last_status=backend.last_status,last_diagnostic=backend.last_diagnostic)
        diagnostics.exception(exc)
    finally:
        for resource in (backend,archive):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    summary.setdefault('cleanup_errors',[]).append(str(exc))
        summary['elapsed_seconds']=time.monotonic()-started
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        diagnostics.close()
        try:
            bundle=make_return_bundle(output)
            print(f'Optional compact archive: {bundle} ({bundle.stat().st_size} bytes)',flush=True)
        except (OSError,ValueError) as exc:
            print(f'Optional archive unavailable: {exc}',flush=True)
        print(f'Full data remain on host: {output}',flush=True)
        from node_report import remember_report,render_page
        try:
            remember_report(output)
            print('Screenshot pages: python3 yolo26_pc/node_report.py --page 1 (or 2, 3)',flush=True)
            print(render_page(output,1),end='',flush=True)
        except (OSError,ValueError,KeyError,IndexError,TypeError) as exc:
            print(f'Screenshot report error: {exc}; saved data remain at {output}',flush=True)
    return exit_code


if __name__=='__main__':
    raise SystemExit(main())
