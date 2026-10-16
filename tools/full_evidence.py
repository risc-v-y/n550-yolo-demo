"""Offline full-node evidence validation and replay. No physical board access."""
from collections import Counter
import json
import math
from pathlib import Path
import struct
import sys
import zlib
from full_replay import Replay,ROOT,sha,source_sha,NUMERIC_RULE

sys.path.insert(0,str(ROOT/'yolo26_pc'))
from node_capture import HEADER,parse_snapshot
from node_compare import compare_tensor
from pcie_backend import PBTools,OUTPUT_BYTES
from pcie_prepared import checked_file,detections,annotated_svg,save_annotation,reference_result


def read_json(path): return json.loads(path.read_text(encoding='utf-8'))


def saved(directory,name,digest=None,size=None):
    path=(directory/name).resolve()
    path.relative_to(directory.resolve())
    data=path.read_bytes()
    if size is not None and len(data)!=size: raise ValueError('Wrong length: '+name)
    if digest is not None and sha(data)!=digest: raise ValueError('SHA256 mismatch: '+name)
    return data


def identity(directory,sample,manifest,prepared):
    summary=read_json(directory/'summary.json')
    if (summary.get('evidence_version')!=1 or summary.get('sample')!=sample or
            summary.get('input_sha256')!=prepared['input_sha256'] or
            summary.get('firmware_sha256')!=sha((ROOT/'firmware/demo-pcie.bin').read_bytes()) or
            summary.get('reference_manifest_sha256') not in
                (source_sha(ROOT/'reference/nodes/manifest.json'),sha((ROOT/'reference/nodes/manifest.json').read_bytes()))):
        raise ValueError('Capture lacks matching graph/sample/input evidence')
    saved(directory,'input.bin',prepared['input_sha256'],manifest['tensors'][0]['bytes'])
    return summary


def evidence(directory,node,tensors):
    prefix='snapshots/{:04d}'.format(node['id'])
    record=read_json(directory/(prefix+'.json'))
    if record.get('version')!=1: raise ValueError('Unsupported evidence version')
    raw=saved(directory,prefix+'.bin',record['snapshot_sha256'],512)
    h=HEADER.unpack_from(raw)
    backend=type('Snapshot',(),{'session':h[4],'frame':h[5],'sequence':h[6],'options':12,
                'link':type('Link',(),{'_range':staticmethod(PBTools._range)})()})()
    state={'detail':node['id'],'output_bytes':512,'output_crc':zlib.crc32(raw)}
    snap=parse_snapshot(raw,state,backend)
    if (snap!=record['snapshot'] or snap['node']!=node['id'] or snap['op']!=node['op'] or
            snap['inputs']!=len(node['inputs']) or snap['outputs']!=len(node['outputs'])):
        raise ValueError('Snapshot does not match the frozen node/evidence')
    for direction,ids in (('input',node['inputs']),('output',[i for i in node['outputs'] if i>=0])):
        ds=[d for d in snap['tensors'] if d['direction']==direction]
        if [d['id'] for d in ds]!=ids: raise ValueError('Node tensor list mismatch')
        for d in ds:
            t=tensors[d['id']]
            if any(d[k]!=t[k] for k in ('dtype','shape','bytes','constant','count')):
                raise ValueError('Node tensor shape/type mismatch')
    if [p['id'] for p in record['inputs']]!=node['inputs']:
        raise ValueError('Input evidence order mismatch')
    if [p['id'] for p in record['outputs']]!=[i for i in node['outputs'] if i>=0]:
        raise ValueError('Output evidence order mismatch')
    inputs={}; issues=[]; outputs={}
    ds={d['id']:d for d in snap['tensors'] if d['direction']=='input'}
    for proof in record['inputs']:
        tid=proof['id']; d=ds[tid]
        try:
            if not proof.get('verified') or proof['crc32_before']!=d['crc_before']:
                raise ValueError('Input pre-operation CRC is not verified')
            data=saved(directory,proof['path'],proof['sha256'],d['bytes'])
            if zlib.crc32(data)!=d['crc_before']: raise ValueError('Input bytes differ from pre-operation CRC')
            inputs[tid]=data
        except (OSError,ValueError,KeyError) as exc:
            issues.append({'tensor':tid,'error':str(exc)})
    ds={d['id']:d for d in snap['tensors'] if d['direction']=='output'}
    for proof in record['outputs']:
        tid=proof['id']; d=ds[tid]
        data=saved(directory,'tensors/{:04d}.bin'.format(tid),proof['sha256'],d['bytes'])
        if zlib.crc32(data)!=d['crc_after']: raise ValueError('Output bytes differ from snapshot CRC')
        outputs[tid]=data
    return snap,inputs,outputs,issues


def final_check(directory,prepared,summary,render=True):
    raw=saved(directory,'tensors/0548.bin',size=OUTPUT_BYTES)
    normal_record=summary.get('normal_repeat',{})
    normal=saved(directory,'normal-candidates.bin',normal_record['sha256'],OUTPUT_BYTES)
    if normal_record.get('byte_identical_to_capture')!=(normal==raw):
        raise ValueError('Normal repeat metadata disagrees with bytes')
    for index,c in enumerate(struct.iter_unpack('<6f',raw)):
        if (not all(map(math.isfinite,c)) or c[5]!=int(c[5]) or not 0<=c[5]<80 or
                not 0<=c[4]<=1 or c[0]>c[2] or c[1]>c[3]):
            raise ValueError('Invalid candidate at index '+str(index))
    found=detections(raw,prepared['transform'])
    # Render from the verified original image again. Coordinates may exceed
    # 416 before inverse letterbox mapping and clipping; that is legitimate.
    source=checked_file(prepared['source'],prepared['source_sha256'])
    svg=annotated_svg(source,prepared['transform'],found)
    if not svg or len(found)!=svg.count('fill="none" stroke='): raise ValueError('Drawing/box count mismatch')
    if render: save_annotation(directory/'full-check-annotated.png',source,prepared['transform'],found)
    reference=checked_file(prepared['reference'],prepared['reference_sha256'],7232)
    return {'valid':True,'normal_repeat_exact':normal==raw,'normal_sha256':sha(normal),
            'capture_sha256':sha(raw),'detections':found,'annotated':'full-check-annotated.png',
            'qemu':reference_result(raw,reference)}


def reusable(directory,sample,manifest,prepared):
    """Reject legacy/partial/corrupt data instead of claiming complete coverage."""
    try:
        summary=identity(directory,sample,manifest,prepared)
        if summary.get('status')!='completed': return False
        session=None
        for node in manifest['nodes']:
            snap,_,_,issues=evidence(directory,node,manifest['tensors'])
            key=(snap['session'],snap['frame'],snap['sequence']-node['id'])
            if issues or session is not None and key!=session: return False
            session=key
        return final_check(directory,prepared,summary,render=False)['normal_repeat_exact']
    except (OSError,ValueError,KeyError,TypeError,struct.error): return False


def audit(directory,sample,manifest,prepared,replay_factory=Replay,progress=None):
    directory=Path(directory); rows=[]; replay=None; session=None
    result={'numeric_rule':NUMERIC_RULE,'sample':sample,'directory':str(directory),'rows':rows,'final':{},'errors':[]}
    result['original_qemu_metrics']='nodes.json' if (directory/'nodes.json').is_file() else 'partial-nodes.json'
    try:
        summary=identity(directory,sample,manifest,prepared)
    except (OSError,ValueError,KeyError) as exc:
        summary={}; result['errors'].append(str(exc))
    qemu={r['id']:r['sha256'] for r in manifest['samples'][sample]['records']}
    for node in manifest['nodes']:
        ids=[tid for tid in node['outputs'] if tid>=0]
        row={'node':node['id'],'kind':node['kind'],'outputs':[],'status':'missing'}
        rows.append(row)
        try:
            if not summary: raise ValueError('Capture identity not verified')
            snap,inputs,actual,issues=evidence(directory,node,manifest['tensors'])
            key=(snap['session'],snap['frame'],snap['sequence']-node['id'])
            if session is not None and key!=session: raise ValueError('Mixed capture sessions/frames/sequences')
            session=key
            row['inputs_verified']=not issues
            row['captured_outputs']=len(actual)
            row['outputs']=[{'tensor':tid,'actual_sha256':sha(actual[tid]),
                             'qemu_equal':sha(actual[tid])==qemu[tid]} for tid in ids]
            if issues:
                row.update(status='input_unverified',input_issues=issues)
            else:
                if replay is None: replay=replay_factory(16)
                try:
                    expected=replay.run(node,manifest['tensors'],inputs)
                except Exception as exc:
                    replay.close(); replay=None
                    row.update(status='unexplained',error='Replay failed: '+str(exc))
                    continue
                for detail in row['outputs']:
                    tid=detail['tensor']
                    data=actual[tid]; predicted=expected[tid]; t=manifest['tensors'][tid]
                    detail.update(equal=data==predicted,predicted_sha256=sha(predicted))
                    if not detail['equal']:
                        detail['difference']=compare_tensor(data,predicted,t['dtype'],t['shape'])
                match=all(d['equal'] for d in row['outputs'])
                amu=node['kind']=='CONV' and node['params'][4]==1
                row['status']=('rtl_match' if amu and any(not d['qemu_equal'] for d in row['outputs'])
                               else 'exact') if match else 'unexplained'
                row['upstream_propagation']=match and not amu and any(not d['qemu_equal'] for d in row['outputs'])
            row['captured_outputs']=len(actual)
        except (OSError,ValueError,KeyError,TypeError,struct.error) as exc:
            row['error']=str(exc)
        finally:
            if progress: progress(sample,node['id'],row['status'])
    if replay is not None: replay.close()
    try: result['final']=final_check(directory,prepared,summary)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,struct.error) as exc: result['errors'].append(str(exc))
    result['counts']=dict(Counter(r['status'] for r in rows))
    result['nodes_captured']=sum('captured_outputs' in r for r in rows)
    result['outputs_captured']=sum(r.get('captured_outputs',0) for r in rows)
    result['inputs_verified_nodes']=sum(bool(r.get('inputs_verified')) for r in rows)
    result['qemu_different_outputs']=sum(not o['qemu_equal'] for r in rows for o in r['outputs'])
    result['passed']=(len(rows)==334 and result['nodes_captured']==334 and result['outputs_captured']==335
                      and result['inputs_verified_nodes']==334 and not result['errors']
                      and not any(result['counts'].get(k) for k in ('missing','unexplained','input_unverified'))
                      and result['final'].get('normal_repeat_exact',False))
    return result
