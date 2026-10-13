"""Lossless node references, full-tensor metrics and limited numerical replays.

Uses Python 3.8+ stdlib only. A matching replay is evidence about that candidate
rule, not proof of the FPGA's unspecified accumulation implementation.
"""
from __future__ import annotations
import array
import gzip
import hashlib
import io
import math
from pathlib import Path
import struct
import sys
import zlib


def digest(data):
    return hashlib.sha256(data).hexdigest()


class PartsReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths=iter(paths)
        self.current=None

    def readable(self):
        return True

    def readinto(self, buffer):
        view=memoryview(buffer)
        used=0
        while used<len(view):
            if self.current is None:
                path=next(self.paths,None)
                if path is None:
                    break
                self.current=path.open('rb')
            size=self.current.readinto(view[used:])
            if size:
                used+=size
            else:
                self.current.close()
                self.current=None
        return used

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


class ReferenceArchive:
    def __init__(self, root, manifest, sample):
        self.root=Path(root).resolve()
        self.tensors=manifest['tensors']
        record=manifest['samples'][sample]
        self.records=iter(record['records'])
        paths=[]
        for entry in record['parts']:
            path=(self.root/entry['path']).resolve()
            # relative_to preserves containment checks on Python 3.8 as well.
            try:
                path.relative_to(self.root)
            except ValueError:
                raise ValueError('Reference path escapes package') from None
            if path.stat().st_size!=entry['bytes']:
                raise ValueError(f'Reference part length mismatch: {path.name}')
            hasher=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    hasher.update(block)
            if hasher.hexdigest()!=entry['sha256']:
                raise ValueError(f'Reference part hash mismatch: {path.name}')
            paths.append(path)
        self.parts=io.BufferedReader(PartsReader(paths))
        self.stream=gzip.GzipFile(fileobj=self.parts,mode='rb')
        header=self.stream.read(16)
        if len(header)!=16 or struct.unpack('<8sII',header)!=(b'Y26SCAL1',1,len(record['records'])):
            self.close()
            raise ValueError('Invalid node reference header')

    def take(self, expected_id):
        record=next(self.records,None)
        if record is None or record['id']!=expected_id:
            raise ValueError('Node reference order mismatch')
        header=self.stream.read(12)
        if len(header)!=12:
            raise ValueError('Truncated tensor header')
        tid,dtype,size=struct.unpack('<3I',header)
        t=self.tensors[expected_id]
        if (tid,dtype,size)!=(expected_id,t['dtype'],t['bytes']):
            raise ValueError('Tensor reference descriptor mismatch')
        data=self.stream.read(size)
        if len(data)!=size or digest(data)!=record['sha256']:
            raise ValueError('Tensor reference data mismatch')
        return data

    def finish(self):
        if next(self.records,None) is not None or self.stream.read(1):
            raise ValueError('Unexpected node reference tail')

    def close(self):
        if hasattr(self,'stream'):
            self.stream.close()
        self.parts.close()


def indices(index, shape):
    result=[]
    for dimension in reversed(shape):
        result.append(index%dimension)
        index//=dimension
    return list(reversed(result))


def ordered(bits):
    return (~bits&0xffffffff) if bits&0x80000000 else (bits|0x80000000)


def compare_tensor(actual, expected, dtype, shape, sample_limit=8):
    width=4 if dtype==1 else 8
    if dtype not in (1,2) or len(actual)!=len(expected) or len(actual)%width:
        raise ValueError('Tensor comparison type/length mismatch')
    result={'byte_identical':actual==expected,'count':len(actual)//width,
            'different_elements':0,'max_absolute_difference':0,'mean_absolute_difference':0.0,
            'rmse':0.0,'max_ulp_difference':0,'nonfinite_actual':0,'nonfinite_reference':0,
            'samples':[],'sha256_actual':digest(actual),'sha256_reference':digest(expected)}
    code='I' if dtype==1 else 'q'
    a=array.array(code); a.frombytes(actual)
    b=array.array(code); b.frombytes(expected)
    if sys.byteorder!='little':
        a.byteswap(); b.byteswap()
    if dtype==1:
        result['nonfinite_actual']=sum((word&0x7f800000)==0x7f800000 for word in a)
        result['nonfinite_reference']=sum((word&0x7f800000)==0x7f800000 for word in b)
    if actual==expected:
        return result
    total=squares=0.0
    worst=None
    for index,(left,right) in enumerate(zip(a,b)):
        if left==right:
            continue
        result['different_elements']+=1
        if dtype==1:
            av=struct.unpack('<f',struct.pack('<I',left))[0]
            bv=struct.unpack('<f',struct.pack('<I',right))[0]
            finite=math.isfinite(av) and math.isfinite(bv)
            ulp=abs(ordered(left)-ordered(right)) if finite else None
            difference=abs(av-bv) if finite else None
            sample={'element':index,'index':indices(index,shape),'actual':av if math.isfinite(av) else str(av),
                    'reference':bv if math.isfinite(bv) else str(bv),
                    'actual_bits':f'{left:08x}','reference_bits':f'{right:08x}','ulp':ulp}
        else:
            av,bv=left,right
            difference=abs(av-bv); ulp=None
            sample={'element':index,'index':indices(index,shape),'actual':av,'reference':bv}
        if len(result['samples'])<sample_limit:
            result['samples'].append(sample)
        if difference is not None:
            total+=difference; squares+=difference*difference
            if worst is None or difference>result['max_absolute_difference']:
                worst=sample
                result['max_absolute_difference']=difference
            if ulp is not None:
                result['max_ulp_difference']=max(result['max_ulp_difference'],ulp)
    if result['count']:
        result['mean_absolute_difference']=total/result['count']
        result['rmse']=math.sqrt(squares/result['count'])
    if worst is not None and worst not in result['samples']:
        result['samples'].append(worst)
    return result


def f32(value):
    return struct.unpack('<f',struct.pack('<f',value))[0]


def half(value):
    raw=struct.pack('<e',value)
    return struct.unpack('<e',raw)[0],raw.hex()


def replay_convolution(node, tensors, inputs, actual, reference, samples):
    """Recompute selected outputs with exact board input bytes, not golden inputs."""
    if node['kind']!='CONV' or node['params'][4]!=1:
        return {'status':'not_replayed','reason':'This stdlib replay covers AMU ordinary convolution.'}
    ids=node['inputs']
    if len(ids)!=3 or any(tid not in inputs for tid in ids):
        return {'status':'not_replayed','reason':'An input was overwritten and could not be reconstructed.'}
    x,w,bias=(inputs[tid] for tid in ids)
    shape=tensors[ids[0]]['shape']
    weights=tensors[ids[1]]['shape']
    output=tensors[node['outputs'][0]]['shape']
    sh,sw,ph,pw=node['params'][:4]
    _,ic,ih,iw=shape; oc,_,kh,kw=weights
    def at(data,index):
        return struct.unpack_from('<f',data,index*4)[0]
    rows=[]
    for sample in samples:
        index=sample['element']
        n,o,y,c=indices(index,output)
        acc=seq=tree_acc=partial=0.0
        block=[]; previews=[]; block_rounds=[]
        total=ic*kh*kw
        k=0
        for channel in range(ic):
            for ky in range(kh):
                for kx in range(kw):
                    iy=y*sh+ky-ph; ix=c*sw+kx-pw
                    value=at(x,((n*ic+channel)*ih+iy)*iw+ix) if 0<=iy<ih and 0<=ix<iw else 0.0
                    weight=at(w,((o*ic+channel)*kh+ky)*kw+kx)
                    xv,xb=half(value); wv,wb=half(weight)
                    product=xv*wv
                    partial+=product
                    seq=f32(seq+product)
                    block.append(product)
                    if len(previews)<32:
                        previews.append({'k':k,'input_fp32':value,'weight_fp32':weight,
                                         'input_fp16_le':xb,'weight_fp16_le':wb})
                    k+=1
                    if len(block)==32 or k==total:
                        acc=f32(acc+partial)
                        tree=block+[0.0]*(32-len(block))
                        while len(tree)>1:
                            tree=[f32(tree[i]+tree[i+1]) for i in range(0,len(tree),2)]
                        tree_acc=f32(tree_acc+tree[0])
                        if len(block_rounds)<16:
                            block_rounds.append({'end_k':k,'qemu_accumulator':acc,
                                                 'sequential_fp32_accumulator':seq,'tree_fp32_accumulator':tree_acc})
                        partial=0.0; block=[]
        offset=at(bias,o)
        values={'qemu_fp64_dot_fp32_per_k32':f32(acc+offset),
                'sequential_fp32':f32(seq+offset),'pairwise_fp32_k32':f32(tree_acc+offset)}
        av=actual[index*4:index*4+4]; rv=reference[index*4:index*4+4]
        rows.append({'element':index,'index':[n,o,y,c],'actual_bits_le':av.hex(),
                     'reference_bits_le':rv.hex(),'rules':{name:{'value':value,
                       'matches_actual_bits':struct.pack('<f',value)==av,
                       'matches_reference_bits':struct.pack('<f',value)==rv} for name,value in values.items()},
                     'bias':offset,'k':total,'first_32_operands':previews if not rows else [],
                     'first_16_blocks':block_rounds})
    return {'status':'replayed_selected_elements','results':rows,
            'note':'Candidate-rule matches do not establish which rule the FPGA implements. Hardware semantics still require confirmation.'}
