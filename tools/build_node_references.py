"""Package verified QEMU FP16 node data for stdlib-only comparison on the host.

The archive is lossless, split into <=24 MiB Git-friendly pieces. No PyTorch,
NumPy or retraining is needed on the lab host. Generated large full tensors are
local build inputs, not part of the compact result returned from that host.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import zlib

ROOT=Path(__file__).resolve().parents[1]
OPS='CONV SILU ADD SUB MUL COPY SLICE CONCAT POOL NEAREST PERMUTE EXPAND BMM SOFTMAX SIGMOID MAX TOPK GATHER REMAINDER FLOORDIV CAST'.split()


def file_hash(path):
    hasher=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            hasher.update(block)
    return hasher.hexdigest()


def parse_graph():
    text=(ROOT/'model/model_generated.c').read_text(encoding='utf-8')
    tensor_pattern=r'^\s*\{(\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+), \{([\d, ]+)\}\},$'
    tensors=[]
    for match in re.finditer(tensor_pattern,text,re.M):
        offset,ref_offset,size,count,rank,dtype,constant=map(int,match.groups()[:7])
        dimensions=list(map(int,match.group(8).split(',')))
        tensors.append({'id':len(tensors),'offset':offset,'bytes':size,'count':count,
                        'dtype':dtype,'constant':constant,'shape':dimensions[:rank]})
    node_pattern=r'^\s*\{OP_(\w+), (\d+), (\d+), \{([\d, \-]+)\}, \{([\d, \-]+)\}, \{([\d, \-]+)\}, "([^"]+)"\},$'
    nodes=[]
    for match in re.finditer(node_pattern,text,re.M):
        kind,inputs,outputs,ins,outs,params,module=match.groups()
        nodes.append({'id':len(nodes),'kind':kind,'op':OPS.index(kind),
                      'inputs':list(map(int,ins.split(',')))[:int(inputs)],
                      'outputs':list(map(int,outs.split(',')))[:int(outputs)],
                      'params':list(map(int,params.split(','))),'module':module})
    if len(tensors)!=549 or len(nodes)!=334:
        raise ValueError('Frozen graph parser coverage mismatch')
    return tensors,nodes


class PartWriter:
    def __init__(self, directory, sample, limit):
        self.directory=directory; self.sample=sample; self.limit=limit
        self.paths=[]; self.stream=None; self.used=0

    def write(self, data):
        view=memoryview(data); total=len(data)
        while view:
            if self.stream is None or self.used==self.limit:
                if self.stream is not None:
                    self.stream.close()
                path=self.directory/f'{self.sample}-tensors.gz.part{len(self.paths):02d}'
                self.paths.append(path); self.stream=path.open('wb'); self.used=0
            size=min(len(view),self.limit-self.used)
            self.stream.write(view[:size]); self.used+=size; view=view[size:]
        return total

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def close(self):
        if self.stream is not None:
            self.stream.close(); self.stream=None


def inspect_source(path, tensors, nodes, prepared, final_reference):
    expected=[0]+[tid for node in nodes for tid in node['outputs'] if tid>=0]
    records=[]
    with path.open('rb') as stream:
        header=stream.read(16)
        if len(header)!=16 or struct.unpack('<8sII',header)!=(b'Y26SCAL1',1,len(expected)):
            raise ValueError('Reference record count/header mismatch')
        for tid in expected:
            raw=stream.read(12)
            t=tensors[tid]
            if len(raw)!=12 or struct.unpack('<3I',raw)!=(tid,t['dtype'],t['bytes']):
                raise ValueError('QEMU dump does not follow the frozen graph')
            data=stream.read(t['bytes'])
            if len(data)!=t['bytes']:
                raise ValueError('Truncated QEMU tensor')
            if tid==0 and data!=prepared:
                raise ValueError('QEMU node input differs from prepared sample')
            if tid==548 and data!=final_reference:
                raise ValueError('QEMU final tensor differs from packaged FP16 reference')
            records.append({'id':tid,'sha256':hashlib.sha256(data).hexdigest(),'crc32':zlib.crc32(data)})
        if stream.read(1):
            raise ValueError('Unexpected QEMU dump tail')
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'yolo26_riscv/build/accel-model/amu-fp16-vlen512')
    parser.add_argument('--part-mib',type=int,default=24)
    args=parser.parse_args()
    if not 1<=args.part_mib<=24:
        parser.error('Parts must be between 1 and 24 MiB')
    frozen=json.loads((ROOT/'model/manifest.json').read_text(encoding='utf-8-sig'))
    for entry in frozen['files']:
        if file_hash(ROOT/'model'/entry['name'])!=entry['sha256']:
            raise ValueError('Frozen model asset hash mismatch')
    tensors,nodes=parse_graph()
    samples=json.loads((ROOT/'reference/prepared/manifest.json').read_text(encoding='utf-8'))['samples']
    directory=ROOT/'reference/nodes'; directory.mkdir(parents=True,exist_ok=True)
    manifest={'version':1,'format':'Y26SCAL1, lossless gzip parts','numeric_reference':
              'Observed QEMU FP16 operands, FP64 dot then FP32 rounding per K<=32 instruction; not a hardware PE definition.',
              'model_files_sha256':{r['name']:r['sha256'] for r in frozen['files']},
              'tensors':tensors,'nodes':nodes,'samples':{}}
    for name,sample in samples.items():
        source=args.source/name/'scalar-tensors.bin'
        prepared=(ROOT/sample['input']).read_bytes()
        final=(ROOT/sample['reference']).read_bytes()[32:]
        records=inspect_source(source,tensors,nodes,prepared,final)
        writer=PartWriter(directory,name,args.part_mib*1024*1024)
        try:
            with source.open('rb') as stream, gzip.GzipFile(filename='',mode='wb',compresslevel=6,
                                                           fileobj=writer,mtime=0) as compressed:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    compressed.write(block)
        finally:
            writer.close()
        parts=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in writer.paths]
        manifest['samples'][name]={'input_sha256':hashlib.sha256(prepared).hexdigest(),
                                   'raw_sha256':file_hash(source),'raw_bytes':source.stat().st_size,
                                   'records':records,'parts':parts}
        print(f'{name}: {len(records)} exact records; raw={source.stat().st_size}, '
              f'compressed={sum(p["bytes"] for p in parts)}, parts={len(parts)}',flush=True)
    path=directory/'manifest.json'
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':
    main()
