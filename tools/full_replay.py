"""Checked client for the packaged static Linux x86_64 numerical auditor."""
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess
import threading

ROOT=Path(__file__).resolve().parents[1]


def sha(data): return hashlib.sha256(data).hexdigest()


def source_sha(path):
    data=path.read_bytes()
    return sha(data if path.suffix=='.bin' else data.replace(b'\r\n',b'\n'))


def verified_binary(root):
    record=json.loads((root/'firmware/host-verify.json').read_text(encoding='utf-8'))
    if (record.get('version')!=1 or record.get('arch')!='linux-x86_64' or
            record.get('static') is not True or record.get('qemu_gate',{}).get('status')!='passed'):
        raise ValueError('Auditor release manifest is not verified')
    binary=root/record['binary']
    if sha(binary.read_bytes())!=record['sha256']: raise ValueError('Auditor binary SHA mismatch')
    if source_sha(root/'reference/nodes/manifest.json')!=record['reference_manifest_sha256']:
        raise ValueError('Auditor release gate/reference manifest mismatch')
    for name,digest in record['sources'].items():
        if source_sha(root/name)!=digest: raise ValueError('Auditor source/model mismatch: '+name)
    return binary


class Replay:
    def __init__(self,mode=16,binary=None,timeout=300):
        if mode not in (16,32): raise ValueError('Replay mode must be 16 or 32')
        self.mode=mode; self.timeout=timeout; self.failed=False
        if binary is None:
            if platform.system()!='Linux' or platform.machine() not in ('x86_64','amd64'):
                raise ValueError('Packaged auditor requires Linux x86_64')
            binary=verified_binary(ROOT)
        self.process=subprocess.Popen([str(Path(binary).resolve()),'--serve'],stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0)
        try:
            with self.deadline():
                line=self.process.stdout.readline(1024)
                info=json.loads(line)
                if info!={'version':1,'arch':'linux-x86_64','nodes':334,'operators':21,'selftest':'passed'}:
                    raise ValueError('Incompatible native auditor')
        except BaseException:
            self.close(); raise

    def deadline(self):
        owner=self
        class Guard:
            def __enter__(self):
                self.timer=threading.Timer(owner.timeout,owner.process.kill); self.timer.daemon=True; self.timer.start()
            def __exit__(self,*exc): self.timer.cancel()
        return Guard()

    def read(self,size):
        pieces=[]; remaining=size
        while remaining:
            data=self.process.stdout.read(remaining)
            if not data: raise RuntimeError('Native auditor exited/timed out')
            pieces.append(data); remaining-=len(data)
        return b''.join(pieces)

    def run(self,node,tensors,inputs):
        if self.failed: raise RuntimeError('Auditor session already failed')
        try:
            with self.deadline():
                self.write(struct.pack('<4I',0x31415259,node['id'],self.mode,len(node['inputs'])))
                for tid in node['inputs']:
                    data=inputs[tid]
                    if len(data)!=tensors[tid]['bytes']: raise ValueError('Replay input length mismatch')
                    self.write(struct.pack('<2I',tid,len(data))); self.write(data)
                self.process.stdin.flush()
                status,count=struct.unpack('<iI',self.read(8))
                ids=[tid for tid in node['outputs'] if tid>=0]
                if status or count!=len(ids): raise ValueError('Native node replay failed: '+str(status))
                result={}
                for tid in ids:
                    got,size=struct.unpack('<2I',self.read(8))
                    if got!=tid or size!=tensors[tid]['bytes']: raise ValueError('Native replay output metadata mismatch')
                    result[tid]=self.read(size)
                return result
        except BaseException:
            self.failed=True; self.close(); raise

    def write(self,data):
        view=memoryview(data)
        while view:
            size=self.process.stdin.write(view)
            if not size: raise RuntimeError('Native auditor input pipe closed')
            view=view[size:]

    def close(self):
        if self.process.poll() is None: self.process.kill()
        self.process.wait()
        self.process.stdin.close(); self.process.stdout.close()
