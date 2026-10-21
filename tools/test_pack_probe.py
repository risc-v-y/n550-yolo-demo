"""First-block snapshot integrity, failure boundaries and stalled progress."""
import json
import hashlib
from types import SimpleNamespace
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import pack_probe as app


def payload(**values):
    record=dict.fromkeys(app.FIELDS,0)
    record.update(magic=app.MAGIC,version=1,bytes=256,generation=2,session=7,frame=1,node=0,
                  stage=11,columns=32,k=27,column=25,vl=27,row=(1<<64)-1,
                  input=0xf0081040,cache=0xf0041000,cache_bytes=7488,converted=2496)
    record.update(values)
    words=[record[key] for key in app.FIELDS]
    checksum=14695981039346656037
    for value in words[:-1]: checksum=((checksum^value)*1099511628211)&((1<<64)-1)
    words[-1]=checksum
    return app.WIRE.pack(*words)


class PackTests(unittest.TestCase):
    def test_identity_integrity_and_signed_row(self):
        probe=app.decode(payload(),7,1)
        self.assertEqual(probe['row'],-1)
        self.assertEqual(probe['stage_name'],'gather.begin')
        self.assertIsNone(app.decode(bytes(256),7,1))
        with self.assertRaises(ValueError): app.decode(payload()[:-1],7,1)
        for fields in (dict(magic=0),dict(session=8),dict(frame=2),dict(node=1),dict(first=32),dict(stage=0),dict(next_inner=28)):
            with self.assertRaises(ValueError): app.decode(payload(**fields),7,1)
        for raw in (payload(generation=3),payload()[:80]+b'\1'+payload()[81:]):
            with self.assertRaises(app.UnstableProbe): app.decode(raw,7,1)

    def backend(self,path,report,raw):
        backend=app.ProbeBackend.__new__(app.ProbeBackend)
        backend.probe_address=0x80950000; backend.probe_output=path; backend.probe_report=report
        backend.probe_last_progress=None; backend.probe_progress_at=None; backend.probe_stall_timeout=20
        backend.session=7; backend.frame=1; backend.diagnostic=lambda event: None
        class Link:
            def read(self,*args): return raw
        backend.link=Link()
        return backend

    def test_stall_detects_unchanged_completed_work_even_if_generation_moves(self):
        with tempfile.TemporaryDirectory() as name:
            report=dict(samples=[],unstable_reads=0)
            backend=self.backend(Path(name),report,payload())
            with patch.object(app.time,'monotonic',side_effect=[1,22]):
                backend.sample_probe(100)
                with self.assertRaises(app.ProbeStalled): backend.sample_probe(100)
            self.assertEqual(len(report['samples']),2)
            self.assertTrue((Path(name)/'probe-001.bin').is_file())

    def test_completed_block_never_triggers_pack_stall(self):
        with tempfile.TemporaryDirectory() as name:
            report=dict(samples=[],unstable_reads=0)
            backend=self.backend(Path(name),report,payload(stage=16,column=32,next_column=32,next_inner=0))
            with patch.object(app.time,'monotonic',side_effect=[1,200]):
                backend.sample_probe(1000); backend.sample_probe(1000)
            page=app.render(dict(report,status='completed',id='test',frame_status='completed'))
            self.assertIn('block.done',page)
            self.assertLessEqual(len(page.splitlines()),22)
            self.assertTrue(all(len(line)<=80 for line in page.splitlines()))

    def test_unstable_reads_are_bounded_and_guard_reason_is_reported(self):
        with tempfile.TemporaryDirectory() as name:
            report=dict(samples=[],unstable_reads=0)
            backend=self.backend(Path(name),report,payload(generation=3))
            self.assertIsNone(backend.sample_probe(100))
            self.assertEqual(report['unstable_reads'],3)
            backend.link.read=lambda *args: payload(stage=17,error=5)
            backend.sample_probe(100)
            page=app.render(dict(report,status='failed',id='test',error='x'*300))
            self.assertIn('invalid VL/K increment',page)
            self.assertLessEqual(len(page.splitlines()),22)
            self.assertTrue(all(len(line)<=80 for line in page.splitlines()))

    def test_timeout_preserves_last_snapshot_and_closes_link(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); output=root/'result'; output.mkdir()
            (root/'firmware').mkdir(); (root/'firmware/demo-pcie.elf').write_bytes(b'elf')
            (root/'reference/prepared').mkdir(parents=True)
            data=bytes(app.INPUT_BYTES); (root/'input.bin').write_bytes(data)
            (root/'reference/prepared/manifest.json').write_text(json.dumps(dict(samples=dict(bus=dict(
                input='input.bin',input_sha256=hashlib.sha256(data).hexdigest())))))
            state={}
            class Backend:
                frame=1; last_diagnostic=None
                def __init__(self,address,directory,report,*args,**kwargs):
                    state['trace']=kwargs['trace']; self.report=report
                    self.link=SimpleNamespace(read=lambda *args:b'prefix')
                def infer_bytes(self,data): raise TimeoutError('synthetic stalled core')
                def sample_probe(self,deadline,enforce_stall=True):
                    state['final_guard']=enforce_stall
                    self.report['samples'].append(dict(probe=app.decode(payload(),7,1),file='fixture.bin'))
                def _snapshot(self,deadline):
                    self.last_diagnostic=dict(node=0,phase_name='amu_pack',error=0)
                def close(self): state['closed']=True
            args=SimpleNamespace(infer_timeout=120,stall_timeout=20,tool_timeout=30,load=False,
                                 pbcopy='copy',pbload='read',reset_dir='reset')
            with patch.object(app,'ROOT',root),patch.object(app,'ProbeBackend',Backend), \
                 patch.object(app.diag,'firmware_info',return_value=({},None,b'prefix')), \
                 patch.object(app.diag,'elf_symbols',return_value=dict(board_pack_probe=0x80950000)):
                self.assertEqual(app.execute(args,output),1)
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['last_board_diagnostic']['phase_name'],'amu_pack')
            self.assertEqual(report['samples'][-1]['probe']['column'],25)
            self.assertEqual(state,dict(trace=True,final_guard=False,closed=True))

    def test_firmware_preflight_failure_never_loads_or_connects(self):
        with tempfile.TemporaryDirectory() as name:
            args=SimpleNamespace(infer_timeout=120,stall_timeout=20,load=True)
            with patch.object(app.diag,'firmware_info',side_effect=ValueError('checksum')), \
                 patch.object(app.subprocess,'run') as load,patch.object(app,'ProbeBackend') as connect:
                self.assertEqual(app.execute(args,Path(name)),1)
            load.assert_not_called(); connect.assert_not_called()

    def test_preserve_current_first_layer_input_and_reject_crc_damage(self):
        import zlib
        data=bytes(app.INPUT_BYTES)
        info=[app.PCIE_MAGIC,1,256,1,0xf0081040,0xf0300000,0x80e60000,app.INPUT_BYTES,7200,832,334,0,0]
        def finish(body): return body+struct.pack('<I',zlib.crc32(body))
        info_raw=finish(app.INFO.pack(*info)[:60])
        state=[app.PCIE_MAGIC,1,app.BUSY,0,7,1,2,0,0,0,2,0,0,0]
        state_raw=finish(app.STATUS.pack(*state)[:60])
        request=finish(struct.pack('<QQ11I',7,1,2,2,app.INPUT_BYTES,zlib.crc32(data),0,0,0,0,0,0,0))
        words=[0]*104; words[0]=1; words[1]=7; words[2]=1; words[3]=0; words[5]=5
        diagnostic=struct.pack('<104Q',*words)
        store={app.MAILBOX:info_raw,app.MAILBOX+192:state_raw,app.MAILBOX+64:request,
               0x80e60000:diagnostic,0xf0081040:data}
        class Link:
            closed=False
            def read(self,address,size,deadline): return store[address]
            def close(self): self.closed=True
        args=SimpleNamespace(pbcopy='copy',pbload='read',tool_timeout=30)
        link=Link(); report={}
        with patch.object(app,'PBTools',return_value=link):
            self.assertEqual(app.preserved_input(args,report,SimpleNamespace(emit=lambda event:None)),data)
        self.assertTrue(link.closed)
        self.assertEqual(report['input']['source'],'preserved first-layer DDR request')
        store[0xf0081040]=b'\1'+data[1:]
        with patch.object(app,'PBTools',return_value=Link()):
            with self.assertRaisesRegex(ValueError,'CRC'): app.preserved_input(args,{},SimpleNamespace(emit=lambda event:None))
        state[2]=4  # A failed core must not be reset by the preservation path.
        store[app.MAILBOX+192]=finish(app.STATUS.pack(*state)[:60])
        with patch.object(app,'PBTools',return_value=Link()):
            with self.assertRaisesRegex(ValueError,'no reset'): app.preserved_input(args,{},SimpleNamespace(emit=lambda event:None))


if __name__=='__main__': unittest.main()
