"""Actual board C mailbox + host transport via executable fake PCIe tools."""
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zlib
from pcie_backend import BASE,MAILBOX,PBTools,PCIeBackend,STATUS,checked,INPUT_BYTES,OUTPUT_BYTES

BINARY=sys.argv.pop(1)


class PCIeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='pcie-test-')
        self.root=Path(self.temp.name)
        self.ddr=self.root/'ddr.bin'
        with self.ddr.open('wb') as f:
            f.truncate(32*1024*1024)
        self.process=subprocess.Popen([BINARY,str(self.ddr)],stdout=subprocess.PIPE)
        if not select.select([self.process.stdout],[],[],5)[0] or self.process.stdout.readline()!=b'READY\n':
            raise RuntimeError('Native PCIe fixture failed to start')
        for command in ('pbcopy','pbload'):
            path=self.root/command
            path.write_text(f'''#!{sys.executable}
from pathlib import Path
import sys,time
root=Path(__file__).parent
if (root/'delay').exists(): time.sleep(10)
if (root/'fail').exists(): print('injected failure',file=sys.stderr); sys.exit(7)
assert sys.argv[1]=='-d'
args=sys.argv[2].split(':')
file=Path(args[0]); offset=int(args[1],0)
with (root/'ddr.bin').open('r+b',buffering=0) as ddr:
    ddr.seek(offset)
    if Path(sys.argv[0]).name=='pbcopy':
        data=file.read_bytes()
        if (root/'badinput').exists() and offset==0x800000: data=bytes([data[0]^1])+data[1:]
        ddr.write(data)
    else:
        data=ddr.read(int(args[2],0))
        if (root/'short').exists(): data=data[:-1]
        if (root/'badresult').exists() and offset==0xa00000: data=bytes([data[0]^1])+data[1:]
        file.write_bytes(data)
''')
            path.chmod(0o755)
        self.backend=None
        self.tools=None

    def tearDown(self):
        if self.backend: self.backend.close()
        elif self.tools: self.tools.close()
        self.process.terminate()
        self.process.wait(timeout=3)
        self.process.stdout.close()
        self.temp.cleanup()

    def connect(self,**kwargs):
        self.backend=PCIeBackend(str(self.root/'pbcopy'),str(self.root/'pbload'),poll_interval=.01,**kwargs)
        return self.backend

    def read(self,address,n):
        with self.ddr.open('rb',buffering=0) as f:
            f.seek(address-BASE); return f.read(n)

    def write(self,address,data):
        with self.ddr.open('r+b',buffering=0) as f:
            f.seek(address-BASE); f.write(data)

    def tensor(self,value=1):
        return struct.pack('<f',value)*(INPUT_BYTES//4)

    def test_continuous_frames_and_duplicate_doorbell(self):
        backend=self.connect()
        for i in (1,2,3):
            result=backend.infer_bytes(self.tensor(i))
            self.assertEqual(len(result),OUTPUT_BYTES)
            self.assertEqual(struct.unpack_from('<f',result)[0],i)
            self.assertEqual(backend.last_status['frame'],i)
        # Re-publishing the same commit must never rerun inference.
        self.write(MAILBOX+128,struct.pack('<I',backend.sequence))
        time.sleep(.03)
        self.write(MAILBOX+128,struct.pack('<I',backend.sequence-1))
        time.sleep(.03)
        self.assertEqual(struct.unpack('<I',self.read(0x80a20000,4))[0],3)
        self.assertEqual(STATUS.unpack(checked(self.read(MAILBOX+192,64)))[6],backend.sequence)

    def test_bad_input_crc_never_runs_model(self):
        backend=self.connect()
        (self.root/'badinput').touch()
        with self.assertRaisesRegex(RuntimeError,'Board PCIe error'):
            backend.infer_bytes(self.tensor())
        self.assertEqual(backend.last_status['error'],0x42)
        self.assertEqual(struct.unpack('<I',self.read(0x80a20000,4))[0],0)

    def test_corrupt_result_rejected(self):
        backend=self.connect()
        (self.root/'badresult').touch()
        with self.assertRaisesRegex(ValueError,'CRC/length'):
            backend.infer_bytes(self.tensor())
        with self.assertRaisesRegex(RuntimeError,'session failed'):
            backend.infer_bytes(self.tensor())

    def test_model_failure_diagnostic(self):
        backend=self.connect()
        self.write(0x80a20004,struct.pack('<I',1))
        with self.assertRaisesRegex(RuntimeError,'Board PCIe error'):
            backend.infer_bytes(self.tensor())
        self.assertEqual(backend.last_status['detail'],-2)
        self.assertEqual(backend.last_diagnostic['detail'],-77)

    def test_invalid_classes_and_nonfinite(self):
        backend=self.connect()
        self.write(0x80a2000c,struct.pack('<I',1))
        with self.assertRaisesRegex(ValueError,'values/classes'):
            backend.infer_bytes(self.tensor())
        backend.close(); self.backend=None
        self.write(0x80a2000c,struct.pack('<2I',0,1))
        backend=self.connect()
        with self.assertRaisesRegex(ValueError,'values/classes'):
            backend.infer_bytes(self.tensor())

    def test_timeout_and_reconnect_wait_for_board(self):
        backend=self.connect(infer_timeout=1.5)
        self.write(0x80a20008,struct.pack('<I',1))
        with self.assertRaises(TimeoutError):
            backend.infer_bytes(self.tensor())
        self.assertTrue(backend.failed)
        backend.close(); self.backend=None
        timer=threading.Timer(.15,lambda:self.write(0x80a20008,struct.pack('<I',0)))
        timer.start()
        try:
            backend=self.connect(infer_timeout=3)
            self.assertEqual(struct.unpack_from('<f',backend.infer_bytes(self.tensor(4)))[0],4)
        finally:
            timer.join()

    def test_request_crc_and_session_rejection(self):
        backend=self.connect()
        request=bytearray(self.read(MAILBOX+64,64))
        sequence=backend.sequence+1
        struct.pack_into('<II',request,16,2,sequence)
        self.write(MAILBOX+64,request)
        self.write(MAILBOX+128,struct.pack('<I',sequence))
        time.sleep(.05)
        self.assertEqual(STATUS.unpack(checked(self.read(MAILBOX+192,64)))[3],0x40)
        sequence+=1
        struct.pack_into('<Q',request,0,backend.session^1 or 2)
        struct.pack_into('<I',request,20,sequence)
        struct.pack_into('<I',request,60,zlib.crc32(request[:60]))
        self.write(MAILBOX+64,request)
        self.write(MAILBOX+128,struct.pack('<I',sequence))
        time.sleep(.05)
        self.assertEqual(STATUS.unpack(checked(self.read(MAILBOX+192,64)))[3],0x41)

    def test_tool_return_short_file_and_cancel(self):
        self.tools=PBTools(str(self.root/'pbcopy'),str(self.root/'pbload'))
        deadline=time.monotonic()+10
        (self.root/'fail').touch()
        with self.assertRaisesRegex(RuntimeError,'failed \\(7\\)'):
            self.tools.read(MAILBOX,64,deadline)
        (self.root/'fail').unlink(); (self.root/'short').touch()
        with self.assertRaisesRegex(ValueError,'63 bytes'):
            self.tools.read(MAILBOX,64,deadline)
        (self.root/'short').unlink(); (self.root/'delay').touch()
        timer=threading.Timer(.1,self.tools.stop.set); timer.start()
        try:
            with self.assertRaises(InterruptedError): self.tools.read(MAILBOX,64,deadline)
        finally: timer.join()

    def test_wrong_frame_and_partial_status(self):
        backend=self.connect()
        original=backend.link.read
        inject={'count':1,'wrong':False}
        def read(address,size,deadline):
            data=original(address,size,deadline)
            if address==MAILBOX+192:
                if inject['count']:
                    inject['count']-=1
                    return bytes([data[0]^1])+data[1:]
                if inject['wrong']:
                    state=STATUS.unpack(data)
                    if state[2]==3 and state[6]==backend.sequence:
                        data=bytearray(data)
                        struct.pack_into('<Q',data,24,999)
                        struct.pack_into('<I',data,60,zlib.crc32(data[:60]))
            return data
        backend.link.read=read
        self.assertEqual(struct.unpack_from('<f',backend.infer_bytes(self.tensor()))[0],1)
        inject['wrong']=True
        with self.assertRaisesRegex(ValueError,'session/frame mismatch'):
            backend.infer_bytes(self.tensor())

    def test_single_owner_lock(self):
        self.connect()
        with self.assertRaisesRegex(RuntimeError,'Another host'):
            PBTools(str(self.root/'pbcopy'),str(self.root/'pbload'))

    def test_video_flow(self):
        # Exercise real CLI, transport, preprocess/postprocess and video recorder.
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest('Video regression needs optional NumPy/OpenCV')
        source=self.root/'source.mp4'
        writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,64))
        self.assertTrue(writer.isOpened())
        for i in range(10): writer.write(np.full((64,64,3),i*20,np.uint8))
        writer.release()
        script=Path(__file__).with_name('live_demo.py')
        result=subprocess.run([sys.executable,str(script),'--backend','pcie','--pbcopy',str(self.root/'pbcopy'),
            '--pbload',str(self.root/'pbload'),'--source',str(source),'--no-display','--max-results','2',
            '--infer-timeout','15','--save-frames','--board-trace'],capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        directory=Path(result.stdout.strip().splitlines()[-1])
        import json
        report=json.loads((directory/'report.json').read_text())
        self.assertEqual(len(report['results']),2)
        self.assertEqual([r['pcie_timing']['pcie_frame'] for r in report['results']],[1,2])
        self.assertEqual(len(list(directory.glob('source-*.png'))),2)
        self.assertEqual(len(list(directory.glob('detected-*.png'))),2)
        capture=cv2.VideoCapture(str(directory/'demo.mp4'))
        self.assertTrue(capture.read()[0]); capture.release()

    def test_trace_and_finite_options(self):
        backend=self.connect(trace=True,check_finite=True)
        backend.infer_bytes(self.tensor())
        self.assertEqual(struct.unpack_from('<I',self.read(MAILBOX+64,64),32)[0],6)
        self.assertEqual(backend.last_diagnostic['options'],6)

    def test_unsupported_option_rejected(self):
        backend=self.connect()
        backend.options=8
        with self.assertRaisesRegex(RuntimeError,'Board PCIe error'):
            backend.infer_bytes(self.tensor())
        self.assertEqual(backend.last_status['error'],0x40)
        self.assertEqual(struct.unpack('<I',self.read(0x80a20000,4))[0],0)


if __name__=='__main__':
    unittest.main()
