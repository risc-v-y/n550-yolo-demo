"""Read crash recovery preserves deadlines, failure logs and data validation."""
from pathlib import Path
import signal
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
import zlib

from node_capture import read_tensor
from pcie_backend import BASE, PBTools, PCIeToolError


class ReadRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.link=PBTools.__new__(PBTools)
        self.link.path=Path(self.temp.name)/'read.bin'
        self.link.cwd=Path(self.temp.name)
        self.link.commands=['pbcopy','pbload']
        self.link.timeout=5
        self.link.stop=threading.Event()
        self.events=[]
        self.link.diagnostic=self.events.append
        self.deadline=time.monotonic()+5

    def test_crashed_full_file_removed_before_retry_and_failure_log_preserved(self):
        output=b'first failure\n'+b'x'*9000
        seen=[]
        def launch(command,**kwargs):
            seen.append(command)
            self.assertFalse(self.link.path.exists())
            self.link.path.write_bytes(b'bad!' if len(seen)==1 else b'good')
            process=Mock(returncode=-signal.SIGSEGV if len(seen)==1 else 0)
            process.communicate.return_value=(output if len(seen)==1 else b'OK',None)
            return process
        with patch('pcie_backend.subprocess.Popen',side_effect=launch):
            self.assertEqual(self.link.read(BASE,4,self.deadline),b'good')
        self.assertEqual(len(seen),2)
        self.assertEqual(self.events[0]['output'],output.decode())
        self.assertEqual(self.events[1]['type'],'pcie_read_retry')

    def fail(self,code):
        self.link.path.write_bytes(b'bad!')
        raise PCIeToolError('pbload',code,'test failure')

    def test_three_crashes_stop_and_discard_last_file(self):
        with patch.object(self.link,'_run',side_effect=lambda *a:self.fail(-signal.SIGSEGV)) as run:
            with self.assertRaises(PCIeToolError):
                self.link.read(BASE,4,self.deadline)
        self.assertEqual(run.call_count,3)
        self.assertFalse(self.link.path.exists())

    def test_other_errors_are_not_retried(self):
        for code in (1,0,-signal.SIGINT,-9):
            with self.subTest(code=code), patch.object(self.link,'_run',side_effect=lambda *a:self.fail(code)) as run:
                with self.assertRaises(PCIeToolError):
                    self.link.read(BASE,4,self.deadline)
                self.assertEqual(run.call_count,1)

    def test_writes_are_never_retried(self):
        with patch.object(self.link,'_run',side_effect=PCIeToolError('pbcopy',-signal.SIGSEGV,'failed')) as run:
            with self.assertRaises(PCIeToolError):
                self.link.write(BASE,b'data',self.deadline)
        self.assertEqual(run.call_count,1)

    def test_short_success_is_rejected(self):
        with patch.object(self.link,'_run',side_effect=lambda *a:self.link.path.write_bytes(b'x')) as run:
            with self.assertRaisesRegex(ValueError,'expected 4'):
                self.link.read(BASE,4,self.deadline)
        self.assertEqual(run.call_count,1)

    def test_retried_data_still_requires_tensor_crc(self):
        def read(*args):
            if not self.events:
                self.fail(-signal.SIGSEGV)
            self.link.path.write_bytes(b'bad!')
        descriptor={'id':375,'address':BASE,'bytes':4,'crc_after':zlib.crc32(b'good')}
        backend=Mock(link=self.link)
        with patch.object(self.link,'_run',side_effect=read):
            with self.assertRaisesRegex(ValueError,'CRC mismatch'):
                read_tensor(backend,descriptor,self.deadline)

    def test_deadline_not_extended(self):
        with patch.object(self.link,'_run',side_effect=lambda *a:self.fail(-signal.SIGSEGV)) as run:
            with self.assertRaises(PCIeToolError):
                self.link.read(BASE,4,time.monotonic()-1)
        self.assertEqual(run.call_count,1)

    def test_cancellation_stops_retry(self):
        def crash(*args):
            self.link.stop.set()
            self.fail(-signal.SIGSEGV)
        with patch.object(self.link,'_run',side_effect=crash) as run:
            with self.assertRaises(InterruptedError):
                self.link.read(BASE,4,self.deadline)
        self.assertEqual(run.call_count,1)


if __name__=='__main__':
    unittest.main()
