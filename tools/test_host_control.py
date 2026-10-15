"""Host menu lifecycle tests use a temporary repository and no physical board."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import host_control as host


@unittest.skipUnless(sys.platform=='linux','Uses flock and detached Linux processes')
class MenuTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='host menu ')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.root_patch=patch.object(host,'ROOT',self.root)
        self.root_patch.start(); self.addCleanup(self.root_patch.stop)
        self.board_patch=patch.object(host,'board_lock',return_value=self.root/'board.lock')
        self.board_patch.start(); self.addCleanup(self.board_patch.stop)

    def test_update_refuses_active_capture_or_board_controller(self):
        for path in (host.workspace()/'control.lock',host.board_lock()):
            with host.guard(path),patch('host_control.subprocess.run') as run:
                with self.assertRaises(RuntimeError): host.update()
                run.assert_not_called()

    def test_update_failure_propagates(self):
        with patch('host_control.subprocess.run',side_effect=subprocess.CalledProcessError(1,['git'])):
            with self.assertRaises(subprocess.CalledProcessError): host.update()

    def test_report_selection_uses_completion_pointer_despite_backdated_clock(self):
        base=self.root/'yolo26_pc/outputs/node-capture'; base.mkdir(parents=True)
        for name in ('2021-run','2026-run'):
            run=base/name; run.mkdir(); (run/'summary.json').write_text('{"status":"failed"}')
        (base/'latest.json').write_text(json.dumps({'directory':str(base/'2021-run')}))
        self.assertEqual(host.selected(),base/'2021-run')
        with patch('builtins.input',return_value='2'): host.choose()
        self.assertEqual(host.selected(),base/'2026-run')

    def test_detached_capture_keeps_lock_and_saves_paths(self):
        script=self.root/'yolo26_pc/node_capture.py'; script.parent.mkdir(parents=True)
        script.write_text('''import json,sys,time,os
from pathlib import Path
root=Path(__file__).parent.parent
(root/'args.json').write_text(json.dumps(sys.argv[1:]))
(root/'session').write_text(str(os.getsid(0)))
while not (root/'release').exists(): time.sleep(.02)
''')
        with patch('host_control.Path.is_file',return_value=True):
            child=host.launch()
        try:
            limit=time.monotonic()+5
            while not (self.root/'session').exists() and time.monotonic()<limit: time.sleep(.02)
            self.assertEqual(int((self.root/'session').read_text()),child.pid)
            job=json.loads((host.workspace()/'last-launch.json').read_text())
            args=json.loads((self.root/'args.json').read_text())
            self.assertIn('--recover-paused',args)
            self.assertEqual(args[-1],job['result'])
            self.assertTrue(Path(job['log']).exists())
            with self.assertRaises(RuntimeError): host.update()
            with self.assertRaises(RuntimeError): host.launch()
            (self.root/'release').touch(); child.wait(timeout=5)
            with host.guard(host.workspace()/'control.lock'): pass
        finally:
            if child.poll() is None: child.kill(); child.wait(timeout=5)


if __name__=='__main__': unittest.main()
