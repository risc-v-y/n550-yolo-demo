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

    def test_probe_reports_are_not_selected_as_node_captures(self):
        directory=host.workspace()/'run-probe/capture'; directory.mkdir(parents=True)
        (directory/'summary.json').write_text('{"status":"completed","runs":[]}')
        self.assertEqual(host.candidates(),[])

    def test_restore_refuses_active_job(self):
        with host.guard(host.workspace()/'control.lock'),patch('host_control.subprocess.run') as run:
            with self.assertRaises(RuntimeError): host.restore()
            run.assert_not_called()

    def test_probe_report_reanalyzes_once_then_displays_same_saved_analysis(self):
        directory=self.root/'saved-probe'; directory.mkdir()
        (directory/'summary.json').write_text('{"status":"completed"}')
        host.save_json(host.workspace()/'last-probe.json',{'directory':str(directory)})
        with patch('host_control.subprocess.run') as run,patch('builtins.input',return_value=''):
            host.probe_pages()
        self.assertEqual(run.call_count,3)
        for page,call in enumerate(run.call_args_list,1):
            command=call.args[0]
            self.assertEqual(command[:4],[sys.executable,str(self.root/'tools/tile_probe.py'),'--report',str(directory)])
            self.assertEqual(command[4:6],['--page',str(page)])
            self.assertEqual('--reanalyze' in command,page==1)
            self.assertNotIn('--load',command)
            self.assertNotIn('--pbcopy',command)
            self.assertNotIn('--pbload',command)

    def test_failed_reanalysis_does_not_display_old_cached_pages(self):
        directory=self.root/'saved-probe'; directory.mkdir()
        (directory/'summary.json').write_text('{"status":"completed"}')
        host.save_json(host.workspace()/'last-probe.json',{'directory':str(directory)})
        with patch('host_control.subprocess.run',side_effect=subprocess.CalledProcessError(1,['analysis'])) as run, \
             patch('builtins.input') as prompt:
            with self.assertRaises(subprocess.CalledProcessError): host.probe_pages()
        self.assertEqual(run.call_count,1)
        prompt.assert_not_called()

    def test_probe_launch_uses_loader_and_records_report_pointer(self):
        previous=self.root/'old-capture'; previous.mkdir()
        (previous/'summary.json').write_text('{"sample":"bus"}')
        with patch('host_control.Path.is_file',return_value=True), \
             patch.object(host,'selected',return_value=previous), \
             patch('host_control.subprocess.Popen') as popen:
            popen.return_value.pid=123
            host.launch(probe=True)
        command=popen.call_args.args[0]
        self.assertIn('--load',command)
        self.assertIn('tile_probe.py',command[2])
        self.assertEqual(command[-2:],['--capture',str(previous)])
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertEqual(len(popen.call_args.kwargs['pass_fds']),1)
        job=json.loads((host.workspace()/'last-launch.json').read_text())
        saved=json.loads((host.workspace()/'last-probe.json').read_text())
        self.assertEqual(saved['directory'],job['result'])

    def test_full_launch_is_detached_and_passes_saved_probe(self):
        host.save_json(host.workspace()/'last-probe.json',{'directory':'/saved/probe'})
        with patch('host_control.Path.is_file',return_value=True),patch('host_control.subprocess.Popen') as popen:
            popen.return_value.pid=321
            host.launch_full()
        command=popen.call_args.args[0]
        self.assertEqual(command[2],str(self.root/'tools/full_check.py'))
        self.assertEqual(command[-2:],['--probe','/saved/probe'])
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertEqual(len(popen.call_args.kwargs['pass_fds']),1)
        full=json.loads((host.workspace()/'last-full.json').read_text())
        job=json.loads((host.workspace()/'last-launch.json').read_text())
        self.assertEqual(full['directory'],job['result'])
        self.assertTrue(Path(job['result']).is_dir())

    def test_full_pages_only_read_existing_report(self):
        directory=self.root/'full'; directory.mkdir(); (directory/'summary.json').write_text('{}')
        host.save_json(host.workspace()/'last-full.json',{'directory':str(directory)})
        with patch('host_control.subprocess.run') as run,patch('builtins.input',return_value=''):
            host.full_pages()
        self.assertEqual(run.call_count,3)
        for page,call in enumerate(run.call_args_list,1):
            command=call.args[0]
            self.assertIn('--report',command)
            self.assertEqual(command[-2:],['--page',str(page)])
            self.assertNotIn('--output',command)

    def test_png_menu_only_runs_offline_exporter(self):
        with patch('host_control.subprocess.run') as run:
            host.export_png()
        command=run.call_args.args[0]
        self.assertEqual(command,[sys.executable,str(self.root/'tools/render_results.py'),'--latest'])

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
