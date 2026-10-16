"""Host menu lifecycle tests use a temporary repository and no physical board."""
import json
import io
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




    def test_full_launch_is_detached_without_obsolete_probe(self):
        host.save_json(host.workspace()/'last-probe.json',{'directory':'/saved/probe'})
        with patch('host_control.Path.is_file',return_value=True),patch('host_control.subprocess.Popen') as popen:
            popen.return_value.pid=321
            host.launch_full()
        command=popen.call_args.args[0]
        self.assertEqual(command[2],str(self.root/'tools/full_check.py'))
        self.assertNotIn('--probe',command)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertEqual(len(popen.call_args.kwargs['pass_fds']),1)
        full=json.loads((host.workspace()/'last-full.json').read_text())
        job=json.loads((host.workspace()/'last-launch.json').read_text())
        self.assertEqual(full['directory'],job['result'])
        self.assertTrue(Path(job['result']).is_dir())

    def saved_bus(self, directory, name='summary.json'):
        bus={'passed':True,'nodes_captured':334,'outputs_captured':335,
             'inputs_verified_nodes':334,'final':{'normal_repeat_exact':True,'valid':True},
             'rows':[{'node':i,'kind':'COPY','status':'exact','outputs':[]} for i in range(334)]}
        report={'numeric_rule':__import__('full_replay').NUMERIC_RULE,'id':'old-paired','status':'failed','tile_gate':{'status':'passed'},
                'samples':{'bus':bus,'zidane':{'passed':False}}}
        (directory/name).write_text(json.dumps(report))
        return report

    def test_full_pages_read_saved_bus_without_replay_and_preserve_source(self):
        directory=self.root/'full'; directory.mkdir()
        self.saved_bus(directory)
        original=(directory/'summary.json').read_bytes()
        host.save_json(host.workspace()/'last-full.json',{'directory':str(directory)})
        with patch('host_control.subprocess.run') as run,patch('builtins.input',return_value=''), \
             patch('sys.stdout',new_callable=io.StringIO) as output:
            host.full_pages()
        run.assert_not_called()
        text=output.getvalue()
        for page in range(1,4): self.assertIn('PAGE={}/3'.format(page),text)
        self.assertEqual(text.count('STATUS: passed'),3)
        self.assertNotIn('zidane',text)
        self.assertEqual(original,(directory/'summary.json').read_bytes())

    def test_full_pages_accept_checkpoint_before_parent_finishes(self):
        directory=self.root/'full'; directory.mkdir()
        self.saved_bus(directory,'checkpoint.json')
        host.save_json(host.workspace()/'last-full.json',{'directory':str(directory)})
        with patch('host_control.subprocess.run') as run,patch('builtins.input',return_value=''), \
             patch.object(host,'progress') as progress:
            host.full_pages()
        run.assert_not_called(); progress.assert_not_called()

    def test_full_pages_missing_analysis_replays_raw_once_offline(self):
        directory=self.root/'full'; capture=directory/'bus/attempt-1'; capture.mkdir(parents=True)
        (capture/'summary.json').write_text('{"sample":"bus","status":"completed"}')
        host.save_json(host.workspace()/'last-full.json',{'directory':str(directory)})
        def offline(*args,**kwargs):
            report=self.saved_bus(directory,'bus-rtl-analysis.json')
            report.update(id='offline',status='passed')
            (directory/'bus-rtl-analysis.json').write_text(json.dumps(report))
        with patch('host_control.subprocess.run',side_effect=offline) as run, \
             patch('builtins.input',return_value=''):
            host.full_pages()
            host.full_pages()
        self.assertEqual(run.call_count,1)
        command=run.call_args.args[0]
        self.assertIn('--reanalyze',command)
        for forbidden in ('--output','--pbcopy','--load'): self.assertNotIn(forbidden,command)

    def test_full_pages_incomplete_data_reports_missing_without_restart(self):
        directory=self.root/'full'; directory.mkdir()
        host.save_json(host.workspace()/'last-full.json',{'directory':str(directory)})
        with patch('host_control.subprocess.run') as run,patch.object(host,'progress') as progress:
            with self.assertRaisesRegex(ValueError,'missing'): host.full_pages()
        run.assert_not_called(); progress.assert_not_called()

    def test_png_menu_only_runs_offline_exporter(self):
        with patch('host_control.subprocess.run') as run:
            host.export_png()
        command=run.call_args.args[0]
        self.assertEqual(command,[sys.executable,str(self.root/'tools/render_results.py'),'--latest'])

    def test_video_menu_and_host_check_respect_controller_lock(self):
        with patch('host_control.subprocess.run') as run:
            host.video(check=True)
            self.assertEqual(run.call_args.args[0][-2:],['--check','--no-display'])
            host.video()
            self.assertEqual(run.call_args.args[0][-2:],[str(self.root/'tools/host_video.py'),'--load'])
        with host.guard(host.workspace()/'control.lock'),patch('host_control.subprocess.run') as run:
            with self.assertRaises(RuntimeError): host.video()
            run.assert_not_called()

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
