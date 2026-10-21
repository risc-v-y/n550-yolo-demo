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
from contextlib import ExitStack
from unittest.mock import patch
import host_control as host


class MenuRoutingTests(unittest.TestCase):
    def invoke(self,answers,action=None):
        argv=['host_control.py']+([action] if action else [])
        with patch.object(sys,'argv',argv),patch.object(sys,'platform','linux'), \
             patch('builtins.input',side_effect=answers), \
             patch('os.execv') as reload_menu,patch('sys.stdout',new_callable=io.StringIO) as output:
            result=host.main()
        return result,output.getvalue(),reload_menu

    def test_every_main_number_dispatches_its_displayed_action(self):
        cases=[('1','update',{}),('2','video',{'check':True}),('3','video',{}),
               ('4','video',{'continuous':True}),('5','video_pages',{}),
               ('6','performance_pages',{}),('7','accuracy_pages',{}),
               ('8','diagnostics',{}),('9','cleanup',{})]
        names={name for _,name,_ in cases}
        for number,target,kwargs in cases:
            with self.subTest(number=number),ExitStack() as stack:
                handlers={name:stack.enter_context(patch.object(host,name)) for name in names}
                result,_,reload_menu=self.invoke([number,'0'])
                self.assertEqual(result,0)
                handlers[target].assert_called_once_with(**kwargs)
                for name,handler in handlers.items():
                    if name!=target: handler.assert_not_called()
                self.assertEqual(reload_menu.call_count,int(number=='1'))

    def test_main_is_contiguous_and_omits_duplicate_capture_entries(self):
        result,text,_=self.invoke(['0'])
        numbers=[line.split()[0] for line in text.splitlines() if line[:1].isdigit()]
        self.assertEqual(numbers,[str(i) for i in range(1,10)]+['0'])
        self.assertIn('3 Run 3 video frames',text)
        self.assertIn('5 Saved video report',text)
        self.assertNotIn('Saved first difference',text)
        self.assertNotIn('Start BUS capture',text)
        self.assertEqual(result,0)

    def test_diagnostic_submenu_routes_and_returns_without_exiting_main(self):
        for number,target in [('1','progress'),('2','launch_full'),
                              ('3','full_pages'),('4','restore'),('5','pack_probe')]:
            with self.subTest(number=number),ExitStack() as stack:
                handlers={name:stack.enter_context(patch.object(host,name))
                          for name in ('progress','launch_full','full_pages','restore','pack_probe','video')}
                result,text,_=self.invoke(['8',number,'0','0'])
                self.assertEqual(result,0)
                handlers[target].assert_called_once_with()
                for name,handler in handlers.items():
                    if name!=target: handler.assert_not_called()
                self.assertEqual(text.count('YOLO host menu'),2)
                self.assertIn('0 Return to main menu',text)

    def test_invalid_or_retired_numbers_do_not_start_actions(self):
        with patch.object(host,'video') as video,patch.object(host,'launch_full') as full, \
             patch.object(host,'restore') as restore,patch.object(host,'cleanup') as cleanup:
            result,text,_=self.invoke(['15','20','-1','abc','8','11','0','0'])
        self.assertEqual(result,0)
        self.assertEqual(text.count('Choose one of the displayed options.'),5)
        for action in (video,full,restore,cleanup): action.assert_not_called()

    def test_legacy_golden_report_number_is_offline_alias_only(self):
        with patch.object(host,'video_pages') as report,patch.object(host,'video') as video, \
             patch.object(host,'restore') as restore:
            result,text,_=self.invoke(['17','0'])
        self.assertEqual(result,0)
        report.assert_called_once_with()
        video.assert_not_called(); restore.assert_not_called()
        self.assertNotIn('\n17 ',text)

    def test_submenu_failure_stays_in_submenu_and_keeps_main_usable(self):
        with patch.object(host,'launch_full',side_effect=RuntimeError('already active')), \
             patch.object(host,'performance_pages') as performance:
            result,text,_=self.invoke(['8','2','0','6','0'])
        self.assertEqual(result,0)
        self.assertEqual(text.count('YOLO diagnostics'),2)
        self.assertIn('ERROR: already active',text)
        performance.assert_called_once_with()

    def test_failed_update_never_reloads_menu(self):
        with patch.object(host,'update',side_effect=subprocess.CalledProcessError(1,['git'])):
            result,text,reload_menu=self.invoke(['1','0'])
        self.assertEqual(result,0)
        self.assertIn('ERROR:',text)
        reload_menu.assert_not_called()

    def test_existing_named_commands_do_not_depend_on_menu_numbers(self):
        cases=[('update','update',(),{}),('capture','launch',(),{}),
               ('status','pages',('status',),{}),('difference','pages',('difference',),{}),
               ('progress','progress',(),{}),('restore','restore',(),{}),
               ('full-check','launch_full',(),{}),('full-report','full_pages',(),{}),
               ('png','export_png',(),{}),('video-check','video',(),{'check':True}),
               ('video','video',(),{}),('video-continuous','video',(),{'continuous':True}),
               ('video-report','video_pages',(),{}),('performance','performance_pages',(),{}),
               ('accuracy','accuracy_pages',(),{}),('cleanup','cleanup',(),{})]
        for name,target,args,kwargs in cases:
            with self.subTest(command=name),patch.object(host,target) as action:
                result,_,reload_menu=self.invoke([],name)
            self.assertEqual(result,0)
            action.assert_called_once_with(*args,**kwargs)
            reload_menu.assert_not_called()

    def test_named_command_failure_returns_failure(self):
        with patch.object(host,'video',side_effect=RuntimeError('cannot display')):
            result,text,_=self.invoke([],'video')
        self.assertEqual(result,1)
        self.assertIn('cannot display',text)


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

    def test_continuous_video_defaults_and_custom_headless_options(self):
        with patch('builtins.input',side_effect=['','','']),patch('host_control.subprocess.run') as run:
            host.video(continuous=True)
        self.assertEqual(run.call_args.args[0][-3:],['--load','--max-frames','30'])
        source=self.root/'my video.mp4'; source.touch()
        with patch('builtins.input',side_effect=[str(source),'0','n']),patch('host_control.subprocess.run') as run:
            host.video(continuous=True)
        self.assertEqual(run.call_args.args[0][-5:],['--source',str(source),'--max-frames','0','--no-display'])

    def test_continuous_invalid_options_never_launch_board(self):
        for answers in (['','-1','y'],['missing.mp4','30','y']):
            with patch('builtins.input',side_effect=answers),patch('host_control.subprocess.run') as run:
                with self.assertRaises(ValueError): host.video(continuous=True)
                run.assert_not_called()

    def test_video_reports_reuse_old_results_and_respect_latest_pointer(self):
        base=self.root/'yolo26_pc/outputs/host-video'
        old=base/'2021-old'; old.mkdir(parents=True)
        (old/'report.json').write_text('{"status":"completed","results":[{"frame_id":0}]}')
        newer=base/'2026-new'; newer.mkdir()
        (newer/'report.json').write_text('{"status":"failed","results":[]}')
        host.save_json(base/'latest.json',{'directory':str(old)})
        from types import SimpleNamespace
        stub=SimpleNamespace(dependencies=lambda: 'fake video libraries')
        with patch.dict(sys.modules,{'host_video':stub}), \
             patch('video_report.analyze',return_value={'preview':None}) as analyze, \
             patch('video_report.render',side_effect=lambda s,p:'PAGE={}/3\n'.format(p)), \
             patch('builtins.input',side_effect=['','','','','']), \
             patch('host_control.subprocess.run') as run,patch('sys.stdout',new_callable=io.StringIO) as output:
            host.video_pages()
        run.assert_not_called()
        analyze.assert_called_once_with(old.resolve(),'fake video libraries')
        for p in (1,2,3): self.assertIn('PAGE={}/3'.format(p),output.getvalue())

    def test_video_report_rejects_unfinished_frame_before_any_flag_or_board_action(self):
        directory=self.root/'video'; directory.mkdir()
        (directory/'report.json').write_text('{"status":"failed","results":[]}')
        from types import SimpleNamespace
        with patch.dict(sys.modules,{'host_video':SimpleNamespace(dependencies=lambda: None)}), \
             patch('builtins.input',side_effect=[str(directory),'0','bad box']), \
             patch('video_report.flag_frame') as flag,patch('host_control.subprocess.run') as run:
            with self.assertRaises(ValueError): host.video_pages()
        flag.assert_not_called(); run.assert_not_called()

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

    def test_performance_menu_only_reuses_saved_data(self):
        directory=self.root/'saved-video'
        with patch.object(host,'select_video_run',return_value=directory), \
             patch('performance_report.analyze',return_value={'fixture':True}) as analyze, \
             patch('performance_report.render',side_effect=lambda r,p:'PERFORMANCE PAGE={}/3\n'.format(p)), \
             patch('builtins.input',side_effect=['','','','']), \
             patch('host_control.subprocess.run') as run,patch('sys.stdout',new_callable=io.StringIO) as output:
            host.performance_pages()
        analyze.assert_called_once_with(directory,0.)
        run.assert_not_called()
        self.assertIn('PAGE=3/3',output.getvalue())

    def test_accuracy_menu_reuses_candidates_without_board_or_golden(self):
        directory=self.root/'saved-video'
        with patch.object(host,'select_video_run',return_value=directory), \
             patch('accuracy_review.analyze',return_value={'frames':[]}) as analyze, \
             patch('builtins.input',side_effect=['','','n']), \
             patch('host_control.subprocess.run') as run,patch('host_control.subprocess.Popen') as popen:
            host.accuracy_pages()
        self.assertEqual(analyze.call_args.args[0],directory)
        self.assertEqual(analyze.call_args.args[2:],(None,None))
        run.assert_not_called(); popen.assert_not_called()

    def test_cleanup_menu_uses_packaged_script(self):
        with patch('host_control.subprocess.run') as run:
            host.cleanup()
        self.assertEqual(run.call_args.args[0], [sys.executable, str(self.root/'tools/cleanup_host.py'), '--apply'])
        self.assertEqual(run.call_args.kwargs['cwd'], self.root)


if __name__=='__main__': unittest.main()
