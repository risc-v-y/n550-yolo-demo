"""Destructive cleanup fixtures: containment, evidence, references and races."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cleanup_host as app


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        (self.root/'yolo26_pc').mkdir()
        self.outputs = self.root/'yolo26_pc/outputs'
        self.outputs.mkdir()

    def put(self, name, text='temporary'):
        path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8'); return path

    def model_run(self,name,version,status='passed'):
        base='yolo26_pc/outputs/clp-model-validation/'+name
        self.put(base+'/report.json',json.dumps(dict(status=status,profile_version=version,
                 qemu_exit=0,profile_sha256='0'*64)))
        self.put(base+'/tensors.bin')
        return self.root/base

    def test_only_latest_successful_optimization_check_is_kept(self):
        old=self.model_run('run-old',6); latest=self.model_run('run-latest',7)
        failed=self.model_run('run-failed',7,'failed')
        plan=app.make_plan(self.root,tracked=[])
        names={str(self.root/item['path']) for item in plan['candidates']}
        self.assertIn(str(old),names)
        self.assertNotIn(str(latest),names); self.assertNotIn(str(failed),names)

    def test_latest_pointer_works_with_backdated_clock(self):
        first=self.model_run('run-first',7); second=self.model_run('run-second',7)
        self.put('yolo26_pc/outputs/clp-model-validation/latest.json',
                 json.dumps(dict(directory=str(first))))
        self.assertEqual(app.superseded_model_runs(self.root),[second])
        self.put('yolo26_pc/outputs/evidence.json',json.dumps(dict(source=str(second/'tensors.bin'))))
        self.assertFalse(app.make_plan(self.root,tracked=[])['candidates'])

    def run_dir(self, name, **fields):
        report = dict(status='failed', stage='host_check', results=[], **fields)
        self.put('yolo26_pc/outputs/host-video/'+name+'/report.json', json.dumps(report))
        return self.outputs/'host-video'/name

    def paths(self, report): return {item['path'].replace('\\', '/') for item in report['candidates']}

    def test_deletes_only_rebuildable_files_and_preflight_failures(self):
        cache = self.put('tools/__pycache__/x.pyc')
        old = self.put('yolo26_riscv/build/accel-model/model.elf')
        uv = self.put('yolo26_pc/.cache/uv/archive/wheel')
        protected = [self.put(p) for p in ('firmware/demo-pcie.bin', 'model/constants.bin',
                     'yolo26_riscv/build/n550-board/demo.elf', 'yolo26_riscv/build/host-verify/auditor',
                     'yolo26_pc/.venv/lib/__pycache__/needed.pyc', 'yolo26_pc/models/yolo26n.pt',
                     'yolo26_pc/outputs/host-video/passed/source-000000.png',
                     'yolo26_pc/outputs/full-check/bus/tensors/0001.bin')]
        empty = self.run_dir('preflight')
        board = self.run_dir('startup', last_board_status={'state':4})
        frame = self.run_dir('partial'); self.put(str(frame.relative_to(self.root))+'/candidates-000000.bin')
        plan = app.make_plan(self.root)
        self.assertIn('yolo26_pc/outputs/host-video/preflight', self.paths(plan))
        audit = app.apply_plan(self.root, plan)
        self.assertEqual(json.loads(audit.read_text())['status'], 'completed')
        for path in (cache, old, uv, empty): self.assertFalse(path.exists())
        for path in protected+[board, frame]: self.assertTrue(path.exists())
        self.assertEqual(app.make_plan(self.root)['candidates'], [])

    def test_pointers_cross_run_dependencies_and_moved_windows_paths_are_preserved(self):
        for name in ('latest', 'referenced', 'windows'): self.run_dir(name)
        self.put('yolo26_pc/outputs/host-video/latest.json', json.dumps({'directory':str(self.outputs/'host-video/latest')}))
        self.put('yolo26_pc/outputs/full-check/analysis.json', json.dumps({'path':'yolo26_pc/outputs/host-video/referenced/report.json'}))
        self.put('yolo26_pc/outputs/full-check/other.json', json.dumps({'directory':'D:\\old\\YOLO移植\\yolo26_pc\\outputs\\host-video\\windows'}))
        self.assertFalse(app.make_plan(self.root)['candidates'])

    def test_unknown_broken_or_board_failure_runs_are_not_disposable(self):
        self.run_dir('bad-log'); self.put('yolo26_pc/outputs/host-video/bad-log/diagnostics.jsonl', '{')
        self.run_dir('board-log'); self.put('yolo26_pc/outputs/host-video/board-log/diagnostics.jsonl', '{"type":"pcie_tool"}')
        self.put('yolo26_pc/outputs/host-video/legacy/report.json', '{"status":"failed","results":[]}')
        self.run_dir('other')
        self.put('yolo26_pc/outputs/capture/broken.json', '{')
        self.assertFalse(app.make_plan(self.root)['candidates'])

    def test_tracked_files_runtime_and_concurrent_modification_stop_deletion(self):
        tracked = self.put('tools/__pycache__/tracked.pyc')
        subprocess.run(['git', '-C', str(self.root), 'add', '.'], check=True)
        uv = self.put('yolo26_pc/.cache/uv/python/interpreter')
        self.put('yolo26_pc/.venv/pyvenv.cfg', 'home = '+str(uv.parent))
        self.assertFalse(app.make_plan(self.root)['candidates'])
        old = self.put('yolo26_riscv/build/basic-v2/data.bin')
        plan = app.make_plan(self.root)
        old.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed'): app.apply_plan(self.root, plan)
        self.assertTrue(tracked.exists() and uv.exists() and old.exists())

    def test_new_reference_added_after_preview_prevents_deletion(self):
        run = self.run_dir('planned')
        plan = app.make_plan(self.root)
        self.put('yolo26_pc/outputs/host-menu/last.json', json.dumps({'directory':str(run)}))
        with self.assertRaisesRegex(ValueError, 'changed'): app.apply_plan(self.root, plan)
        self.assertTrue(run.exists())

    def test_nonpath_strings_and_relative_pointers(self):
        run = self.run_dir('relative')
        self.put('yolo26_pc/outputs/host-video/latest.json', json.dumps({'directory':'relative'}))
        self.put('yolo26_pc/outputs/metadata.json', json.dumps({'output':'x'*1000, 'hash':'a'*2000}))
        self.assertFalse(app.make_plan(self.root)['candidates'])
        self.assertTrue(run.exists())

    def test_failed_partial_deletion_is_audited_and_valid_results_survive(self):
        self.put('tools/__pycache__/temporary.pyc')
        protected = self.put('yolo26_pc/outputs/host-video/valid/source-000000.png')
        plan = app.make_plan(self.root)
        with patch.object(app.shutil, 'rmtree', side_effect=OSError('injected deletion error')):
            with self.assertRaises(OSError): app.apply_plan(self.root, plan)
        audit = next((self.outputs/'cleanup').glob('host-cleanup-*.json'))
        self.assertEqual(json.loads(audit.read_text())['status'], 'failed')
        self.assertTrue(protected.exists())

    @unittest.skipUnless(sys.platform == 'linux', 'Linux symlinks and locks')
    def test_symlink_targets_and_active_controller_are_refused(self):
        outside = self.root/'outside'; outside.mkdir(); (outside/'keep').write_text('keep')
        link = self.root/'yolo26_pc/.cache/uv'; link.parent.mkdir(parents=True)
        link.symlink_to(outside, target_is_directory=True)
        self.assertFalse(app.make_plan(self.root)['candidates'])
        self.assertTrue((outside/'keep').exists())
        cache = self.put('tools/__pycache__/x.pyc')
        plan = app.make_plan(self.root)
        (self.outputs/'cleanup').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'link'): app.apply_plan(self.root, plan)
        self.assertTrue(cache.exists())
        self.assertEqual(sorted(p.name for p in outside.iterdir()), ['keep'])
        import host_control as host
        with patch.object(host, 'ROOT', self.root), patch.object(app, 'ROOT', self.root), \
             patch.object(host, 'board_lock', return_value=self.root/'board.lock'), \
             patch.object(sys, 'argv', ['cleanup_host.py', '--apply']):
            for lock in (host.workspace()/'control.lock', host.board_lock()):
                with host.guard(lock), patch.object(app, 'make_plan') as scan:
                    with self.assertRaises(RuntimeError): app.main()
                    scan.assert_not_called()

    @unittest.skipUnless(sys.platform == 'linux', 'Linux symlink fixture')
    def test_file_and_directory_symlink_consumers_protect_old_builds(self):
        old = self.put('yolo26_riscv/build/conv/input.bin')
        consumer = self.outputs/'saved'; consumer.mkdir()
        (consumer/'input.bin').symlink_to(old)
        self.assertFalse(app.make_plan(self.root)['candidates'])
        (consumer/'input.bin').unlink()
        (consumer/'linked-directory').symlink_to(old.parent, target_is_directory=True)
        self.assertFalse(app.make_plan(self.root)['candidates'])
        (consumer/'linked-directory').unlink()
        (consumer/'linked-builds').symlink_to(old.parent.parent, target_is_directory=True)
        self.assertFalse(app.make_plan(self.root)['candidates'])


if __name__ == '__main__': unittest.main()
