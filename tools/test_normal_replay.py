"""Normal-mode input identity, no-reset failure handling and strict golden gate."""
from contextlib import ExitStack
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import normal_replay as app
import host_control as host
from board_profile import decode_profile
from test_pack_probe import payload as probe_payload
from test_performance_report import payload as profile_payload


class NormalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.probe = self.root / 'old-probe'; self.probe.mkdir()
        self.output = self.root / 'normal'; self.output.mkdir()
        self.data = bytes(app.INPUT_BYTES)
        self.actual = struct.pack('<6f', 1, 2, 3, 4, .5, 0) * 300
        raw = probe_payload(stage=16, column=32, inner=27, next_column=32)
        (self.probe / 'source-input.bin').write_bytes(self.data)
        (self.probe / 'probe-000.bin').write_bytes(raw)
        self.firmware = dict(sha256=dict(bin='image', elf='symbols'))
        self.saved = dict(id=self.probe.name, status='completed', firmware=self.firmware,
                          input=dict(sha256=app.sha(self.data), crc32=zlib.crc32(self.data)),
                          samples=[dict(file='probe-000.bin', probe=app.pack_probe.decode(raw, 7, 1))])
        app.save_json(self.probe / 'report.json', self.saved)
        self.args = SimpleNamespace(core_mhz=40, pbcopy='copy', pbload='read', tool_timeout=30,
                                    infer_timeout=120, load=False, reset_dir='reset')
        self.flags = dict(reads=[], writes=[], closed=0, golden=0)
        owner = self
        status = [app.pack_probe.PCIE_MAGIC, 1, app.DONE, 0, 7, 1, 2, 7200, 0, 0, 2, 0, 0, 0]
        body = app.STATUS.pack(*status)[:60]
        self.state = body + struct.pack('<I', zlib.crc32(body))

        class Link:
            def read(self, address, size, deadline):
                owner.flags['reads'].append(address)
                return {app.BASE: b'prefix', app.MAILBOX+192: owner.state,
                        app.MAILBOX+128: struct.pack('<16I', 2, *([0]*15))}[address]
            def close(self): owner.flags['closed'] += 1
        self.link = Link()

        class Backend:
            session = 7
            last_diagnostic = dict(node=333, phase_name='node_end', error=0)
            def __init__(self, **kwargs):
                owner.flags['options'] = {k: kwargs[k] for k in ('trace', 'check_finite', 'capture')}
                self.link = kwargs['transport']
                self.last_timing = dict(pcie_frame=1, board_profile=decode_profile(profile_payload(1, 7, version=9), 1, 7),
                                        upload_seconds=1, run_roundtrip_seconds=2, download_seconds=1, profile_read_seconds=1)
            def infer_bytes(self, data):
                owner.flags['writes'].append(data)
                return owner.actual
            def _snapshot(self, deadline): owner.flags['snapshot'] = True
            def close(self): self.link.close()
        self.Backend = Backend

        class Golden:
            def __init__(self, *args, **kwargs): pass
            def check(self, frame, data, actual):
                owner.flags['golden'] += 1
                owner.flags['golden_input'] = data
                return dict(status='EXACT', different_elements=0)
            def close(self): pass
        self.Golden = Golden

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(app.pack_probe.diag, 'firmware_info', return_value=(self.firmware, {}, b'prefix')))
        stack.enter_context(patch.object(app, 'PBTools', return_value=self.link))
        stack.enter_context(patch.object(app, 'PCIeBackend', self.Backend))
        stack.enter_context(patch.object(app, 'VideoGolden', self.Golden))
        stack.enter_context(patch.object(app, 'timing_files', return_value=[]))
        return stack

    def report(self): return json.loads((self.output / 'report.json').read_bytes())

    def test_one_normal_frame_uses_identical_input_and_strict_golden(self):
        with self.patches():
            self.assertEqual(app.execute(self.args, self.output, self.probe), 0)
        report = self.report()
        self.assertEqual(report['status'], 'passed')
        self.assertTrue(report['normal_profile_verified'])
        self.assertEqual(self.flags['writes'], [self.data])
        self.assertEqual(self.flags['golden_input'], self.data)
        self.assertEqual(self.flags['options'], dict(trace=False, check_finite=False, capture=False))
        self.assertEqual(self.flags['closed'], 1)
        self.assertEqual((self.output / 'candidates.bin').read_bytes(), self.actual)
        page = app.render(report)
        self.assertIn('PASSED', page)
        self.assertLessEqual(len(page.splitlines()), 22)
        self.assertTrue(all(len(line) <= 80 for line in page.splitlines()))

    def test_corrupt_input_snapshot_or_firmware_stops_before_board_access(self):
        valid_raw = (self.probe / 'probe-000.bin').read_bytes()
        for kind in ('input', 'snapshot', 'firmware'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                data = self.data if kind != 'input' else b'\1' + self.data[1:]
                (self.probe / 'source-input.bin').write_bytes(data)
                self.saved['firmware'] = self.firmware if kind != 'firmware' else dict(sha256={})
                app.save_json(self.probe / 'report.json', self.saved)
                (self.probe / 'probe-000.bin').write_bytes(bytes(256) if kind == 'snapshot' else valid_raw)
                with self.patches(), patch.object(app, 'PBTools') as connect:
                    self.assertEqual(app.execute(self.args, out, self.probe), 1)
                    connect.assert_not_called()

    def test_busy_board_never_opens_or_overwrites_input(self):
        values = list(app.STATUS.unpack(self.state)); values[2] = app.pack_probe.BUSY
        body = app.STATUS.pack(*values)[:60]
        self.state = body + struct.pack('<I', zlib.crc32(body))
        with self.patches(): self.assertEqual(app.execute(self.args, self.output, self.probe), 1)
        self.assertEqual(self.flags['writes'], [])
        self.assertEqual(self.flags['closed'], 1)
        self.assertIn('no OPEN/RUN', self.report()['error'])

    def test_reference_preflight_failure_does_not_touch_board(self):
        with self.patches(), patch.object(app, 'VideoGolden', side_effect=ValueError('auditor checksum')), \
             patch.object(app, 'PBTools') as connect:
            self.assertEqual(app.execute(self.args, self.output, self.probe), 1)
            connect.assert_not_called()

    def test_new_firmware_reuses_original_input_and_loads_exactly_once(self):
        self.args.load = True
        self.saved['firmware'] = dict(sha256=dict(bin='a'*64, elf='b'*64))
        app.save_json(self.probe / 'report.json', self.saved)
        with self.patches(), patch.object(app.subprocess, 'run') as load:
            self.assertEqual(app.execute(self.args, self.output, self.probe), 0)
        load.assert_called_once()
        self.assertEqual(Path(load.call_args.args[0][1]).name, 'load_board.sh')
        self.assertEqual(self.flags['writes'], [self.data])
        self.assertTrue(self.report()['source']['firmware_changed'])
        self.assertEqual(self.report()['timing']['board_profile']['version'], 9)
        self.assertEqual(self.flags['closed'], 2)

    def test_load_failure_never_submits_a_frame_or_retries(self):
        self.args.load = True
        with self.patches(), patch.object(app.subprocess, 'run', side_effect=app.subprocess.CalledProcessError(1, ['load'])) as load:
            self.assertEqual(app.execute(self.args, self.output, self.probe), 1)
        load.assert_called_once()
        self.assertEqual(self.flags['writes'], [])
        self.assertEqual(self.flags['golden'], 0)
        self.assertEqual(self.flags['closed'], 1)

    def test_old_profile_is_offline_compatible_but_not_current_firmware(self):
        timing = dict(pcie_frame=1, board_profile=decode_profile(profile_payload(1, 7, version=7), 1, 7))
        self.assertEqual(app.normal_profile(timing, 7)['version'], 7)
        with self.assertRaises(ValueError): app.normal_profile(timing, 7, expected_version=8)

    def test_timeout_keeps_scene_without_retry_and_golden_is_not_run(self):
        def timeout(backend, data):
            self.flags['writes'].append(data)
            backend.last_diagnostic = dict(node=0, phase_name='amu_pack', error=0)
            raise TimeoutError('first-block stall')
        with self.patches(), patch.object(self.Backend, 'infer_bytes', timeout):
            self.assertEqual(app.execute(self.args, self.output, self.probe), 1)
        self.assertEqual(self.flags['writes'], [self.data])
        self.assertTrue(self.flags['snapshot'])
        self.assertEqual(self.flags['golden'], 0)
        self.assertEqual(self.report()['last_board_diagnostic']['node'], 0)
        self.assertFalse((self.output / 'candidates.bin').exists())

    def test_numeric_difference_and_diagnostic_profile_cannot_pass(self):
        with self.patches(), patch.object(self.Golden, 'check', return_value=dict(status='DIFFERENT', different_elements=1)):
            self.assertEqual(app.execute(self.args, self.output, self.probe), 1)
        self.assertEqual(self.report()['golden']['different_elements'], 1)
        for options in (2, 4, 8):
            timing = dict(pcie_frame=1, board_profile=decode_profile(profile_payload(1, 7, options=options, version=7), 1, 7))
            with self.assertRaisesRegex(ValueError, 'disabled'): app.normal_profile(timing, 7)
        with self.assertRaisesRegex(ValueError, 'disabled'): app.normal_profile(self.report()['timing'], 99)

    def test_offline_resume_never_touches_board_and_rejects_damaged_candidates(self):
        with self.patches(), patch.object(self.Golden, 'check', side_effect=KeyboardInterrupt):
            self.assertEqual(app.execute(self.args, self.output, self.probe), 1)
        original = (self.output / 'report.json').read_bytes()
        with self.patches(), patch.object(app, 'PBTools') as link, patch.object(app, 'PCIeBackend') as backend:
            self.assertEqual(app.reanalyze(self.output), 0)
            link.assert_not_called(); backend.assert_not_called()
        self.assertEqual((self.output / 'report.json').read_bytes(), original)
        (self.output / 'candidates.bin').write_bytes(b'\1' + self.actual[1:])
        with self.patches(): self.assertEqual(app.reanalyze(self.output), 1)
        self.assertIn('SHA256', json.loads((self.output / 'analysis.json').read_bytes())['error'])

    def test_full_success_and_failure_reports_fit_one_page(self):
        with self.patches(): self.assertEqual(app.execute(self.args, self.output, self.probe), 0)
        report = self.report()
        report['slowest'] = [dict(node=i, op='conv', total=12345) for i in range(3)]
        for failed in (False, True):
            if failed:
                report.update(status='failed', error='x' * 200, snapshot_error='y' * 200)
            page = app.render(report)
            self.assertLessEqual(len(page.splitlines()), 22)
            self.assertTrue(all(len(line) <= 80 for line in page.splitlines()))

    def test_node_csv_covers_all_334_nodes_with_operator_names(self):
        profile = decode_profile(profile_payload(1, 7, version=7), 1, 7)
        slow = app.timing_files(self.output, profile)
        import csv
        with (self.output / 'node-timing.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row['node']) for row in rows], list(range(334)))
        self.assertEqual(rows[0]['op'], 'conv')
        self.assertEqual(len(slow), 3)

    def test_menu_runs_normal_check_without_loader_or_probe(self):
        for answers, action in ((['8', '6', '0', '0'], None), ([], 'normal-replay')):
            with self.subTest(action=action), patch.object(sys, 'argv', ['host_control.py'] + ([action] if action else [])), \
                 patch.object(sys, 'platform', 'linux'), patch('builtins.input', side_effect=answers), \
                 patch.object(host.subprocess, 'run') as run:
                self.assertEqual(host.main(), 0)
                command = run.call_args.args[0]
                self.assertEqual(Path(command[1]).name, 'normal_replay.py')
                self.assertEqual(command[2:], ['--load'])


if __name__ == '__main__':
    unittest.main()
