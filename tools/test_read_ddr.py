"""Exercise the DDR reader with a fake PCIe tool; no board or license required."""
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import read_ddr


class FormatTests(unittest.TestCase):
    def test_little_endian_halfwords_and_cpu_addresses(self):
        data = struct.pack('<8H', 0xc200, 0xbc00, 0x3c00, 0x4200, 0xc000, 0, 0x4000, 0xc200)
        self.assertEqual(list(read_ddr.hex_lines(data, 0x80003a40, 16)),
                         ['0x80003a40: c200 bc00 3c00 4200 c000 0000 4000 c200'])

    def test_32_bit_words_and_multiple_rows(self):
        data = struct.pack('<5I', 0x42f60000, 0x40000000, 0x42f60000, 0x42f60000, 0)
        self.assertEqual(list(read_ddr.hex_lines(data, 0x80003440, 32)),
                         ['0x80003440: 42f60000 40000000 42f60000 42f60000', '0x80003450: 00000000'])


@unittest.skipUnless(sys.platform == 'linux', 'PCIe tools and flock run on Linux')
class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ddr-read-test-')
        self.base = Path(self.temp.name)
        self.cwd = self.base / 'licensed working directory'
        self.cwd.mkdir()
        self.root = self.base / 'repo'
        self.root.mkdir()
        self.root_patch = patch.object(read_ddr, 'ROOT', self.root)
        self.root_patch.start()
        self.out = self.base / 'result'
        self.data = struct.pack('<8H', 0xc200, 0xbc00, 0x3c00, 0x4200, 0xc000, 0, 0x4000, 0xc200)
        self.mode = 'ok'
        self.pbload = self.base / 'fake pbload'
        self.pbload.write_text('''#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import time
config = json.loads(Path('fixture.json').read_text())
with Path('events.jsonl').open('a') as output:
    output.write(json.dumps(sys.argv[1:]) + '\\n')
assert sys.argv[1] == '-d'
destination, address, size = sys.argv[2].split(':')
assert Path(destination).name == destination and len(destination) <= 13
assert int(address, 16) == config['offset']
assert int(size, 16) == len(bytes.fromhex(config['data']))
print('pbload started', flush=True)
if config['mode'] == 'timeout':
    Path(destination).write_bytes(b'partial')
    time.sleep(5)
if config['mode'] == 'exit_failure':
    Path(destination).write_bytes(b'partial')
    print('platform driver failed')
    sys.exit(7)
if config['mode'] == 'false_success':
    print('[x] ERROR: Connect to device failed')
    sys.exit(0)
if config['mode'] == 'missing':
    Path(destination).unlink()
    sys.exit(0)
data = bytes.fromhex(config['data'])
Path(destination).write_bytes(data[:-1] if config['mode'] == 'short' else data)
''')
        self.pbload.chmod(0o755)

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def run_read(self, *extra, address='0x80003a40', size=None, width='16', output=True):
        config = {'mode': self.mode, 'data': self.data.hex(), 'offset': int(address, 0) - read_ddr.diag.BASE}
        (self.cwd / 'fixture.json').write_text(json.dumps(config))
        args = [address, str(len(self.data)) if size is None else size, '--width', width,
                '--pbload', str(self.pbload), '--tool-cwd', str(self.cwd), *extra]
        if output:
            args += ['--output', str(self.out)]
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            return read_ddr.main(args)

    def report(self):
        return json.loads((self.out / 'report.json').read_text())

    def events(self):
        path = self.cwd / 'events.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_read_only_success_saves_exact_data_and_offset(self):
        self.assertEqual(self.run_read(address='0x8000_3a40', size='0x10'), 0)
        report = self.report()
        self.assertEqual(report['status'], 'read')
        self.assertEqual(report['pcie_offset'], '0x3a40')
        self.assertEqual(report['received_bytes'], 16)
        self.assertFalse(report['all_zero'])
        self.assertEqual(report['sha256'], hashlib.sha256(self.data).hexdigest())
        self.assertEqual((self.out / '001-ddr.bin').read_bytes(), self.data)
        self.assertIn('c200 bc00 3c00 4200', self.stdout.getvalue())
        self.assertEqual((self.out / 'data.txt').read_text().strip(),
                         '0x80003a40: c200 bc00 3c00 4200 c000 0000 4000 c200')
        self.assertEqual(len(self.events()), 1)
        self.assertEqual(self.events()[0][0], '-d')
        self.assertTrue(self.events()[0][1].endswith(':0x3a40:0x10'))
        self.assertEqual(list(self.cwd.glob('p*.bin')), [])

    def test_zero_data_is_reported_only_after_verified_read(self):
        self.data = bytes(16)
        self.assertEqual(self.run_read(width='32'), 0)
        self.assertTrue(self.report()['all_zero'])
        self.assertIn('All bytes zero: True', self.stdout.getvalue())

    def test_preview_limit_preserves_full_raw_data(self):
        self.data = bytes(300) + b'\x01\x00' + bytes(210)
        self.assertEqual(self.run_read('--max-print', '32'), 0)
        self.assertEqual(self.report()['preview_bytes'], 32)
        self.assertFalse(self.report()['all_zero'])
        self.assertEqual((self.out / '001-ddr.bin').read_bytes(), self.data)
        self.assertEqual(len((self.out / 'data.txt').read_text().splitlines()), 2)
        self.assertIn('first 32 of 512 bytes', self.stdout.getvalue())

    def test_long_result_path_never_reaches_platform_tool(self):
        self.out = self.base / ('long result directory ' * 8) / 'result'
        self.assertEqual(self.run_read(), 0)
        name = self.events()[0][1].split(':')[0]
        self.assertEqual(name, Path(name).name)
        self.assertLessEqual(len(name), 13)
        self.assertEqual(list(self.cwd.glob('p*.bin')), [])

    def test_failure_exit_and_exit_zero_failure_text(self):
        for mode in ('exit_failure', 'false_success'):
            with self.subTest(mode=mode):
                self.mode, self.out = mode, self.base / mode
                self.assertEqual(self.run_read(), 1)
                report = self.report()
                self.assertEqual(report['status'], 'failed')
                self.assertNotIn('all_zero', report)
                self.assertIn('failed', (self.out / '001-ddr.log').read_text())
                self.assertFalse((self.out / 'data.txt').exists())
                if mode == 'exit_failure':
                    self.assertEqual((self.out / '001-ddr.bin').read_bytes(), b'partial')

    def test_missing_or_short_read_is_not_displayed_as_valid(self):
        for mode in ('missing', 'short'):
            with self.subTest(mode=mode):
                self.mode, self.out = mode, self.base / mode
                self.assertEqual(self.run_read(), 1)
                self.assertIn('output length', self.report()['error'])
                self.assertNotIn('all_zero', self.report())
                self.assertNotIn('0x80003a40:', self.stdout.getvalue())

    def test_timeout_preserves_partial_data_and_logs(self):
        self.mode = 'timeout'
        started = time.monotonic()
        self.assertEqual(self.run_read('--tool-timeout', '.15'), 1)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(self.report()['exception'], 'TimeoutExpired')
        self.assertEqual((self.out / '001-ddr.bin').read_bytes(), b'partial')
        self.assertIn('pbload started', (self.out / '001-ddr.log').read_text())
        self.assertEqual(list(self.cwd.glob('p*.bin')), [])

    def test_link_owner_prevents_any_pcie_read(self):
        with read_ddr.diag.link_lock():
            self.assertEqual(self.run_read(), 1)
        self.assertEqual(self.events(), [])
        self.assertIn('owns the YOLO PCIe link', self.report()['error'])

    def test_invalid_ranges_and_formats_prevent_pcie_access(self):
        cases = [dict(address='0x7ffffffc'), dict(address='0x81fffff8'), dict(size='0'),
                 dict(size='-1'), dict(size='15'), dict(width='7')]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(SystemExit) as raised:
                self.run_read(**case)
            self.assertEqual(raised.exception.code, 2)
        self.assertEqual(self.events(), [])
        self.assertFalse(self.out.exists())

    def test_nonfinite_or_invalid_options_prevent_pcie_access(self):
        for extra in (('--tool-timeout', 'nan'), ('--tool-timeout', '-1'),
                      ('--max-print', '0'), ('--tool-cwd', str(self.base / 'missing'))):
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as raised:
                self.run_read(*extra)
            self.assertEqual(raised.exception.code, 2)
        self.assertEqual(self.events(), [])

    def test_existing_output_cannot_overwrite_previous_read(self):
        self.out.mkdir()
        previous = self.out / '001-ddr.bin'
        previous.write_bytes(b'keep previous data')
        with self.assertRaises(FileExistsError):
            self.run_read()
        self.assertEqual(previous.read_bytes(), b'keep previous data')
        self.assertEqual(self.events(), [])

    def test_backdated_clock_does_not_reuse_results(self):
        class FixedClock:
            @staticmethod
            def now(unused):
                return datetime(2022, 1, 1, tzinfo=timezone.utc)

        with patch.object(read_ddr, 'datetime', FixedClock):
            self.assertEqual(self.run_read(output=False), 0)
            self.assertEqual(self.run_read(output=False), 0)
        directories = list((self.root / 'yolo26_pc/outputs/ddr-read').iterdir())
        self.assertEqual(len(directories), 2)
        self.assertNotEqual(*directories)

    def test_interruption_keeps_failure_report(self):
        with patch.object(read_ddr.diag.Reader, 'read', side_effect=KeyboardInterrupt):
            self.assertEqual(self.run_read(), 130)
        self.assertEqual(self.report()['exception'], 'KeyboardInterrupt')


if __name__ == '__main__':
    unittest.main()
