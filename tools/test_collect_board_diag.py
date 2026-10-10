"""Diagnostic collection with shipped ELF files and a fake, read-only pbload."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import collect_board_diag as diag


class FirmwareTests(unittest.TestCase):
    def test_shipped_firmware_addresses_and_distinct_prefixes(self):
        prefixes = []
        for name in ('selftest', 'demo-pcie'):
            info, symbols, prefix = diag.firmware_info(diag.ROOT, name)
            self.assertEqual(set(symbols), set(diag.SYMBOLS))
            self.assertEqual(symbols['board_diag'] % 64, 0)
            self.assertEqual(len(info['sha256']['elf']), 64)
            prefixes.append(prefix)
        self.assertNotEqual(*prefixes)

    def test_malformed_or_wrong_architecture_elf(self):
        blob = bytearray((diag.ROOT / 'firmware/selftest.elf').read_bytes())
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            diag.elf_symbols(blob[:63])
        struct.pack_into('<H', blob, 18, 62)  # x86-64 is not an N550 image.
        with self.assertRaisesRegex(ValueError, 'RISC-V'):
            diag.elf_symbols(blob)

    def test_missing_symbol(self):
        blob = (diag.ROOT / 'firmware/selftest.elf').read_bytes()
        blob = blob.replace(b'board_test_stage\0', b'other_test_stage\0')
        with self.assertRaisesRegex(ValueError, 'Missing ELF symbols: board_test_stage'):
            diag.elf_symbols(blob)

    def test_ddr_read_bounds(self):
        self.assertEqual(diag.offset(diag.BASE, 256), 0)
        for address, size in ((diag.BASE - 1, 4), (diag.END - 4, 8), (diag.BASE, 0)):
            with self.assertRaises(ValueError):
                diag.offset(address, size)


@unittest.skipUnless(sys.platform == 'linux', 'PCIe tools and flock run on Linux')
class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='board-diag-test-')
        self.base = Path(self.temp.name)
        self.root = self.base / 'repo with spaces'
        firmware = self.root / 'firmware'
        firmware.mkdir(parents=True)
        entries = []
        for suffix in ('elf', 'bin'):
            relative = f'firmware/selftest.{suffix}'
            blob = (diag.ROOT / relative).read_bytes()
            (self.root / relative).write_bytes(blob)
            entries.append(f'{hashlib.sha256(blob).hexdigest()}  {relative}\n')
        (firmware / 'SHA256SUMS').write_text(''.join(entries))
        info, self.symbols, prefix = diag.firmware_info(self.root, 'selftest')
        values = {'board_uart_status': 0, 'board_uart_baud': 114942, 'board_uart_dlf_bits': 4,
                  'board_test_stage': 3, 'board_error': 0, 'board_trap_cause': 0,
                  'board_trap_pc': 0, 'board_trap_value': 0}
        snapshot = [0] * 102
        snapshot[0], snapshot[1], snapshot[3], snapshot[4], snapshot[16] = 1, 8, 2**64 - 1, 2**64 - 1, 2**64 - 1
        self.config = {'mode': 'ok', 'data': {'0': prefix.hex()}}
        for name, fmt in diag.SYMBOLS.items():
            blob = struct.pack(fmt, values[name]) if fmt else struct.pack('<102Q', *snapshot)
            self.config['data'][str(self.symbols[name] - diag.BASE)] = blob.hex()
        self.cwd = self.base / 's2c license working directory'
        self.cwd.mkdir()
        self.pbload = self.base / 'fake pbload'
        self.pbload.write_text('''#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import time
config = json.loads(Path('fixture.json').read_text())
with Path('events.jsonl').open('a') as log:
    log.write(json.dumps(sys.argv[1:]) + '\\n')
assert sys.argv[1] == '-d'
destination, address, size = sys.argv[2].split(':')
print('pbload started', flush=True)
if len(destination) > 13 or Path(destination).is_absolute():
    print('*** buffer overflow detected ***: terminated')
    sys.exit(134)
if config['mode'] == 'timeout':
    Path(destination).write_bytes(b'partial')
    time.sleep(5)
if config['mode'] == 'false_success':
    print('[x] ERROR: Connect to device failed')
    sys.exit(0)
if config['mode'] == 'exit_failure':
    Path(destination).write_bytes(b'partial')
    print('platform driver failed')
    sys.exit(7)
if config['mode'] == 'missing':
    Path(destination).unlink(missing_ok=True)
    sys.exit(0)
data = bytes.fromhex(config['data'][str(int(address, 16))])
assert len(data) == int(size, 16)
if config['mode'] == 'short':
    data = data[:-1]
Path(destination).write_bytes(data)
''')
        self.pbload.chmod(0o755)
        self.root_patch = patch.object(diag, 'ROOT', self.root)
        self.root_patch.start()
        self.serial_patch = patch.object(diag.serial_probe, 'discover', return_value=[])
        self.serial_patch.start()
        self.out = self.base / 'result'

    def tearDown(self):
        self.serial_patch.stop()
        self.root_patch.stop()
        self.temp.cleanup()

    def run_collect(self, *extra, output=True):
        (self.cwd / 'fixture.json').write_text(json.dumps(self.config))
        args = ['--firmware', 'selftest', '--pbload', str(self.pbload),
                '--tool-cwd', str(self.cwd), '--samples', '2', '--interval', '0', *extra]
        if output:
            args += ['--output', str(self.out)]
        with redirect_stdout(io.StringIO()):
            code = diag.main(args)
        return code

    def report(self):
        return json.loads((self.out / 'report.json').read_text())

    def events(self):
        path = self.cwd / 'events.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_read_only_success_and_address_conversion(self):
        load = self.base / 'load.log'
        load.write_text('Deployment completed\n')
        uart = self.base / 'terminal.log'
        uart.write_text('[YOLO] UART OK\n')
        self.assertEqual(self.run_collect('--load-log', str(load), '--uart-log', str(uart)), 0)
        report = self.report()
        self.assertEqual(report['status'], 'collected')
        self.assertEqual(len(report['snapshots']), 2)
        self.assertFalse(report['serial']['device_opened'])
        self.assertEqual((self.out / 'load.log').read_text(), load.read_text())
        self.assertEqual((self.out / 'uart.log').read_text(), uart.read_text())
        for item in report['snapshots']:
            self.assertEqual(item['uart_status_name'], 'ready')
            self.assertEqual(item['test_stage_name'], 'passed')
            self.assertEqual(item['diagnostic']['stage_name'], 'done')
        events = self.events()
        self.assertEqual(len(events), 1 + 2 * len(diag.SYMBOLS))
        self.assertTrue(all(event[0] == '-d' for event in events))
        self.assertTrue(events[0][1].endswith(':0x0:0x100'))
        wanted = f":0x{self.symbols['board_diag'] - diag.BASE:x}:0x330"
        self.assertTrue(any(event[1].endswith(wanted) for event in events))

    def test_long_archive_path_uses_unique_short_filenames(self):
        self.out = self.base / ('long archive path ' * 8) / 'result'
        self.assertEqual(self.run_collect(), 0)
        names = [event[1].split(':')[0] for event in self.events()]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(len(name) <= 13 and Path(name).name == name for name in names))
        self.assertEqual(list(self.cwd.glob('p*.bin')), [])
        self.assertEqual(len(list(self.out.glob('*.bin'))), len(names))
        self.assertEqual((self.out / '001-program-prefix.bin').stat().st_size, 256)

    def test_checksum_failure_prevents_any_pcie_access(self):
        (self.root / 'firmware/selftest.bin').write_bytes(b'wrong image')
        self.assertEqual(self.run_collect(), 1)
        self.assertEqual(self.events(), [])
        self.assertIn('checksum', self.report()['error'])

    def test_wrong_loaded_program_prevents_symbol_reads(self):
        self.config['data']['0'] = bytes(256).hex()
        self.assertEqual(self.run_collect(), 1)
        self.assertEqual(len(self.events()), 1)
        self.assertIn('program prefix', self.report()['error'])

    def test_exit_zero_failure_text_is_not_accepted(self):
        self.config['mode'] = 'false_success'
        self.assertEqual(self.run_collect(), 1)
        self.assertIn('failed', self.report()['error'])
        self.assertIn('Connect to device failed', (self.out / '001-program-prefix.log').read_text())

    def test_nonzero_exit_keeps_tool_log(self):
        self.config['mode'] = 'exit_failure'
        self.assertEqual(self.run_collect(), 1)
        self.assertIn('exit 7', self.report()['error'])
        self.assertIn('platform driver failed', (self.out / '001-program-prefix.log').read_text())
        self.assertEqual((self.out / '001-program-prefix.bin').read_bytes(), b'partial')
        self.assertEqual(list(self.cwd.glob('p*.bin')), [])

    def test_timeout_keeps_partial_log_and_report(self):
        self.config['mode'] = 'timeout'
        started = time.monotonic()
        self.assertEqual(self.run_collect('--tool-timeout', '.15'), 1)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(self.report()['exception'], 'TimeoutExpired')
        self.assertIn('pbload started', (self.out / '001-program-prefix.log').read_text())
        self.assertEqual((self.out / '001-program-prefix.bin').read_bytes(), b'partial')
        self.assertEqual(list(self.cwd.glob('p*.bin')), [])

    def test_missing_and_short_output_are_rejected(self):
        for mode in ('missing', 'short'):
            with self.subTest(mode=mode):
                self.out = self.base / mode
                self.config['mode'] = mode
                self.assertEqual(self.run_collect(), 1)
                self.assertIn('output length', self.report()['error'])

    def test_uninitialized_diagnostic_is_preserved_as_partial(self):
        self.config['data'][str(self.symbols['board_diag'] - diag.BASE)] = bytes(816).hex()
        self.assertEqual(self.run_collect(), 1)
        report = self.report()
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(len(report['snapshots']), 1)
        self.assertIn('version: 0', report['snapshots'][0]['read_errors']['board_diag'])
        self.assertEqual(len(list(self.out.glob('*board_diag.bin'))), 1)

    def test_board_error_is_collected_without_claiming_pass(self):
        self.config['data'][str(self.symbols['board_uart_status'] - diag.BASE)] = struct.pack('<i', -7).hex()
        snapshot = [0] * 102
        snapshot[0], snapshot[1], snapshot[6] = 1, 10, 0x100
        snapshot[20], snapshot[21] = 2, diag.BASE + 24
        self.config['data'][str(self.symbols['board_diag'] - diag.BASE)] = struct.pack('<102Q', *snapshot).hex()
        self.assertEqual(self.run_collect(), 0)
        report = self.report()
        self.assertEqual(report['status'], 'collected')
        item = report['snapshots'][0]
        self.assertEqual(item['uart_status_name'], 'tx_timeout')
        self.assertEqual(item['diagnostic']['stage_name'], 'trap')
        self.assertEqual(item['diagnostic']['trap_pc'], diag.BASE + 24)

    def test_active_controller_prevents_pcie_reads(self):
        with diag.link_lock():
            self.assertEqual(self.run_collect(), 1)
        self.assertEqual(self.events(), [])
        self.assertIn('owns the YOLO PCIe link', self.report()['error'])

    def test_interruption_preserves_report(self):
        with patch.object(diag.Reader, 'read', side_effect=KeyboardInterrupt):
            self.assertEqual(self.run_collect(), 130)
        self.assertEqual(self.report()['exception'], 'KeyboardInterrupt')

    def test_backdated_clock_does_not_reuse_run_directory(self):
        class FixedClock:
            @staticmethod
            def now(unused):
                return datetime(2022, 1, 1, tzinfo=timezone.utc)

        with patch.object(diag, 'datetime', FixedClock):
            self.assertEqual(self.run_collect('--samples', '1', output=False), 0)
            self.assertEqual(self.run_collect('--samples', '1', output=False), 0)
        directories = list((self.root / 'yolo26_pc/outputs/host-diagnostics').iterdir())
        self.assertEqual(len(directories), 2)
        self.assertNotEqual(*directories)


@unittest.skipUnless(sys.platform == 'linux', 'README upload commands use Bash')
class UploadTests(unittest.TestCase):
    def test_readme_upload_creates_then_appends_without_changing_source(self):
        readme = (diag.ROOT / 'README.md').read_text(encoding='utf-8')
        block = readme.split('```bash\nRUN_DIR=', 1)[1].split('```', 1)[0]
        lines = ('RUN_DIR=' + block).splitlines()
        lines[:2] = ['RUN_DIR="$UPLOAD_RUN_DIR"', 'UPLOAD_DIR="$UPLOAD_WORK_DIR"']
        script = '\n'.join(lines).replace('git@github.com:risc-v-y/n550-yolo-demo.git', '"$UPLOAD_REMOTE"')
        with tempfile.TemporaryDirectory(prefix='diag-upload-test-') as temporary:
            base = Path(temporary)
            remote, source = base / 'remote.git', base / 'source'
            source.mkdir()

            def git(directory, *args):
                result = subprocess.run(['git', '-C', str(directory), *args],
                                        text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout.strip()

            subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
            git(source, 'init', '-b', 'main')
            git(source, 'config', 'user.name', 'Fixture')
            git(source, 'config', 'user.email', 'fixture@example.invalid')
            (source / 'source.txt').write_text('source must remain unchanged\n')
            git(source, 'add', 'source.txt')
            git(source, 'commit', '-m', 'source')
            git(source, 'remote', 'add', 'origin', str(remote))
            git(source, 'push', 'origin', 'main')
            original = git(source, 'rev-parse', 'HEAD')
            for number in (1, 2):
                run = base / f'run-{number}'
                work = base / f'upload-{number}'
                run.mkdir()
                work.mkdir()
                (run / 'report.json').write_text('{"status":"collected"}\n')
                (run / 'tool.log').write_text('platform log\n')
                result = subprocess.run(['bash', '-c', script], cwd=source,
                                        env=dict(os.environ, UPLOAD_RUN_DIR=str(run),
                                                 UPLOAD_WORK_DIR=str(work), UPLOAD_REMOTE=str(remote)),
                                        text=True, capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(git(source, 'branch', '--show-current'), 'main')
                self.assertEqual(git(source, 'rev-parse', 'HEAD'), original)
                self.assertEqual(git(source, 'status', '--porcelain'), '')
            self.assertEqual(git(remote, 'rev-list', '--count', 'host-diagnostics'), '2')
            for number in (1, 2):
                self.assertEqual(git(remote, 'show', f'host-diagnostics:run-{number}/tool.log'), 'platform log')
            self.assertEqual(git(remote, 'rev-parse', 'main'), original)


if __name__ == '__main__':
    unittest.main()
