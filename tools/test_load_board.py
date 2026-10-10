"""Deployment ordering and failure isolation using fake host tools, no hardware."""
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='yolo-load-test-')
        self.root = Path(self.temp.name) / 'repo with spaces'
        self.reset = Path(self.temp.name) / 's2c tools'
        (self.root / 'tools').mkdir(parents=True)
        (self.root / 'firmware').mkdir()
        self.reset.mkdir()
        shutil.copyfile(Path(__file__).with_name('load_board.sh'), self.root / 'tools/load_board.sh')
        entries = []
        for name in ('demo-pcie.bin', 'selftest.bin'):
            blob = name.encode()
            (self.root / 'firmware' / name).write_bytes(blob)
            entries.append(f'{hashlib.sha256(blob).hexdigest()}  firmware/{name}\n')
        (self.root / 'firmware/SHA256SUMS').write_text(''.join(entries))
        for name, action in (('reset1.sh', 'hold'), ('reset.sh', 'release'), ('pbcopy', 'load')):
            path = self.reset / name
            path.write_text(f'''#!/usr/bin/env bash
echo "{action}" >> "$EVENTS"
if [[ "${{FAIL_AT:-}}" == {action} ]]; then echo 'injected failed' >&2; exit 7; fi
if [[ "${{FALSE_SUCCESS:-}}" == {action} ]]; then echo 'pbcopy failed!'; exit 0; fi
if [[ {action} == load ]]; then
    [[ "$1" == -d && "$2" == *:0x0 ]] || exit 8
    printf '%s\\n' "$2" > "$ARGUMENT"
else
    [[ -f ./reset1.sh && -f ./reset.sh ]] || exit 9
fi
''')
            path.chmod(0o755)
        self.env = dict(os.environ, EVENTS=str(Path(self.temp.name) / 'events'),
                        ARGUMENT=str(Path(self.temp.name) / 'argument'), TMPDIR=self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_load(self, *args, **env):
        return subprocess.run(['bash', str(self.root / 'tools/load_board.sh'),
                               '--reset-dir', str(self.reset), '--pbcopy', str(self.reset / 'pbcopy'),
                               *args], env=dict(self.env, **env), text=True, capture_output=True, timeout=10)

    def events(self):
        path = Path(self.env['EVENTS'])
        return path.read_text().splitlines() if path.exists() else []

    def test_success_and_log(self):
        result = self.run_load()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.events(), ['hold', 'load', 'release'])
        self.assertEqual(Path(self.env['ARGUMENT']).read_text().strip(),
                         f'{self.root}/firmware/demo-pcie.bin:0x0')
        logs = list((self.root / 'yolo26_pc/outputs').glob('deploy-*/load.log'))
        self.assertEqual(len(logs), 1)
        self.assertIn('Deployment completed', logs[0].read_text())

    def test_selftest_image(self):
        result = self.run_load('--selftest')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('selftest.bin:0x0', Path(self.env['ARGUMENT']).read_text())

    def test_hold_failure_never_loads(self):
        self.assertNotEqual(self.run_load(FAIL_AT='hold').returncode, 0)
        self.assertEqual(self.events(), ['hold'])

    def test_load_failure_keeps_reset(self):
        result = self.run_load(FAIL_AT='load')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.events(), ['hold', 'load'])
        self.assertIn('NOT called', result.stderr)

    def test_false_success_keeps_reset(self):
        self.assertNotEqual(self.run_load(FALSE_SUCCESS='load').returncode, 0)
        self.assertEqual(self.events(), ['hold', 'load'])

    def test_release_failure(self):
        result = self.run_load(FAIL_AT='release')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.events(), ['hold', 'load', 'release'])
        self.assertIn('uncertain', result.stderr)

    def test_bad_hash_does_not_touch_hardware(self):
        (self.root / 'firmware/demo-pcie.bin').write_bytes(b'changed')
        self.assertNotEqual(self.run_load().returncode, 0)
        self.assertEqual(self.events(), [])

    def test_missing_reset_script(self):
        (self.reset / 'reset.sh').unlink()
        self.assertNotEqual(self.run_load().returncode, 0)
        self.assertEqual(self.events(), [])

    def test_active_controller_blocks_deployment(self):
        with (Path(self.temp.name) / 'yolo26-pcie.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertNotEqual(self.run_load().returncode, 0)
            self.assertEqual(self.events(), [])


if __name__ == '__main__':
    unittest.main()
