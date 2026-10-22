"""Deployment ordering and failure isolation using fake host tools, no hardware."""
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import unittest


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='yolo-load-test-')
        self.root = Path(self.temp.name) / 'repo with spaces'
        self.reset = Path(self.temp.name) / 's2c tools'
        (self.root / 'tools').mkdir(parents=True)
        (self.root / 'firmware').mkdir()
        self.reset.mkdir()
        self.cwd = Path(self.temp.name) / 'license working directory'
        self.cwd.mkdir()
        shutil.copyfile(Path(__file__).with_name('load_board.sh'), self.root / 'tools/load_board.sh')
        entries = []
        for name in ('demo-pcie.bin', 'clp-selftest.bin'):
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
    # Emulate the vendor filename limit, and require the original license cwd.
    [[ "$2" != */* && ${{#2}} -le 32 && "$PWD" == "$EXPECTED_CWD" ]] || exit 10
    printf '%s\\n' "$2" > "$ARGUMENT"
    cat -- "${{2%:0x0}}" > "$UPLOADED"
    if [[ "${{WAIT_LOAD:-}}" == 1 ]]; then sleep 30; fi
else
    [[ -f ./reset1.sh && -f ./reset.sh ]] || exit 9
fi
''')
            path.chmod(0o755)
        self.env = dict(os.environ, EVENTS=str(Path(self.temp.name) / 'events'),
                        ARGUMENT=str(Path(self.temp.name) / 'argument'), TMPDIR=self.temp.name,
                        EXPECTED_CWD=str(self.cwd), UPLOADED=str(Path(self.temp.name) / 'uploaded'))

    def tearDown(self):
        leftovers = list(self.cwd.glob('p*.bin'))
        self.temp.cleanup()
        self.assertEqual(leftovers, [], 'Temporary upload must be removed on every exit')

    def run_load(self, *args, **env):
        return subprocess.run(['bash', str(self.root / 'tools/load_board.sh'),
                               '--reset-dir', str(self.reset), '--pbcopy', str(self.reset / 'pbcopy'),
                               *args], cwd=self.cwd, env=dict(self.env, **env),
                              text=True, capture_output=True, timeout=10)

    def events(self):
        path = Path(self.env['EVENTS'])
        return path.read_text().splitlines() if path.exists() else []

    def test_success_and_log(self):
        result = self.run_load()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.events(), ['hold', 'load', 'release'])
        self.assertRegex(Path(self.env['ARGUMENT']).read_text().strip(), r'^p[A-Za-z0-9]{8}\.bin:0x0$')
        self.assertEqual(Path(self.env['UPLOADED']).read_bytes(), b'demo-pcie.bin')
        logs = list((self.root / 'yolo26_pc/outputs').glob('deploy-*/load.log'))
        self.assertEqual(len(logs), 1)
        self.assertIn('Deployment completed', logs[0].read_text())


    def test_clp_selftest_image_and_order(self):
        result = self.run_load('--clp-selftest')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.events(), ['hold', 'load', 'release'])
        self.assertEqual(Path(self.env['UPLOADED']).read_bytes(), b'clp-selftest.bin')
        self.assertIn('PCIe offset 0x70000000', result.stdout)

    def test_conflicting_images_do_not_touch_hardware(self):
        self.assertEqual(self.run_load('--clp-selftest', '--clp-selftest').returncode, 2)
        self.assertEqual(self.events(), [])


    def test_retired_images_are_rejected_before_board_access(self):
        for flag in ('--selftest','--tile-probe'):
            self.assertEqual(self.run_load(flag).returncode,2)
        self.assertEqual(self.events(),[])

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

    def test_long_explicit_baseline_path(self):
        folder = self.root / 'outputs' / ('long-run-' + 'x'*100) / ('baseline-' + 'y'*80)
        folder.mkdir(parents=True)
        source = folder / 'baseline.bin'
        blob = bytes(range(256))*4096
        source.write_bytes(blob)
        result = self.run_load('--image', str(source), '--sha256', hashlib.sha256(blob).hexdigest())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.events(), ['hold', 'load', 'release'])
        self.assertEqual(Path(self.env['UPLOADED']).read_bytes(), blob)
        self.assertEqual(source.read_bytes(), blob)

    def test_corrupted_copy_is_rejected_before_reset(self):
        fake_bin = Path(self.temp.name) / 'fake-bin'
        fake_bin.mkdir()
        copy = fake_bin / 'cp'
        copy.write_text('#!/usr/bin/env bash\nprintf corrupt > "${@: -1}"\n')
        copy.chmod(0o755)
        result = self.run_load(PATH=str(fake_bin)+os.pathsep+os.environ['PATH'])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.events(), [])

    def test_interruption_cleans_copy_without_releasing_reset(self):
        command = ['bash', str(self.root/'tools/load_board.sh'), '--reset-dir', str(self.reset),
                   '--pbcopy', str(self.reset/'pbcopy')]
        process = subprocess.Popen(command, cwd=self.cwd, env=dict(self.env, WAIT_LOAD='1'),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   start_new_session=True)
        try:
            deadline = time.monotonic()+5
            while not Path(self.env['UPLOADED']).exists() and process.poll() is None and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertTrue(Path(self.env['UPLOADED']).exists(), 'Fake upload did not start')
            os.killpg(process.pid, signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=5)
            self.assertNotEqual(process.returncode, 0, stdout+stderr)
            self.assertEqual(self.events(), ['hold', 'load'])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()


if __name__ == '__main__':
    unittest.main()
