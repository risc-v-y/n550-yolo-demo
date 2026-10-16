"""BUS-only bootstrap safety tests: no physical board or real Git update."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch
import adopt_bus_only as switch


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name).resolve()
        (self.repo / '.git').mkdir()
        (self.repo / 'tools').mkdir()
        (self.repo / 'tools/full_check.py').write_text('BUS_ONLY_VERSION = 2\n')
        self.menu = self.repo / 'yolo26_pc/outputs/host-menu'
        self.menu.mkdir(parents=True)
        self.run = self.repo / 'yolo26_pc/outputs/full-check/run-old'
        self.run.mkdir(parents=True)
        self.pid = 12345
        (self.menu / 'last-full.json').write_text(json.dumps({'directory': str(self.run)}))
        (self.menu / 'last-launch.json').write_text(json.dumps({
            'pid': self.pid, 'result': str(self.run), 'log': str(self.run / 'terminal.log')}))
        self.args = argparse.Namespace(repo=self.repo, wait_seconds=1, stop_timeout=1, poll_seconds=.01)
        self.capture = self.run / 'bus/attempt-1'
        (self.capture / 'tensors').mkdir(parents=True)
        self.raw = b'12345678' * 900
        (self.capture / 'tensors/0548.bin').write_bytes(self.raw)
        (self.capture / 'normal-candidates.bin').write_bytes(self.raw)
        self.summary = {'status': 'completed', 'sample': 'bus', 'evidence_version': 1,
                        'nodes_captured': 334, 'tensors_captured': 335,
                        'normal_repeat': {'byte_identical_to_capture': True,
                                          'sha256': hashlib.sha256(self.raw).hexdigest()}}
        self.write_summary()

    def write_summary(self):
        (self.capture / 'summary.json').write_text(json.dumps(self.summary))

    def test_paths_must_stay_in_selected_repository_full_check(self):
        self.assertEqual(switch.selected_job(self.repo), (self.run, self.pid))
        for directory in (self.repo / 'outside', self.repo / 'yolo26_pc/outputs/full-check'):
            (self.menu / 'last-full.json').write_text(json.dumps({'directory': str(directory)}))
            with self.assertRaises(ValueError):
                switch.selected_job(self.repo)

    def test_pid_reuse_or_changed_identity_is_never_signalled(self):
        with patch.object(switch, 'process_identity', return_value='other-start'), patch.object(switch.os, 'killpg', create=True) as kill:
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                switch.stop_old(self.repo, self.run, self.pid, 'old-start', 1, .01)
            kill.assert_not_called()

    def test_already_ended_job_needs_no_signal_and_locks_release_before_launch(self):
        events = []

        @contextmanager
        def locks(*args):
            events.append('lock')
            yield
            events.append('unlock')

        def command(argv, **kwargs):
            events.append('pull' if argv[0] == 'git' else 'report')
            if argv[0] != 'git':
                self.assertEqual(argv[-1], 'full-report')
                self.assertNotIn('full-check', argv)

        with patch.object(switch, 'process_identity', return_value=None), \
             patch.object(switch.os, 'killpg', create=True) as kill, \
             patch.object(switch, 'update_locks', side_effect=locks), \
             patch.object(switch.subprocess, 'run', side_effect=command):
            self.assertEqual(switch.run(self.args), 0)
            kill.assert_not_called()
        self.assertEqual(events, ['lock', 'pull', 'unlock', 'report'])

    def test_incomplete_bus_is_stopped_without_waiting_for_capture(self):
        (self.capture / 'summary.json').unlink()
        @contextmanager
        def locks(*args):
            yield
        with patch.object(switch, 'process_identity', side_effect=['start', 'start', None]), \
             patch.object(switch.os, 'killpg', create=True) as kill, \
             patch.object(switch, 'update_locks', side_effect=locks), \
             patch.object(switch.subprocess, 'run') as command:
            self.assertEqual(switch.run(self.args), 0)
            kill.assert_called_once_with(self.pid, signal.SIGINT)
            self.assertEqual(command.call_args.args[0][-1], 'full-report')
            self.assertEqual(switch.selected_job(self.repo), (self.run, self.pid))

    def test_complete_bus_requests_only_sigint_then_waits_for_exit(self):
        with patch.object(switch, 'process_identity', side_effect=['start', 'start', None]), \
             patch.object(switch.os, 'killpg', create=True) as kill, patch.object(switch.time, 'sleep'):
            switch.stop_old(self.repo, self.run, self.pid, 'start', 1, .01)
            kill.assert_called_once_with(self.pid, signal.SIGINT)

    def test_shutdown_timeout_does_not_force_kill_or_update(self):
        with patch.object(switch, 'process_identity', return_value='start'), \
             patch.object(switch.time, 'monotonic', side_effect=[0, 2]), \
             patch.object(switch.os, 'killpg', create=True) as kill, \
             patch.object(switch.subprocess, 'run') as command:
            with self.assertRaisesRegex(TimeoutError, 'no force kill'):
                switch.stop_old(self.repo, self.run, self.pid, 'start', 1, .01)
            kill.assert_called_once_with(self.pid, signal.SIGINT)
            command.assert_not_called()

    def test_blocked_locks_never_allow_git_pull(self):
        with patch.object(switch, 'process_identity', return_value=None), \
             patch.object(switch, 'update_locks', side_effect=TimeoutError('still held')), \
             patch.object(switch.subprocess, 'run') as command:
            with self.assertRaises(TimeoutError):
                switch.run(self.args)
            command.assert_not_called()

    def test_failed_pull_never_launches_and_paired_release_is_not_restarted(self):
        @contextmanager
        def locks(*args):
            yield

        (self.repo / 'tools/full_check.py').write_text("SAMPLES = ('bus', 'zidane')\n")
        with patch.object(switch, 'process_identity', return_value=None), \
             patch.object(switch, 'update_locks', side_effect=locks), \
             patch.object(switch.subprocess, 'run') as command:
            with self.assertRaisesRegex(ValueError, 'not the BUS-only release'):
                switch.run(self.args)
            self.assertEqual(command.call_count, 1)
            self.assertEqual(command.call_args.args[0], ['git', 'pull', '--ff-only'])

    @unittest.skipUnless(sys.platform == 'linux', 'Uses Linux flock')
    def test_real_linux_locks_are_exclusive_and_failed_attempt_releases_first(self):
        import fcntl
        first, second = self.repo / 'control.lock', self.repo / 'pcie.lock'
        with second.open('a') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(TimeoutError, 'lock is still held'):
                with switch.update_locks((first, second), .01, .005):
                    self.fail('Acquired an already held lock')
            with first.open('a') as available:
                fcntl.flock(available, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with switch.update_locks((first, second), .1, .005):
            for path in (first, second):
                with path.open('a') as other:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with switch.update_locks((first, second), .1, .005):
            pass

    @unittest.skipUnless(sys.platform == 'linux', 'Uses Linux process/session checks')
    def test_proc_owner_command_output_session_and_start_time(self):
        proc = self.repo / 'proc'
        directory = proc / str(self.pid)
        directory.mkdir(parents=True)
        stat = ['S'] + ['0'] * 18 + ['123456'] + ['0'] * 3
        (directory / 'stat').write_text('{} (python3) {}'.format(self.pid, ' '.join(stat)))
        command = ['python3', '-u', str(self.repo / 'tools/full_check.py'), '--output', str(self.run)]
        (directory / 'cmdline').write_bytes(b'\0'.join(os.fsencode(a) for a in command) + b'\0')
        with patch.object(switch.os, 'getpgid', return_value=self.pid), \
             patch.object(switch.os, 'getsid', return_value=self.pid):
            self.assertEqual(switch.process_identity(self.repo, self.run, self.pid, proc), '123456')
            for index, value in ((2, str(self.repo / 'tools/other.py')), (4, str(self.repo / 'elsewhere'))):
                changed = list(command)
                changed[index] = value
                (directory / 'cmdline').write_bytes(b'\0'.join(os.fsencode(a) for a in changed) + b'\0')
                with self.assertRaises(ValueError):
                    switch.process_identity(self.repo, self.run, self.pid, proc)
            (directory / 'cmdline').write_bytes(b'\0'.join(os.fsencode(a) for a in command) + b'\0')
            with patch.object(switch.os, 'getsid', return_value=1):
                with self.assertRaisesRegex(ValueError, 'session leader'):
                    switch.process_identity(self.repo, self.run, self.pid, proc)
            with patch.object(switch.os, 'getuid', return_value=os.getuid() + 1):
                with self.assertRaisesRegex(ValueError, 'different UID'):
                    switch.process_identity(self.repo, self.run, self.pid, proc)


if __name__ == '__main__':
    unittest.main()
