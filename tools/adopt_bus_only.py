"""Stop the selected running job, or skip an ended job, then show saved BUS results.

Standalone Python 3.8+ bootstrap: this file can run from /tmp before git pull.
It never reads or writes the board and never treats a derived report as proof.
"""
import argparse
import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def inside(path, base):
    path = Path(path).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError:
        raise ValueError('Path escapes allowed directory: ' + str(path)) from None
    return path


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def selected_job(repo):
    menu = inside(repo / 'yolo26_pc/outputs/host-menu', repo)
    base = inside(repo / 'yolo26_pc/outputs/full-check', repo)
    full = read_json(inside(menu / 'last-full.json', menu))
    job = read_json(inside(menu / 'last-launch.json', menu))
    run = inside(full['directory'], base)
    if run == base or inside(job['result'], base) != run:
        raise ValueError('last-full and last-launch do not identify the same full-check run')
    if inside(job['log'], run) != run / 'terminal.log':
        raise ValueError('Unexpected full-check log path')
    pid = job['pid']
    if type(pid) is not int or pid <= 1:
        raise ValueError('Invalid saved full-check PID')
    return run, pid


def process_identity(repo, run, pid, proc_root=Path('/proc')):
    """Return start time, or None for a terminated process; reject PID reuse."""
    directory = proc_root / str(pid)
    try:
        owner = directory.stat().st_uid
        stat = (directory / 'stat').read_text(encoding='utf-8')
        fields = stat[stat.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        command = (directory / 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return None
    if owner != os.getuid():
        raise ValueError('Saved job PID belongs to a different UID; refusing to signal')
    argv = [os.fsdecode(part) for part in command if part]
    script = str((repo / 'tools/full_check.py').resolve())
    script_index = 2 if len(argv) > 1 and argv[1] == '-u' else 1
    if (len(argv) <= script_index or not Path(argv[0]).name.startswith('python') or
            argv[script_index] != script or argv.count('--output') != 1):
        raise ValueError('Saved PID is not this repository full_check.py process')
    position = argv.index('--output')
    if position <= script_index or position + 1 >= len(argv) or Path(argv[position + 1]).resolve() != run:
        raise ValueError('Saved PID output directory differs from the selected run')
    try:
        if os.getpgid(pid) != pid or os.getsid(pid) != pid:
            raise ValueError('Saved full-check PID is not its independent process/session leader')
    except ProcessLookupError:
        return None
    if len(fields) <= 19:
        raise ValueError('Cannot verify saved PID start time')
    return fields[19]


def stop_old(repo, run, pid, expected, seconds, poll):
    current = process_identity(repo, run, pid)
    if current is None:
        print('Old job already ended; no signal needed.', flush=True)
        return
    if expected is None or current != expected:
        raise ValueError('Saved PID identity changed; refusing to signal')
    print('Requesting old job shutdown with SIGINT (PID {}).'.format(pid), flush=True)
    try:
        os.killpg(pid, signal.SIGINT)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + seconds
    while process_identity(repo, run, pid) is not None:
        if time.monotonic() >= deadline:
            raise TimeoutError('Old job did not exit after SIGINT; no force kill or git update was performed')
        time.sleep(min(poll, max(0, deadline - time.monotonic())))


@contextmanager
def update_locks(paths, seconds, poll):
    """Wait for both locks, then retain them for the complete git update."""
    import fcntl
    deadline = time.monotonic() + seconds
    streams = []
    while True:
        try:
            for path in paths:
                stream = path.open('a')
                streams.append(stream)
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            for stream in reversed(streams):
                stream.close()
            streams = []
            if time.monotonic() >= deadline:
                raise TimeoutError('Controller/PCIe lock is still held; repository was not updated')
            time.sleep(min(poll, max(0, deadline - time.monotonic())))
        except BaseException:
            for stream in reversed(streams):
                stream.close()
            raise
    try:
        yield
    finally:
        for stream in reversed(streams):
            stream.close()


def require_single_image(repo):
    """Never restart an old paired executable if pull did not install the fix."""
    tree = ast.parse((repo / 'tools/full_check.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BUS_ONLY_VERSION' for t in node.targets):
            if ast.literal_eval(node.value) == 2:
                return
    raise ValueError('Updated full_check.py is not the BUS-only release; nothing was started')


def run(args):
    repo = args.repo.resolve()
    if not (repo / 'tools/full_check.py').is_file() or not (repo / '.git').exists():
        raise ValueError('--repo must name the YOLO Git working directory')
    run_directory, pid = selected_job(repo)
    expected = process_identity(repo, run_directory, pid)
    stop_old(repo, run_directory, pid, expected, args.stop_timeout, args.poll_seconds)
    locks = (inside(repo / 'yolo26_pc/outputs/host-menu/control.lock', repo), Path('/tmp/yolo26-pcie.lock'))
    with update_locks(locks, args.stop_timeout, args.poll_seconds):
        print('Both locks released. Updating repository with git pull --ff-only.', flush=True)
        subprocess.run(['git', 'pull', '--ff-only'], cwd=repo, check=True)
        require_single_image(repo)
    print('Reading saved BUS results; no new inference is started.', flush=True)
    subprocess.run([sys.executable, str(repo / 'tools/host_control.py'), 'full-report'], cwd=repo, check=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--stop-timeout', type=float, default=120,
                        help='Maximum wait for graceful shutdown and released locks')
    parser.add_argument('--poll-seconds', type=float, default=5)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run this standalone bootstrap on the Linux host')
    if min(args.stop_timeout, args.poll_seconds) <= 0:
        parser.error('Waits and polling interval must be positive')
    try:
        return run(args)
    except (OSError, ValueError, KeyError, TimeoutError, subprocess.CalledProcessError) as exc:
        print('SWITCH FAILED: ' + str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
