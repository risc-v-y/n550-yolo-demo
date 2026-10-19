"""Remove rebuildable caches and unreferenced zero-frame preflight failures.

Python 3.8 stdlib only. No board access. --apply is explicit; default is preview.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
OLD_BUILDS = ('accel-model amu-kernel amu-semantics basic basic-v2 board-sync-test '
              'cleanup-tests clp-qemu-tests clp-selftest conv conv-v2 host-protocol-tests '
              'int8-kernel memory-tests quant-policy rvv-amu scalar-model '
              'elf_attributes.txt environment.txt golden_cases.h golden_manifest.json '
              'matmul.disasm matmul.elf matmul.map qemu-vlen128.log qemu-vlen256.log '
              'qemu-vlen512.log sha256.txt').split()
PREFLIGHT_FILES = {'report.json', 'diagnostics.jsonl', 'check-bus.png', 'check.mp4',
                   'host-check.json', 'video-summary.json', 'video-page-1.txt',
                   'video-page-2.txt', 'video-page-3.txt'}


def within(path, parent):
    try: path.relative_to(parent); return True
    except ValueError: return False


def validate_path(path, root):
    if path == root or not within(path, root): raise ValueError('Target leaves repository')
    for parent in (path, *path.parents):
        if parent == root: break
        if parent.is_symlink(): raise ValueError('Symbolic link refused')
    if not within(path.resolve(), root): raise ValueError('Resolved target leaves repository')


def walk_error(error): raise error


def snapshot(path, root):
    """Refuse traversal through links or special files; verify before deletion."""
    validate_path(path, root)
    entries = [path]
    if path.is_dir():
        for current, dirs, files in os.walk(str(path), followlinks=False, onerror=walk_error):
            entries.extend(Path(current)/name for name in dirs+files)
    signature, size, files = [], 0, 0
    for item in sorted(entries):
        value = item.lstat()
        if not (stat.S_ISDIR(value.st_mode) or stat.S_ISREG(value.st_mode)):
            raise ValueError('Link or special file refused')
        signature.append((str(item.relative_to(root)), value.st_dev, value.st_ino,
                          value.st_mode, value.st_size, value.st_mtime_ns))
        if stat.S_ISREG(value.st_mode): size += value.st_size; files += 1
    digest = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
    return dict(bytes=size, files=files, fingerprint=digest)


def json_files(outputs, links):
    if not outputs.exists(): return
    pending = [outputs]
    while pending:
        with os.scandir(str(pending.pop())) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink():
                    links.append((path, path.resolve()))
                elif entry.is_dir(follow_symlinks=False):
                    if entry.name != 'cleanup': pending.append(path)
                elif entry.name.endswith('.json') and entry.is_file(follow_symlinks=False):
                    yield path


def references(root):
    """Incoming references, including pointers and moved Windows evidence paths."""
    outputs = root/'yolo26_pc/outputs'
    result, unreadable = [], []
    def strings(value, key=''):
        if isinstance(value, str): yield key, value
        elif isinstance(value, dict):
            for name, item in value.items(): yield from strings(item, name)
        elif isinstance(value, list):
            for item in value: yield from strings(item, key)
    for origin in json_files(outputs, result):
        try: value = json.loads(origin.read_text(encoding='utf-8'))
        except (OSError, ValueError): unreadable.append(str(origin.relative_to(root))); continue
        for key, text in strings(value):
            if not text or '\x00' in text or '\n' in text or len(text) > 4096: continue
            if not any(c in text for c in '/\\.') and key not in (
                    'directory', 'path', 'file', 'source', 'result', 'log', 'input', 'output'): continue
            text = text.replace('\\', '/')
            choices = [root/text, origin.parent/text, outputs/text]
            for marker in ('yolo26_pc/outputs/', 'yolo26_riscv/build/'):
                if marker in text: choices.append(root/(marker+text.split(marker, 1)[1]))
            for choice in choices:
                try:
                    # Metadata needs lexical normalization, not repeated disk
                    # lookups for every tensor hash/module name. Actual links
                    # are inspected once during the directory scan above.
                    target = Path(os.path.abspath(str(choice)))
                    relevant = within(target, root)
                except (OSError, ValueError): continue
                if relevant: result.append((origin, target))
    return result, unreadable


def preflight_only(path):
    try:
        report = json.loads((path/'report.json').read_text(encoding='utf-8'))
        if (report.get('status') != 'failed' or report.get('stage') != 'host_check' or
                report.get('results') != [] or any(report.get(k) for k in
                    ('board', 'last_board_status', 'last_board_diagnostic'))): return False
        if any(p.name not in PREFLIGHT_FILES or not p.is_file() or p.is_symlink() for p in path.iterdir()):
            return False
        log = path/'diagnostics.jsonl'
        if log.exists():
            for line in log.read_text(encoding='utf-8').splitlines():
                event = json.loads(line)
                if event.get('type') != 'host_error': return False
        return True
    except (OSError, ValueError, TypeError, AttributeError): return False


def runtime_paths(root):
    paths = [Path(sys.executable).resolve(), Path(sys.prefix).resolve()]
    for config in (root/'yolo26_pc').glob('.venv*/pyvenv.cfg'):
        for line in config.read_text(encoding='utf-8').splitlines():
            if line.startswith('home = '): paths.append(Path(line[7:]).resolve())
    return paths


def make_plan(root, tracked=None):
    root = root.resolve()
    validate_path(root/'yolo26_pc/outputs', root)
    if tracked is None:
        command = ['git', '-c', 'safe.directory='+str(root), 'ls-files', '-z']
        raw = subprocess.run(command, cwd=root, check=True, stdout=subprocess.PIPE).stdout
        tracked = [root/os.fsdecode(name) for name in raw.split(b'\0') if name]
    targets = [root/'yolo26_pc/.cache/uv']
    targets += [root/'yolo26_riscv/build'/name for name in OLD_BUILDS]
    for base in ('tools', 'yolo26_pc', 'yolo26_riscv'):
        if not (root/base).exists(): continue
        for current, dirs, _ in os.walk(str(root/base), followlinks=False, onerror=walk_error):
            for name in dirs:
                if name == '__pycache__': targets.append(Path(current)/name)
            dirs[:] = [name for name in dirs if not name.startswith('.') and name not in
                       ('__pycache__', 'outputs', 'build', 'models', 'samples', 'Ultralytics')
                       and not (Path(current)/name).is_symlink()]
    outputs = root/'yolo26_pc/outputs'
    targets += [p for p in (outputs/'host-video').glob('*') if p.is_dir() and preflight_only(p)]
    incoming, unreadable = references(root)
    runtime = runtime_paths(root)
    planned, skipped = [], []
    for path in sorted(set(targets)):
        if not path.exists() and not path.is_symlink(): continue
        relative = str(path.relative_to(root))
        try:
            facts = snapshot(path, root)
            if any(within(Path(t), path) for t in tracked): raise ValueError('Contains Git-tracked files')
            if any(within(p, path) for p in runtime): raise ValueError('Python runtime depends on this path')
            if unreadable and within(path, outputs): raise ValueError('Some result JSON is unreadable; keep runs')
            owners = [str(origin.relative_to(root)) for origin, target in incoming
                      if not within(origin, path) and (within(target, path) or
                          target not in (root, outputs) and within(path, target))]
            if owners: raise ValueError('Referenced by '+owners[0])
            planned.append(dict(path=relative, **facts))
        except (OSError, ValueError) as exc: skipped.append(dict(path=relative, reason=str(exc)))
    return dict(version=1, root=str(root), candidates=planned, skipped=skipped,
                unreadable_json=unreadable, deleted=[], status='preview')


def save_audit(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def apply_plan(root, report):
    root = root.resolve()
    if report['root'] != str(root): raise ValueError('Cleanup root changed')
    # Replan before any deletion: pointers/new evidence may have appeared.
    current = {item['path']: item for item in make_plan(root)['candidates']}
    for item in report['candidates']:
        if current.get(item['path']) != item: raise ValueError('Cleanup target changed: '+item['path'])
    directory = root/'yolo26_pc/outputs/cleanup'
    validate_path(directory, root)
    directory.mkdir(parents=True, exist_ok=True)
    audit = directory/('host-cleanup-'+uuid.uuid4().hex[:12]+'.json')
    report['status'] = 'running'; save_audit(audit, report)
    try:
        for item in report['candidates']:
            path = root/item['path']
            if snapshot(path, root)['fingerprint'] != item['fingerprint']:
                raise ValueError('Cleanup target changed: '+item['path'])
            if path.is_dir(): shutil.rmtree(str(path))
            else: path.unlink()
            report['deleted'].append(item); save_audit(audit, report)
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed', error=str(exc)); raise
    finally: save_audit(audit, report); print('Cleanup audit:', audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Delete approved categories and save audit')
    args = parser.parse_args()
    if sys.platform != 'linux': parser.error('Run on the Linux host')
    from host_control import guard, workspace, board_lock
    validate_path(ROOT/'yolo26_pc/outputs/host-menu', ROOT)
    with guard(workspace()/'control.lock'), guard(board_lock()):
        print('HOST CLEANUP: scanning disposable files and result references...', flush=True)
        report = make_plan(ROOT)
        total = sum(item['bytes'] for item in report['candidates'])
        print('HOST CLEANUP: {} targets, {:.2f} MiB logical size'.format(len(report['candidates']), total/1048576))
        for item in report['candidates'][:12]: print('DELETE:', item['path'])
        print('Protected/skipped={}, unreadable JSON={}'.format(len(report['skipped']), len(report['unreadable_json'])))
        if args.apply:
            apply_plan(ROOT, report)
            print('HOST CLEANUP COMPLETED: {} targets'.format(len(report['deleted'])))
        else: print('PREVIEW ONLY; use --apply or host menu 20 to delete.')
    return 0


if __name__ == '__main__':
    try: sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print('HOST CLEANUP FAILED:', exc, file=sys.stderr); sys.exit(1)
