"""Export existing BUS evidence for offline analysis (Python 3.8 stdlib only).

No model replay, firmware load or board communication is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_ARCHIVE_BYTES = 3_000_000_000
REPORTS = ('bus-analysis.json', 'summary.json', 'checkpoint.json', 'bus/analysis.json')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def contained(path, parent):
    path = Path(path).resolve()
    try:
        path.relative_to(parent.resolve())
    except ValueError:
        raise ValueError('Evidence path leaves allowed directory: ' + str(path)) from None
    return path


def locate(report=None):
    """Use the analyzed BUS pointer, even when capture was reused elsewhere."""
    outputs = ROOT / 'yolo26_pc/outputs'
    if report is None:
        report = Path(read_json(outputs / 'host-menu/last-full.json')['directory'])
    report = contained(report, outputs)
    bus = None
    gate = None
    gate_file = report / 'tile-gate.json'
    if gate_file.is_file():
        gate = read_json(gate_file)
    for name in REPORTS:
        path = report / name
        if not path.is_file():
            continue
        value = read_json(path)
        result = value if name == 'bus/analysis.json' else value.get('samples', {}).get('bus', {})
        if bus is None and result.get('directory'):
            candidate = Path(result['directory'])
            bus = contained(candidate if candidate.is_absolute() else ROOT / candidate, outputs)
        if gate is None and value.get('tile_gate', {}).get('directory'):
            gate = value['tile_gate']
    if bus is None:
        from full_check import bus_capture
        bus = contained(bus_capture(report), outputs)
    if not gate or not gate.get('directory'):
        raise ValueError('Original tile-probe directory is missing from the report/tile-gate.json')
    probe_path = Path(gate['directory'])
    probe = contained(probe_path if probe_path.is_absolute() else ROOT / probe_path, outputs)
    return report, bus, probe, gate


def validate_capture(directory, manifest, prepared):
    """Check saved identity, every snapshot/input/output and ordinary repeat."""
    from full_evidence import identity, evidence, final_check
    summary = identity(directory, 'bus', manifest, prepared)
    nodes = manifest['nodes']
    total_outputs = sum(sum(tid >= 0 for tid in node['outputs']) for node in nodes)
    if (summary.get('status') != 'completed' or summary.get('nodes_captured') != len(nodes) or
            summary.get('tensors_captured') != total_outputs):
        raise ValueError('BUS capture is incomplete: ' + str(directory))
    session = None
    for node in nodes:
        snap, _, _, issues = evidence(directory, node, manifest['tensors'])
        key = (snap['session'], snap['frame'], snap['sequence'] - node['id'])
        if issues or session is not None and key != session:
            raise ValueError('BUS input/session not verified at node {}: {}'.format(node['id'], issues))
        session = key
    final = final_check(directory, prepared, summary, render=False)
    return {'nodes': len(nodes), 'outputs': total_outputs, 'verified_input_nodes': len(nodes),
            'normal_repeat_exact': final['normal_repeat_exact'],
            'capture_sha256': final['capture_sha256']}


def require_files(bus, probe, manifest):
    required = {bus / 'summary.json', bus / 'input.bin', bus / 'normal-candidates.bin',
                probe / 'summary.json', probe / 'probe-1.bin', probe / 'probe-2.bin'}
    for node in manifest['nodes']:
        snapshot = bus / 'snapshots/{:04d}.json'.format(node['id'])
        required.update((snapshot, snapshot.with_suffix('.bin')))
        required.update(bus / 'tensors/{:04d}.bin'.format(tid) for tid in node['outputs'] if tid >= 0)
        if snapshot.is_file():
            for proof in read_json(snapshot).get('inputs', []):
                if proof.get('path'):
                    required.add(contained(bus / proof['path'], bus))
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        raise ValueError('Missing {} required file(s):\n{}'.format(len(missing), '\n'.join(missing)))


def validate_probe(directory, gate):
    from tile_probe import saved_inputs
    original, source, raws = saved_inputs(directory)
    digest = hashlib.sha256(original).hexdigest()
    hashes = [hashlib.sha256(raw).hexdigest() for raw in raws]
    if (source.get('mode') != 'physical board' or
            gate.get('source_report_sha256') != digest or gate.get('raw_sha256') != hashes):
        raise ValueError('Tile-probe evidence differs from the saved gate: ' + str(directory))
    return {'source_report_sha256': digest, 'raw_sha256': hashes}


def inventory(report, bus, probe):
    entries = {}
    def add(path, name):
        if path.is_symlink():
            raise ValueError('Symlink is not exported: ' + str(path))
        if not path.is_file():
            raise ValueError('Not a regular evidence file: ' + str(path))
        entries[name] = path
    for base, prefix in ((bus, 'bus'), (probe, 'probe')):
        if not base.is_dir():
            raise ValueError('Evidence directory missing: ' + str(base))
        for path in sorted(base.rglob('*')):
            if path.is_symlink():
                raise ValueError('Symlink is not exported: ' + str(path))
            if path.is_file() and path.suffix.lower() != '.zip':
                add(contained(path, base), prefix + '/' + path.relative_to(base).as_posix())
    # Only parent reports/logs and BUS analysis; do not traverse Zidane data.
    for base in (report, report / 'bus'):
        if base.is_dir():
            for path in sorted(base.iterdir()):
                if path.suffix.lower() in ('.json', '.log', '.csv', '.png', '.svg'):
                    add(path, 'check/' + path.relative_to(report).as_posix())
    for name in ('model/manifest.json', 'reference/nodes/manifest.json',
                 'reference/prepared/manifest.json', 'firmware/host-verify.json', 'firmware/SHA256SUMS'):
        add(ROOT / name, 'metadata/' + name)
    return entries


def git_metadata():
    result = {}
    for key, command in (('commit', ['rev-parse', 'HEAD']),
                         ('tracked_changes', ['status', '--porcelain', '--untracked-files=no'])):
        proc = subprocess.run(['git', '-c', 'safe.directory=' + str(ROOT)] + command,
                              cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=20)
        result[key] = proc.stdout.strip() if proc.returncode == 0 else 'unavailable'
    return result


def make_archive(entries, manifest, directory):
    """Verify archived bytes against their manifest before publishing the ZIP."""
    directory.mkdir(parents=True, exist_ok=True)
    name = 'bus-evidence-' + secrets.token_hex(6)
    final = directory / (name + '.zip')
    temporary = directory / (name + '.zip.part')
    manifest_path = directory / (name + '.manifest.json')
    checksum_path = directory / (name + '.zip.sha256')
    manifest['files'] = [{'path': name, 'bytes': path.stat().st_size, 'sha256': digest_file(path)}
                         for name, path in sorted(entries.items())]
    manifest['source_bytes'] = sum(entry['bytes'] for entry in manifest['files'])
    print('PACK: {} files, {:.2f} MiB source; verifying ZIP before publication.'.format(
        len(entries), manifest['source_bytes'] / 2**20), flush=True)
    body = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
    published = False
    try:
        with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for name, path in sorted(entries.items()):
                archive.write(path, name)
            archive.writestr('manifest.json', body)
        if temporary.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError('Archive exceeds 3 GB; no complete archive was published')
        with zipfile.ZipFile(temporary) as archive:
            for entry in manifest['files']:
                digest = hashlib.sha256()
                with archive.open(entry['path']) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                if (archive.getinfo(entry['path']).file_size != entry['bytes'] or
                        digest.hexdigest() != entry['sha256']):
                    raise ValueError('File changed during export: ' + entry['path'])
            if archive.read('manifest.json') != body:
                raise ValueError('Archive manifest does not match')
        checksum = digest_file(temporary)
        manifest_path.write_bytes(body)
        checksum_path.write_text(checksum + '  ' + final.name + '\n', encoding='utf-8')
        temporary.replace(final)
        published = True
        return final, manifest_path, checksum
    finally:
        if not published:
            for path in (temporary, manifest_path, checksum_path):
                if path.exists():
                    path.unlink()


def export(report=None, output_dir=None):
    started = time.monotonic()
    report, bus, probe, gate = locate(report)
    print('CHECK:', report, '\nBUS:', bus, '\nPROBE:', probe, flush=True)
    entries = inventory(report, bus, probe)
    from node_capture import verify_reference_manifest
    manifest = read_json(ROOT / 'reference/nodes/manifest.json')
    verify_reference_manifest(manifest)
    require_files(bus, probe, manifest)
    prepared = read_json(ROOT / 'reference/prepared/manifest.json')['samples']['bus']
    capture = validate_capture(bus, manifest, prepared)
    physical_probe = validate_probe(probe, gate)
    metadata = {'version': 1, 'status': 'complete', 'sample': 'bus',
                'scope': 'File integrity/completeness only; numerical acceptance is unchanged',
                'origins': {'check': str(report), 'bus': str(bus), 'probe': str(probe)},
                'relocation': {'actual_bus': 'bus/', 'physical_probe': 'probe/', 'parent_reports': 'check/'},
                'capture': capture, 'probe': physical_probe, 'git': git_metadata()}
    output_dir = output_dir or ROOT / 'yolo26_pc/outputs/evidence-export'
    for source in (report, bus, probe):
        try:
            output_dir.resolve().relative_to(source.resolve())
        except ValueError:
            continue
        raise ValueError('Output directory must be outside source evidence directories')
    archive, listing, digest = make_archive(entries, metadata, output_dir)
    print('EXPORT COMPLETED: BUS evidence verified; no board operations.', flush=True)
    print('ZIP:', archive, flush=True)
    print('SIZE: {:.2f} MiB; {} files'.format(archive.stat().st_size / 2**20, len(entries)), flush=True)
    print('SHA256:', digest, '\nMANIFEST:', listing, flush=True)
    print('Send the ZIP; its manifest is included. Elapsed: {:.1f}s'.format(time.monotonic() - started), flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Full-check directory; default: last-full.json')
    parser.add_argument('--output-dir', type=Path, help='Destination; default: outputs/evidence-export/')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run export on the Linux host')
    try:
        from host_control import guard, workspace
        with guard(workspace() / 'control.lock'):
            export(args.report, args.output_dir)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, EOFError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        print('EXPORT FAILED: ' + str(exc), file=sys.stderr, flush=True)
        print('Source evidence was preserved; no board action was taken.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
