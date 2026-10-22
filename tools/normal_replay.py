"""Replay one preserved probe input without trace; check golden and CPU cycles."""
import argparse
import csv
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import textwrap
import time
import zlib

import pack_probe
from pack_probe import ROOT, INPUT_BYTES
from board_diagnostics import DiagnosticLog, OPS
from board_profile import validate_saved_profile
from pcie_backend import BASE, MAILBOX, READY, DONE, STATUS, STATUS_KEYS, checked, PBTools, PCIeBackend
from video_golden import VideoGolden, sha, save_json


def source_input(directory, firmware, allow_firmware_change=False):
    """Check the saved input and completed probe before any board access."""
    raw = (directory / 'report.json').read_bytes()
    report = json.loads(raw)
    if report.get('status') != 'completed':
        raise ValueError('Selected probe did not complete; original evidence preserved')
    previous = report.get('firmware', {}).get('sha256', {})
    changed = previous != firmware['sha256']
    if changed:
        if not allow_firmware_change:
            raise ValueError('Probe/current firmware hashes differ; use --load for the current frozen model')
        if (set(previous) != {'bin', 'elf'} or any(not isinstance(v, str) or len(v) != 64 or
                any(c not in '0123456789abcdef' for c in v) for v in previous.values())):
            raise ValueError('Source probe firmware hashes are missing/corrupt')
    data = (directory / 'source-input.bin').read_bytes()
    if len(data) != INPUT_BYTES or sha(data) != report['input']['sha256']:
        raise ValueError('Preserved input length/SHA256 mismatch')
    if 'crc32' in report['input'] and zlib.crc32(data) != report['input']['crc32']:
        raise ValueError('Preserved input differs from its original RUN CRC')
    if not all(math.isfinite(v) for (v,) in struct.iter_unpack('<f', data)):
        raise ValueError('Preserved input contains NaN/Inf')
    last = report['samples'][-1]
    filename = last['file']
    if Path(filename).name != filename or filename in ('.', '..'):
        raise ValueError('Invalid saved probe filename')
    probe_raw = (directory / filename).read_bytes()
    p = last['probe']
    probe = pack_probe.decode(probe_raw, p['session'], p['frame'])
    if (probe is None or sha(probe_raw) != p['sha256'] or probe['stage'] != 16 or
            probe['error'] or (probe['columns'], probe['next_column'], probe['k']) != (32, 32, 27)):
        raise ValueError('Missing/corrupt completed first-block evidence')
    return data, dict(directory=str(directory), run=report['id'], report_sha256=sha(raw),
                      input_sha256=sha(data), original=report['input'], firmware_sha256=previous,
                      firmware_changed=changed)


def normal_profile(timing, session, expected_version=None):
    profile = validate_saved_profile(timing['board_profile'], timing['pcie_frame'])
    if (profile['session'] != session or profile['version'] not in (7, 8, 9) or profile['options'] != 0 or
            (expected_version is not None and profile['version'] != expected_version)):
        raise ValueError('Expected verified normal profile with all diagnostic options disabled, version {}'.format(expected_version))
    return profile


def timing_files(directory, profile):
    manifest = json.loads((ROOT / 'reference/nodes/manifest.json').read_text(encoding='utf-8'))
    names = {node['id']: OPS[node['op']] for node in manifest['nodes']}
    fields = ('node', 'op', 'total', 'cache', 'compute', 'check')
    rows = [dict((key, row[key]) for key in fields if key != 'op') for row in profile['nodes_detail']]
    for row in rows:
        row['op'] = names[row['node']]
    with (directory / 'node-timing.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return sorted(rows, key=lambda row: -row['total'])[:3]


def render(report):
    lines = ['NORMAL REPLAY: ' + report['status'].upper(), 'Run: ' + report['id'],
             'Probe source: ' + report.get('source', {}).get('run', 'not verified'),
             'Mode: normal, options=0; one input; no automatic frame retry.',
             'Frame: ' + report.get('frame_status', 'not submitted')]
    golden = report.get('golden', {})
    lines.append('Golden: {} | different={}/1800 (no tolerance)'.format(
        golden.get('status', 'NOT_RUN'), golden.get('different_elements', 'unknown')))
    timing = report.get('timing', {})
    profile = timing.get('board_profile') if report.get('normal_profile_verified') else None
    if profile:
        lines.append('Profile: v{}, 334/334 nodes; diagnostic options verified OFF.'.format(profile['version']))
        for key in ('model_cycles', 'service_cycles'):
            mhz = report.get('core_mhz', 0)
            seconds = ' = {:.6f}s @ {}MHz'.format(profile[key] / (mhz * 1e6), mhz) if mhz else ''
            lines.append('{}: {}{}'.format(key, profile[key], seconds))
        lines.append('input_pack={} bias_dequant={}'.format(
            profile['phases']['input_pack']['ticks'], profile['phases']['bias_dequant']['ticks']))
    if 'run_roundtrip_seconds' in timing:
        lines.append('run_roundtrip_seconds: {:.4f}'.format(timing['run_roundtrip_seconds']))
        lines.append('upload/download/profile: {:.4f}/{:.4f}/{:.4f}s'.format(
            timing['upload_seconds'], timing['download_seconds'], timing.get('profile_read_seconds', 0)))
    if 'inference_seconds' in report:
        lines.append('PCIe frame wall: {:.4f}s (excludes OPEN/golden/graphics).'.format(report['inference_seconds']))
    for row in report.get('slowest', []):
        lines.append('Slow node {} {}: {} cycles'.format(row['node'], row['op'], row['total']))
    d = report.get('last_board_diagnostic')
    if d:
        lines.append('Board: node={} phase={} error=0x{:x}'.format(d['node'], d['phase_name'], d['error']))
    if report.get('error'):
        lines.append('Error: ' + report['error'])
    if report.get('snapshot_error'):
        lines.append('Snapshot: ' + report['snapshot_error'])
    lines.append('One input/run only; a pass does not establish the old stall cause.')
    lines.append('Files: report.json, normal-check.txt, node-timing.csv, golden-*.')
    return '\n'.join(textwrap.shorten(line, width=80, placeholder='...') for line in lines) + '\n'


def publish(directory, report, name='report.json'):
    save_json(directory / name, report)
    page = render(report)
    (directory / 'normal-check.txt').write_text(page, encoding='utf-8')
    print(page, end='', flush=True)
    print('Output:', directory, flush=True)


def execute(args, directory, probe_directory):
    report = dict(version=1, kind='normal_replay', id=directory.name, status='failed',
                  core_mhz=args.core_mhz, normal_profile_verified=False,
                  runner_sha256=sha(Path(__file__).read_bytes().replace(b'\r\n', b'\n')),
                  load_requested=getattr(args, 'load', False),
                  note='No reset/reload/frame retry after failure. Core may still be running.')
    log = DiagnosticLog(directory)
    engine = backend = link = None
    try:
        firmware, _, prefix = pack_probe.diag.firmware_info(ROOT, 'demo-pcie')
        data, source = source_input(probe_directory, firmware, allow_firmware_change=getattr(args, 'load', False))
        report.update(firmware=firmware, source=source, input_sha256=sha(data))
        (directory / 'source-input.bin').write_bytes(data)
        # Verify the independent reference before touching the board.
        engine = VideoGolden(directory, root=ROOT)
        report['stage'] = 'board_preflight'
        link = PBTools(args.pbcopy, args.pbload, timeout=args.tool_timeout, diagnostic=log.emit)
        deadline = time.monotonic() + args.tool_timeout * 3
        state = dict(zip(STATUS_KEYS, STATUS.unpack(checked(link.read(MAILBOX + 192, 64, deadline)))))
        doorbell = struct.unpack_from('<I', link.read(MAILBOX + 128, 64, deadline))[0]
        if (state['magic'], state['version']) != (pack_probe.PCIE_MAGIC, 1) or state['state'] not in (READY, DONE) or doorbell != state['sequence']:
            raise ValueError('Board not idle/ready; original state preserved, no OPEN/RUN performed')
        if getattr(args, 'load', False):
            link.close()
            link = None
            report['stage'] = 'load_current_firmware'
            save_json(directory / 'report.json', report)
            subprocess.run(['bash', str(ROOT/'tools/load_board.sh'), '--reset-dir', args.reset_dir,
                            '--pbcopy', args.pbcopy], cwd=ROOT, check=True)
            link = PBTools(args.pbcopy, args.pbload, timeout=args.tool_timeout, diagnostic=log.emit)
            deadline = time.monotonic() + args.tool_timeout
        if link.read(BASE, len(prefix), deadline) != prefix:
            raise ValueError('DDR program prefix differs; current firmware not accepted')
        owned, link = link, None  # PCIeBackend closes its transport on constructor failure.
        backend = PCIeBackend(transport=owned, infer_timeout=args.infer_timeout,
                              tool_timeout=args.tool_timeout, diagnostic=log.emit,
                              progress=lambda message: print(message, flush=True),
                              trace=False, check_finite=False, capture=False)
        report.update(session=backend.session, stage='normal_inference',
                      frame_status='normal inference started; completion not confirmed')
        save_json(directory / 'report.json', report)
        started = time.monotonic()
        actual = backend.infer_bytes(data)
        report.update(inference_seconds=time.monotonic() - started, timing=backend.last_timing,
                      output_sha256=sha(actual), frame_status='completed (7200 bytes; transport/values checked)',
                      last_board_diagnostic=backend.last_diagnostic)
        (directory / 'candidates.bin').write_bytes(actual)
        profile = normal_profile(report['timing'], backend.session, expected_version=9)
        report['normal_profile_verified'] = True
        report['slowest'] = timing_files(directory, profile)
        backend.close()
        backend = None
        report['stage'] = 'offline_golden'
        save_json(directory / 'report.json', report)  # Retain a completed frame if reference is interrupted.
        report['golden'] = engine.check(1, data, actual)
        report['status'] = 'passed' if report['golden']['status'] == 'EXACT' else 'failed'
        report['stage'] = 'finished'
    except KeyboardInterrupt:
        report.update(status='stopped', error='Interrupted; no board reset/reload/frame retry after failure')
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        log.exception(exc)
    finally:
        if backend is not None:
            try:
                backend._snapshot(time.monotonic() + args.tool_timeout)
                report['last_board_diagnostic'] = backend.last_diagnostic
            except Exception as exc:
                report['snapshot_error'] = str(exc)
            finally:
                backend.close()
        if link is not None:
            link.close()
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                report.update(status='failed', error='Golden cleanup failed: ' + str(exc))
        log.close()
        publish(directory, report)
    return int(report['status'] != 'passed')


def reanalyze(directory):
    """Finish only the CPU comparison; never connect to PCIe."""
    raw = (directory / 'report.json').read_bytes()
    report = json.loads(raw)
    if report.get('kind') != 'normal_replay' or report.get('version') != 1:
        raise ValueError('Expected a normal replay report')
    report['source_report_sha256'] = sha(raw)
    report.pop('golden', None)
    report['normal_profile_verified'] = False
    engine = None
    try:
        data = (directory / 'source-input.bin').read_bytes()
        actual = (directory / 'candidates.bin').read_bytes()
        if (len(data) != INPUT_BYTES or sha(data) != report['input_sha256'] or
                len(actual) != 7200 or sha(actual) != report['output_sha256']):
            raise ValueError('Saved input/candidate length or SHA256 mismatch')
        normal_profile(report['timing'], report['session'])
        report['normal_profile_verified'] = True
        engine = VideoGolden(directory, root=ROOT)
        report['golden'] = engine.check(1, data, actual)
        report['status'] = 'passed' if report['golden']['status'] == 'EXACT' else 'failed'
        report.pop('error', None)
    except KeyboardInterrupt:
        report.update(status='stopped', error='Offline golden interrupted; board was not accessed')
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                report.update(status='failed', error='Golden cleanup failed: ' + str(exc))
        publish(directory, report, 'analysis.json')
    return int(report['status'] != 'passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--probe-run', type=Path, help='Completed probe directory; default: latest.json')
    source.add_argument('--report', type=Path, help='Resume only offline golden from a saved normal replay')
    parser.add_argument('--load', action='store_true', help='Verify saved input and idle board, then load current firmware once')
    parser.add_argument('--reset-dir', default='/rv/fpga/bin')
    parser.add_argument('--pbcopy', default='/rv/fpga/bin/pbcopy')
    parser.add_argument('--pbload', default='/rv/fpga/bin/pbload')
    parser.add_argument('--infer-timeout', type=float, default=120)
    parser.add_argument('--tool-timeout', type=float, default=30)
    parser.add_argument('--core-mhz', type=float, default=40, help='Known N550 clock; 0 prints cycles only')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run on the Linux host connected to S2C')
    if (any(not math.isfinite(v) or v <= 0 for v in (args.infer_timeout, args.tool_timeout)) or
            not math.isfinite(args.core_mhz) or args.core_mhz < 0):
        parser.error('Positive finite timeouts and nonnegative finite core MHz required')
    from host_control import guard, workspace, save_json as pointer
    try:
        with guard(workspace() / 'control.lock'):
            if args.report:
                return reanalyze(args.report.expanduser().resolve())
            probe = args.probe_run or Path(json.loads(
                (ROOT / 'yolo26_pc/outputs/pack-probe/latest.json').read_text(encoding='utf-8'))['directory'])
            base = ROOT / 'yolo26_pc/outputs/normal-replay'
            base.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix='run-', dir=str(base)))
            pointer(base / 'latest.json', dict(directory=str(directory)))
            return execute(args, directory, probe.expanduser().resolve())
    except (OSError, ValueError, KeyError, IndexError, RuntimeError) as exc:
        print('NORMAL REPLAY FAILED:', exc, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
