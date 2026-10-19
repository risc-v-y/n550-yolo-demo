"""Three offline timing pages. No board access, graphics or ML dependencies."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'yolo26_pc'))
from board_profile import validate_saved_profile
from board_diagnostics import OPS


def analyze(directory, core_mhz=0):
    if not math.isfinite(core_mhz) or core_mhz < 0:
        raise ValueError('Core MHz must be zero (unknown) or positive')
    report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
    records = report['results']
    timings = {}
    for key in ('prepare_seconds', 'postprocess_seconds', 'elapsed_seconds',
                'upload_seconds', 'run_roundtrip_seconds', 'download_seconds', 'profile_read_seconds'):
        values = [r.get(key, r.get('pcie_timing', {}).get(key)) for r in records]
        values = [v for v in values if isinstance(v, (float, int)) and math.isfinite(v) and v >= 0]
        timings[key] = dict(mean=statistics.mean(values), max=max(values), count=len(values)) if values else None
    profiles, excluded, errors = [], [], []
    for record in records:
        timing = record.get('pcie_timing', {})
        if not timing.get('board_profile'):
            continue
        try:
            profile = validate_saved_profile(timing['board_profile'], timing['pcie_frame'])
            if profile['representative']:
                profiles.append(profile)
            else:
                excluded.append(record['frame_id'])
        except (ValueError, TypeError, KeyError, OverflowError, struct.error) as exc:
            errors.append(dict(frame=record['frame_id'], error=str(exc)))
    tools = {}
    log = directory/'diagnostics.jsonl'
    if log.is_file():
        with log.open(encoding='utf-8') as stream:
            for line in stream:
                try: event = json.loads(line)
                except ValueError: continue  # Interrupted final log line.
                if event.get('type') != 'pcie_tool': continue
                key = event.get('message', '').split(':', 1)[0]
                seconds = event.get('elapsed_seconds')
                if key not in ('pbcopy', 'pbload') or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                    continue
                item = tools.setdefault(key, dict(calls=0, seconds=0., failed=0))
                item['calls'] += 1; item['seconds'] += seconds
                item['failed'] += int(event.get('returncode') != 0)
    manifest = json.loads((ROOT/'reference/nodes/manifest.json').read_text(encoding='utf-8'))
    ops = {n['id']: OPS[n['op']] for n in manifest['nodes']}
    nodes = []
    for i in range(334) if profiles else ():
        row = dict(node=i, op=ops[i])
        for key in ('cache', 'compute', 'check', 'total'):
            row[key] = statistics.mean(p['nodes_detail'][i][key] for p in profiles)
        nodes.append(row)
    cycles = {key: statistics.mean(p[key] for p in profiles) for key in
              ('service_cycles', 'model_cycles', 'graph_cycles', 'input_check_cycles',
               'input_crc_cycles', 'output_cycles', 'graph_overhead_cycles')} if profiles else {}
    for key in ('cache', 'compute', 'check'):
        if nodes: cycles[key] = sum(n[key] for n in nodes)
    result = dict(version=1, run=report.get('id', directory.name), frames=len(records),
                  timings=timings, profiled=len(profiles), excluded=excluded, errors=errors,
                  cycles=cycles, core_mhz=core_mhz, nodes=sorted(nodes, key=lambda n: -n['total']))
    result['tools'] = tools
    if profiles and core_mhz:
        result['model_seconds'] = cycles['model_cycles']/(core_mhz*1e6)
        result['service_seconds'] = cycles['service_cycles']/(core_mhz*1e6)
    (directory/'performance-summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    with (directory/'node-timing.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=('node', 'op', 'cache', 'compute', 'check', 'total'))
        writer.writeheader(); writer.writerows(result['nodes'])
    for page in (1, 2, 3):
        (directory/'performance-page-{}.txt'.format(page)).write_text(render(result, page), encoding='utf-8')
    return result


def render(report, page):
    lines = ['PERFORMANCE {} PAGE={}/3'.format(report['run'][-35:], page)]
    if page == 1:
        lines += ['Completed frames={} | CPU-profiled={}'.format(report['frames'], report['profiled']),
                  'Host timing, seconds: mean / max / count']
        for key, value in report['timings'].items():
            lines.append('{}: {}'.format(key, '{:.4f} / {:.4f} / {}'.format(
                value['mean'], value['max'], value['count']) if value else 'not recorded'))
        lines += ['PCIe wait includes polling and tool startup.',
                  'Frame elapsed includes recording; not capture-to-visible latency.',
                  'First-frame setup is included; offline golden is excluded.',
                  'Old runs without board counters still have host timings.']
        for name, item in report.get('tools', {}).items():
            lines.append('{}: {} calls / {:.3f}s / {} failed (whole run)'.format(
                name, item['calls'], item['seconds'], item['failed']))
    elif page == 2:
        lines += ['CPU counter means (cycles); only normal runs are ranked.']
        for key, value in report['cycles'].items(): lines.append('{}: {:.0f}'.format(key, value))
        if report['core_mhz'] and report['profiled']:
            lines += ['Core MHz: {}'.format(report['core_mhz']),
                      'Model {:.6f}s | board service {:.6f}s'.format(report['model_seconds'], report['service_seconds'])]
        else: lines += ['Core frequency unknown: seconds/FPS are not inferred.']
        lines += ['Service: request accepted -> before timing/status publication.',
                  'Graph includes diagnostics; node sums exclude progress logging.',
                  'Excluded trace/finite/capture frames: {}'.format(len(report['excluded'])),
                  'Invalid timing records: {}'.format(len(report['errors']))]
    elif page == 3:
        lines += ['Slowest nodes: mean cycles; share of graph wall cycles',
                  'node op           total       %    cache     compute']
        graph = report['cycles'].get('graph_cycles', 0)
        for row in report['nodes'][:10]:
            lines.append('{:3d} {:10.10s} {:11.0f} {:5.1f} {:10.0f} {:11.0f}'.format(
                row['node'], str(row['op']), row['total'], 100*row['total']/graph if graph else 0,
                row['cache'], row['compute']))
        if not report['nodes']: lines.append('No normal board profile. Run updated video firmware once.')
        lines += ['All 334 nodes and check cycles: node-timing.csv',
                  'Counters and reads add overhead; no speedup is claimed yet.']
        for error in report['errors'][:2]: lines.append('ERROR frame={}: {}'.format(error['frame'], error['error']))
    else: raise ValueError('Page must be 1, 2 or 3')
    return '\n'.join(line[:80] for line in lines[:22])+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--core-mhz', type=float, default=0)
    args = parser.parse_args()
    result = analyze(args.directory.resolve(), args.core_mhz)
    for page in (1, 2, 3): print(render(result, page), end='')
    return 1 if result['errors'] else 0


if __name__ == '__main__': sys.exit(main())
