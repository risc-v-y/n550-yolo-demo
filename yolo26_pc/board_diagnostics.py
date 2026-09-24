"""Versioned board snapshots and flushed host-side diagnostic logs (stdlib only)."""
from datetime import datetime, timezone
import json
import struct
import threading
import traceback

FIELDS = ('version stage frame node op phase error detail received rx_errors rx_timeouts '
          'tx_errors crc_errors protocol_errors retries amu_flags tensor element observed expected '
          'trap_cause trap_pc trap_value node_start node_cycles inputs outputs options').split()
STAGES = ('unknown boot uart amu_init selftest ready receive run done failed trap').split()
PHASES = ('none input node_begin cache compute amu_pack amu_submit amu_wait amu_store '
          'check node_end').split()
OPS = ('conv silu add sub mul copy slice concat pool nearest permute expand bmm softmax '
       'sigmoid max topk gather remainder floordiv cast').split()
WIRE = struct.Struct('<102Q')


def decode_snapshot(data):
    if len(data) != WIRE.size:
        raise ValueError(f'Invalid diagnostic length: {len(data)}')
    values = WIRE.unpack(data)
    report = dict(zip(FIELDS, values))
    if report['version'] != 1:
        raise ValueError(f"Unsupported diagnostic version: {report['version']}")
    if report['inputs'] > 4 or report['outputs'] > 2:
        raise ValueError('Invalid diagnostic tensor counts')
    if report['detail'] >= 1 << 63:
        report['detail'] -= 1 << 64
    for field in ('node', 'tensor'):
        if report[field] == (1 << 64)-1:
            report[field] = None
    for field, names in (('stage', STAGES), ('phase', PHASES), ('op', OPS)):
        value = report[field]
        report[field + '_name'] = names[value] if value < len(names) else f'unknown:{value}'
    report['tensors'] = []
    for slot in range(6):
        item = values[28+slot*7:35+slot*7]
        valid = slot < report['inputs'] if slot < 4 else slot-4 < report['outputs']
        if valid:
            if item[2] > 4:
                raise ValueError('Invalid diagnostic tensor rank')
            report['tensors'].append({'direction': 'input' if slot < 4 else 'output',
                                      'id': item[0], 'dtype': item[1], 'shape': list(item[3:3+item[2]])})
    report['registers'] = {f'x{i}': value for i, value in enumerate(values[70:])}
    return report


def describe(event):
    if event.get('type') != 'board':
        return event.get('message', str(event))
    d = event['diagnostic']
    return (f"board frame={d['frame']} stage={d['stage_name']} node={d['node']} "
            f"op={d['op_name']} phase={d['phase_name']} error=0x{d['error']:x} "
            f"detail={d['detail']} cycles={d['node_cycles']}")


class DiagnosticLog:
    def __init__(self, directory):
        self.lock = threading.Lock()
        self.stream = (directory / 'diagnostics.jsonl').open('x', encoding='utf-8')

    def emit(self, event):
        item = dict(event, utc=datetime.now(timezone.utc).isoformat())
        with self.lock:
            self.stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            self.stream.flush()
            print(describe(event), flush=True)

    def exception(self, exc):
        self.emit({'type': 'host_error', 'message': str(exc),
                   'exception': type(exc).__name__,
                   'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))})

    def close(self):
        with self.lock:
            self.stream.close()
