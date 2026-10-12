"""Read a CPU DDR range through PCIe, display little-endian words and save logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import secrets
import shutil
import sys
import time
import traceback

import collect_board_diag as diag

ROOT = Path(__file__).resolve().parents[1]


def number(value):
    try:
        return int(value, 0)
    except ValueError:
        raise argparse.ArgumentTypeError('Use a decimal number or 0x-prefixed hexadecimal number') from None


def hex_lines(data, address, width):
    """Display at most 16 bytes per line; the raw file always preserves all bytes."""
    word_bytes = width // 8
    for start in range(0, len(data), 16):
        row = data[start:start + 16]
        words = [f'{int.from_bytes(row[i:i + word_bytes], "little"):0{word_bytes * 2}x}'
                 for i in range(0, len(row), word_bytes)]
        yield f'0x{address + start:08x}: ' + ' '.join(words)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('address', type=number, help='CPU address, e.g. 0x80003a40 (not a PCIe offset)')
    parser.add_argument('size', type=number, help='Number of bytes to read, e.g. 16 or 0x10')
    parser.add_argument('--width', type=int, choices=(8, 16, 32, 64), default=32,
                        help='Display word width in bits (default: 32)')
    parser.add_argument('--clp-test', action='store_true',
                        help='Also allow the reserved CLP probe DDR 0xf0000000..0xf0000fff')
    parser.add_argument('--max-print', type=number, default=256,
                        help='Maximum bytes shown on screen and in data.txt (default: 256)')
    parser.add_argument('--pbload', help='PCIe reader executable; PATH then /rv/fpga/bin/pbload by default')
    parser.add_argument('--tool-cwd', type=Path, default=Path.cwd(),
                        help='Writable working directory where the platform license is available')
    parser.add_argument('--tool-timeout', type=float, default=30,
                        help='Timeout in seconds for one pbload invocation (default: 30)')
    parser.add_argument('--output', type=Path, help='New output directory; existing directories are refused')
    args = parser.parse_args(argv)
    if sys.platform != 'linux':
        parser.error('Run on the Linux host physically connected to S2C.')
    try:
        relative = diag.offset(args.address, args.size, clp_test=args.clp_test)
    except ValueError as exc:
        parser.error(str(exc))
    word_bytes = args.width // 8
    if args.size % word_bytes or args.max_print < word_bytes or args.max_print % word_bytes:
        parser.error('size and max-print must be multiples of width/8; max-print must be positive')
    if not math.isfinite(args.tool_timeout) or args.tool_timeout <= 0:
        parser.error('tool-timeout must be finite and positive')
    cwd = args.tool_cwd.resolve()
    if not cwd.is_dir():
        parser.error('tool-cwd must be an existing directory')
    directory = (args.output or ROOT / 'yolo26_pc/outputs/ddr-read' / ('run-' + secrets.token_hex(8))).resolve()
    if ':' in str(directory):
        parser.error('Output path must not contain a colon')
    directory.mkdir(parents=True, exist_ok=False)
    log = diag.Log(directory)
    report = {'version': 1, 'status': 'starting', 'mode': 'read_only',
              'host_time_utc': datetime.now(timezone.utc).isoformat(),
              'cpu_address': f'0x{args.address:x}', 'pcie_offset': f'0x{relative:x}',
              'requested_bytes': args.size, 'width_bits': args.width, 'tool_cwd': str(cwd),
              'clp_test_region_enabled': args.clp_test,
              'raw_file': '001-ddr.bin', 'tool_log': '001-ddr.log',
              'notes': ['A successful read is not a board selftest pass.',
                        'Live DDR reads are not atomic; halt the CPU when comparing test stages.',
                        'Only the displayed preview is saved in data.txt; the BIN contains the full read.']}
    code = 1
    try:
        command = args.pbload or shutil.which('pbload')
        if command is None:
            command = '/rv/fpga/bin/pbload' if Path('/rv/fpga/bin/pbload').is_file() else 'pbload'
        reader = diag.Reader(command, cwd, args.tool_timeout, log, clp_test=args.clp_test)
        report['pbload'] = reader.command
        with diag.link_lock():
            data = reader.read(args.address, args.size, 'ddr')
        preview = data[:args.max_print]
        text = '\n'.join(hex_lines(preview, args.address, args.width)) + '\n'
        (directory / 'data.txt').write_text(text, encoding='utf-8')
        report.update(status='read', received_bytes=len(data), preview_bytes=len(preview),
                      all_zero=not any(data), sha256=hashlib.sha256(data).hexdigest())
        log.emit({'type': 'read_verified', 'bytes': len(data), 'all_zero': report['all_zero'],
                  'sha256': report['sha256']})
        print(f'CPU 0x{args.address:x} -> PCIe 0x{relative:x}; {len(data)} bytes; u{args.width} little-endian')
        print(text, end='')
        if len(preview) < len(data):
            print(f'Preview: first {len(preview)} of {len(data)} bytes; full data in 001-ddr.bin')
        print(f'All bytes zero: {report["all_zero"]}')
        code = 0
    except BaseException as exc:
        report.update(status='failed', error=str(exc), exception=type(exc).__name__)
        log.emit({'type': 'read_error', 'error': str(exc), 'exception': type(exc).__name__,
                  'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))})
        print(f'DDR read failed: {exc}', file=sys.stderr, flush=True)
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        raw = directory / report['raw_file']
        report['received_bytes'] = raw.stat().st_size if raw.exists() else 0
        report['elapsed_seconds'] = time.monotonic() - log.started
        (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        log.stream.close()
        print(f'Saved: {directory}', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
