"""Read-only N550 DDR diagnostics on the Linux host; Python standard library only."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'yolo26_pc'))
from board_diagnostics import decode_snapshot
import serial_probe

BASE, END = 0x80000000, 0x82000000
PREFIX_BYTES = 256
SYMBOLS = {
    'board_uart_status': '<i', 'board_uart_baud': '<I', 'board_uart_dlf_bits': '<I',
    'board_test_stage': '<I', 'board_error': '<Q', 'board_trap_cause': '<Q',
    'board_trap_pc': '<Q', 'board_trap_value': '<Q', 'board_diag': None,
}
UART_STATUS = {0: 'ready', -1: 'busy_timeout', -2: 'dlf_mask_invalid',
               -3: 'lcr_dlab_verify_failed', -4: 'divisor_verify_failed',
               -5: 'lcr_8n1_verify_failed', -6: 'not_initialized', -7: 'tx_timeout'}
TEST_STAGES = {0: 'not_started', 1: 'rvv_cache', 2: 'amu', 4: 'buffer_reuse', 3: 'passed'}
FAILURE = re.compile(r'(?i)(?:\b(?:failed|failure)\b|\bfail!|\[x\]\s*ERROR|EXIT_WITH_NO_LICENSE)')


def size_of(name):
    return struct.calcsize(SYMBOLS[name]) if SYMBOLS[name] else 816


def offset(address, size):
    if size <= 0 or address < BASE or address + size > END:
        raise ValueError(f'Read outside reserved DDR: 0x{address:x}, {size} bytes')
    return address - BASE


def elf_symbols(data):
    """Parse ELF64 little-endian RISC-V symbols without a cross toolchain."""
    def region(start, size):
        if start < 0 or size < 0 or start + size > len(data):
            raise ValueError('Truncated ELF table')
        return data[start:start + size]

    header = struct.unpack('<16sHHIQQQIHHHHHH', region(0, 64))
    ident, kind, machine, version, entry = header[:5]
    if ident[:7] != b'\x7fELF\x02\x01\x01' or (kind, machine, version) != (2, 243, 1):
        raise ValueError('Expected an executable ELF64 little-endian RISC-V image')
    if entry != BASE:
        raise ValueError('Unexpected ELF entry point; expected 0x80000000')
    section_offset, section_size, section_count = header[6], header[11], header[12]
    if section_size != 64 or not section_count:
        raise ValueError('ELF section headers/symbols unavailable')
    sections = [struct.unpack('<IIQQQQIIQQ', region(section_offset + i * 64, 64))
                for i in range(section_count)]
    found = {}
    for section in sections:
        if section[1] != 2:  # SHT_SYMTAB; do not accept stripped ELF files.
            continue
        if section[9] != 24 or section[5] % 24 or section[6] >= section_count:
            raise ValueError('Invalid ELF symbol table')
        strings = sections[section[6]]
        if strings[1] != 3:
            raise ValueError('Invalid ELF string table')
        names = region(strings[4], strings[5])
        table = region(section[4], section[5])
        for name_at, info, unused, index, address, size in struct.iter_unpack('<IBBHQQ', table):
            if not index:
                continue
            end = names.find(b'\0', name_at)
            if name_at >= len(names) or end < 0:
                raise ValueError('Invalid ELF symbol name')
            name = names[name_at:end].decode('ascii', errors='replace')
            if name not in SYMBOLS:
                continue
            if info & 15 != 1 or size < size_of(name):  # STT_OBJECT
                raise ValueError(f'Invalid object symbol: {name}')
            offset(address, size_of(name))
            if name in found and found[name] != address:
                raise ValueError(f'Conflicting ELF symbol: {name}')
            found[name] = address
    missing = SYMBOLS.keys() - found.keys()
    if missing:
        raise ValueError('Missing ELF symbols: ' + ', '.join(sorted(missing)))
    if found['board_diag'] % 64:
        raise ValueError('Diagnostic symbol is not 64-byte aligned')
    return found


def firmware_info(root, name):
    sums = {}
    for line in (root / 'firmware/SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, path = line.split(maxsplit=1)
        path = path.lstrip('*')
        if path in sums:
            raise ValueError(f'Duplicate checksum entry: {path}')
        sums[path] = digest
    blobs, hashes = {}, {}
    for suffix in ('elf', 'bin'):
        relative = f'firmware/{name}.{suffix}'
        blob = (root / relative).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if sums.get(relative) != digest:
            raise ValueError(f'Firmware checksum mismatch/missing: {relative}')
        blobs[suffix], hashes[suffix] = blob, digest
    if len(blobs['bin']) < PREFIX_BYTES:
        raise ValueError('Firmware BIN too short')
    symbols = elf_symbols(blobs['elf'])
    return {'name': name, 'sha256': hashes,
            'symbols': {key: f'0x{value:x}' for key, value in symbols.items()}}, symbols, blobs['bin'][:PREFIX_BYTES]


class Log:
    def __init__(self, directory):
        self.directory = directory
        self.started = time.monotonic()
        self.stream = (directory / 'diagnostics.jsonl').open('x', encoding='utf-8')

    def emit(self, event):
        event = dict(event, elapsed_seconds=time.monotonic() - self.started)
        self.stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        self.stream.flush()


@contextmanager
def link_lock():
    import fcntl
    with (Path(tempfile.gettempdir()) / 'yolo26-pcie.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError('Another host process owns the YOLO PCIe link') from None
        yield


class Reader:
    def __init__(self, command, cwd, timeout, log):
        command = shutil.which(command)
        if command is None:
            raise FileNotFoundError('pbload executable not found; use --pbload /absolute/path/pbload')
        self.command = str(Path(command).resolve())
        self.cwd, self.timeout, self.log = cwd, timeout, log
        self.number = 0

    def read(self, address, size, label):
        relative = offset(address, size)
        self.number += 1
        prefix = f'{self.number:03d}-{label}'
        destination = self.log.directory / (prefix + '.bin')
        # The deployed S2C pbload aborts with long output paths. Pass only a
        # short basename in the license working directory, then archive it.
        # mkstemp creates a unique empty file, so stale data is never accepted.
        handle, temporary = tempfile.mkstemp(prefix='p', suffix='.bin', dir=self.cwd)
        os.close(handle)
        transfer = Path(temporary)
        args = [self.command, '-d', f'{transfer.name}:0x{relative:x}:0x{size:x}']
        try:
            self.log.emit({'type': 'read_begin', 'command': args, 'cwd': str(self.cwd),
                           'cpu_address': f'0x{address:x}', 'bytes': size,
                           'archive': destination.name})
            with (self.log.directory / (prefix + '.log')).open('wb') as output:
                process = subprocess.Popen(args, cwd=self.cwd, stdout=output, stderr=subprocess.STDOUT)
                try:
                    process.wait(timeout=self.timeout)  # Monotonic on supported Python/Linux.
                except BaseException:
                    process.kill()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass  # A blocked platform driver may require manual intervention.
                    self.log.emit({'type': 'read_failed', 'label': label, 'reason': 'timeout_or_interruption'})
                    raise
        finally:
            # Preserve partial data on failure/timeout too. Moving works across
            # filesystems and leaves no transfer files in the working directory.
            if transfer.exists():
                shutil.move(str(transfer), str(destination))
        text = (self.log.directory / (prefix + '.log')).read_text(encoding='utf-8', errors='replace')
        self.log.emit({'type': 'read_end', 'label': label, 'returncode': process.returncode,
                       'tool_log': prefix + '.log'})
        if process.returncode or FAILURE.search(text):
            reason = f'exit {process.returncode}'
            if process.returncode < 0:
                reason += f', signal {signal.Signals(-process.returncode).name}'
            raise RuntimeError(f'pbload failed for {label}; see {prefix}.log ({reason})')
        if not destination.exists() or destination.stat().st_size != size:
            raise ValueError(f'pbload missing/wrong output length for {label}; expected {size} bytes')
        return destination.read_bytes()


def collect(reader, symbols, expected_prefix, samples, interval, log):
    actual = reader.read(BASE, len(expected_prefix), 'program-prefix')
    if actual != expected_prefix:
        raise ValueError('DDR program prefix differs from selected BIN; no diagnostic symbols were read')
    log.emit({'type': 'program_prefix_matches', 'bytes': len(actual)})
    observations = []
    for sample in range(samples):
        item = {'sample': sample + 1, 'started_seconds': time.monotonic() - log.started,
                'values': {}, 'read_errors': {}}
        for name, fmt in SYMBOLS.items():
            try:
                blob = reader.read(symbols[name], size_of(name), f's{sample + 1}-{name}')
                if fmt:
                    item['values'][name] = struct.unpack(fmt, blob)[0]
                else:
                    item['diagnostic'] = decode_snapshot(blob)
            except Exception as exc:
                item['read_errors'][name] = str(exc)
                log.emit({'type': 'symbol_error', 'sample': sample + 1, 'symbol': name, 'error': str(exc)})
                # A failed link must not trigger dozens of repeated license/timeouts.
                if not isinstance(exc, ValueError):
                    break
        uart = item['values'].get('board_uart_status')
        test = item['values'].get('board_test_stage')
        item['uart_status_name'] = UART_STATUS.get(uart, f'unknown:{uart}')
        item['test_stage_name'] = TEST_STAGES.get(test, f'unknown:{test}')
        item['finished_seconds'] = time.monotonic() - log.started
        observations.append(item)
        log.emit({'type': 'snapshot', **item})
        print(f"Sample {sample + 1}: UART={uart} ({item['uart_status_name']}), "
              f"selftest={test} ({item['test_stage_name']}), "
              f"stage={item.get('diagnostic', {}).get('stage_name', 'unavailable')}", flush=True)
        if item['read_errors']:
            break
        if sample + 1 < samples:
            time.sleep(interval)
    return observations


def git_metadata(root):
    result = {}
    for key, args in (('commit', ['rev-parse', 'HEAD']),
                      ('tracked_changes', ['status', '--porcelain', '--untracked-files=no'])):
        try:
            command = subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                                     text=True, timeout=5)
            result[key] = command.stdout.strip() if command.returncode == 0 else 'unavailable'
        except (OSError, subprocess.TimeoutExpired):
            result[key] = 'unavailable'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firmware', required=True, choices=('selftest', 'demo-pcie'),
                        help='Must match the BIN currently loaded; never resets or reloads the board')
    parser.add_argument('--pbload', default='pbload')
    parser.add_argument('--tool-cwd', type=Path, default=Path.cwd(),
                        help='Working directory used by pbload for its platform/license dependencies')
    parser.add_argument('--tool-timeout', type=float, default=30)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--load-log', type=Path, help='Exact load.log for this deployment (not auto-selected by clock)')
    parser.add_argument('--uart-log', type=Path, help='Existing terminal capture; serial devices are never opened')
    parser.add_argument('--output', type=Path, help='New output directory; default uses a random run ID')
    args = parser.parse_args(argv)
    if sys.platform != 'linux':
        parser.error('Run on the Linux host physically connected to S2C.')
    if (args.samples < 1 or not math.isfinite(args.tool_timeout) or args.tool_timeout <= 0
            or not math.isfinite(args.interval) or args.interval < 0):
        parser.error('samples >= 1, finite tool-timeout > 0 and finite interval >= 0 are required')
    directory = (args.output or ROOT / 'yolo26_pc/outputs/host-diagnostics' / ('run-' + secrets.token_hex(8))).resolve()
    if ':' in str(directory):
        parser.error('Output path must not contain a colon (pbload argument separator)')
    directory.mkdir(parents=True, exist_ok=False)
    log = Log(directory)
    report = {'version': 1, 'status': 'starting', 'mode': 'read_only',
              'host_time_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'git': git_metadata(ROOT), 'tool_cwd': str(args.tool_cwd.resolve()),
              'requested_samples': args.samples,
              'notes': ['Host wall clock may be intentionally backdated; durations use monotonic time.',
                        'Reads are live, sequential observations, not an atomic CPU snapshot.',
                        '256-byte DDR prefix check is not a full firmware/weights verification.',
                        'Collection success does not mean the board or model passed validation.'],
              'attachments': {}, 'warnings': []}
    code = 1
    try:
        for name, path in (('load.log', args.load_log), ('uart.log', args.uart_log)):
            if path is not None:
                try:
                    shutil.copyfile(path, directory / name)
                    report['attachments'][name] = str(path.resolve())
                except OSError as exc:
                    report['warnings'].append(f'Cannot copy {name}: {exc}')
        try:
            ports = serial_probe.discover()
            restricted = serial_probe.add_holders(ports)
            report['serial'] = {'ports': ports, 'restricted_inspections': restricted,
                                'device_opened': False}
        except OSError as exc:
            report['warnings'].append(f'Serial enumeration failed: {exc}')
        report['firmware'], symbols, expected_prefix = firmware_info(ROOT, args.firmware)
        reader = Reader(args.pbload, args.tool_cwd.resolve(), args.tool_timeout, log)
        report['pbload'] = reader.command
        with link_lock():
            report['snapshots'] = collect(reader, symbols, expected_prefix, args.samples, args.interval, log)
        errors = any(item['read_errors'] for item in report['snapshots'])
        report['status'], code = ('partial', 1) if errors else ('collected', 0)
    except BaseException as exc:
        report.update(status='failed', error=str(exc), exception=type(exc).__name__)
        log.emit({'type': 'collector_error', 'error': str(exc), 'exception': type(exc).__name__,
                  'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))})
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        report['elapsed_seconds'] = time.monotonic() - log.started
        (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        log.stream.close()
        print(f"Collection {report['status']}. Logs: {directory}", flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
